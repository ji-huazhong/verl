# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Original-forward Megatron/vLLM snapshots with explicit token coordinates.

Diagnostic hooks do not add collectives or change model arithmetic. Sequence
parallel pieces are joined on CPU after execution. Only DP replica zero is
retained; vLLM's replicated activations are retained on TP rank zero.
"""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from types import MethodType

import torch


class BackendPrefixTrace:
    def __init__(
        self, directory, backend, rank, tp_rank, tp_size, *, enabled=True, tokens=32, token_start=0, trace_qsa=False
    ):
        if tokens < 1 or token_start < 0:
            raise ValueError("Trace window must have a nonnegative start and positive length")
        self.directory = Path(directory) / f"rank-{rank:05d}"
        self.backend, self.rank = backend, rank
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.enabled, self.tokens = enabled, tokens
        self.token_start, self.trace_qsa = token_start, trace_qsa
        self.default_token_start = token_start
        self.current = None
        self.active_qsa = None
        self.active_gdn = None
        self.qsa_layers = {}
        self.vllm_window = None
        self.vllm_position_hooks = False
        self.forward_chunks = []
        self.values, self.records, self.handles, self.methods = {}, [], [], []
        self.bytes = 0
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=False)

    def start(self, prompt, padded_length, config_sha256):
        if self.current is not None or self.values:
            raise RuntimeError("Previous backend trace has not been drained")
        token_start = prompt.get("trace_token_start", self.default_token_start)
        if type(token_start) is not int or token_start < 0:
            raise ValueError("Per-prompt trace start must be a nonnegative integer")
        self.token_start = token_start
        if self.token_start >= len(prompt["input_ids"]):
            raise ValueError("Trace window starts beyond the prompt")
        query_length = prompt.get("trace_query_length", len(prompt["input_ids"]))
        if type(query_length) is not int or not 1 <= query_length <= len(prompt["input_ids"]):
            raise ValueError("Invalid expected query coverage")
        if self.backend != "vllm" and query_length != len(prompt["input_ids"]):
            raise ValueError("Partial query coverage is only supported for vLLM decode")
        if min(self.token_start + self.tokens, len(prompt["input_ids"])) > query_length:
            raise ValueError("Trace window extends beyond expected query coverage")
        self.current = dict(
            prompt, padded_length=padded_length, config_sha256=config_sha256, trace_query_length=query_length
        )
        self.forward_chunks = []

    def capture(self, name, value, *, all_tp=False):
        if not self.enabled or self.current is None:
            return
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor backend activation: {name}")
        if value.ndim == 3 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 2:
            raise ValueError(f"Expected a single token stream at {name}: {tuple(value.shape)}")
        if all_tp:
            name += f"/tp-{self.tp_rank:02d}"
        padded = self.current["padded_length"]
        if self.backend == "vllm" and self.vllm_window is not None:
            offset, count = self.vllm_window
            if value.shape[0] != count:
                raise ValueError(f"Activation rows differ from forward positions at {name}")
            if self.tp_rank != 0 and not all_tp:
                return
        elif self.backend == "megatron" and value.shape[0] == padded // self.tp_size:
            offset = self.tp_rank * value.shape[0]
        elif value.shape[0] >= len(self.current["input_ids"]):
            if self.tp_rank != 0 and not all_tp:
                return
            offset = 0
        else:
            raise ValueError(f"Unknown token coordinates at {name}: {tuple(value.shape)}, padded={padded}")
        begin = max(self.token_start, offset)
        end = min(self.token_start + self.tokens, len(self.current["input_ids"]), offset + value.shape[0])
        if end <= begin:
            return
        prefix = value[begin - offset : end - offset].detach().clone()
        self._store_window(name, prefix, begin)

    def _store_window(self, name, prefix, begin):
        if name in self.values:
            previous = self.values[name]
            if self.vllm_window is None or begin != previous["offset"] + previous["rows"]:
                raise RuntimeError(f"Multiple forwards or overlapping token coordinates at {name}")
            previous["pieces"].append(prefix)
            previous["rows"] += prefix.shape[0]
        else:
            self.values[name] = dict(offset=begin, pieces=[prefix], rows=prefix.shape[0])
        self.bytes += prefix.numel() * prefix.element_size()
        if self.bytes > 512 * 1024**2:
            raise RuntimeError("Backend prefix trace exceeded 512 MiB per rank")

    def attach_vllm_positions(self, model):
        """Read actual absolute query positions, including hybrid cache boundary splits."""
        original = model.forward
        self.vllm_position_hooks = True

        def forward(_model, *args, **kwargs):
            if self.current is None:
                return original(*args, **kwargs)
            if self.vllm_window is not None:
                raise RuntimeError("Nested traced vLLM forwards")
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            if not isinstance(positions, torch.Tensor):
                raise ValueError("Missing absolute vLLM query positions")
            positions = positions.detach().cpu()
            if positions.ndim == 2:
                if not torch.equal(positions, positions[:1].expand_as(positions)):
                    raise ValueError("Expected text-only vLLM positions")
                positions = positions[0]
            if positions.ndim != 1 or not positions.numel():
                raise ValueError("Expected one nonempty vLLM token stream")
            start, count = int(positions[0]), positions.numel()
            expected_start = sum(chunk["tokens"] for chunk in self.forward_chunks)
            if start != expected_start or start + count > self.current["trace_query_length"]:
                raise ValueError("Missing, overlapping, cached or decoded vLLM token coordinates")
            if not torch.equal(positions, torch.arange(start, start + count, dtype=positions.dtype)):
                raise ValueError("Noncontiguous vLLM query positions")
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if (
                input_ids is not None
                and input_ids.detach().cpu().tolist() != self.current["input_ids"][start : start + count]
            ):
                raise ValueError("vLLM traced token IDs differ from the fixed sample")
            self.vllm_window = (start, count)
            try:
                result = original(*args, **kwargs)
                self.forward_chunks.append(dict(start=start, tokens=count))
                return result
            finally:
                self.vllm_window = None

        self.methods.append((model, "forward", original))
        model.forward = MethodType(forward, model)

    def output_hook(self, name, module, *, all_tp=False):
        def hook(_module, _inputs, output):
            self.capture(name, output, all_tp=all_tp)

        self.handles.append(module.register_forward_hook(hook))

    def method_hook(self, module, method, capture, capture_input=None):
        original = getattr(module, method)

        def wrapped(_module, *args, **kwargs):
            if capture_input is not None:
                capture_input(args, kwargs)
            output = original(*args, **kwargs)
            capture(output)
            return output

        self.methods.append((module, method, original))
        setattr(module, method, MethodType(wrapped, module))

    def capture_gdn_partition(self, name, value, *, batch_first=False, flatten_head_rows=False):
        if not self.enabled or self.current is None:
            return
        if isinstance(value, tuple):
            value = value[0]
        if batch_first:
            if value.shape[0] != 1:
                raise ValueError("GDN diagnostic requires one packed token stream")
            value = value[0]
        padded = self.current["padded_length"]
        if self.backend == "vllm" and self.vllm_window is not None:
            padded = self.vllm_window[1]
        if flatten_head_rows:
            if value.ndim != 2 or value.shape[0] % padded:
                raise ValueError("Invalid flattened GDN norm coordinates")
        elif value.shape[0] != padded:
            raise ValueError("Incomplete GDN token coordinates")
        # Native gated norm flattens token and head axes.
        value = value.reshape(padded, -1)
        if self.backend == "vllm":
            self.capture(name, value, all_tp=True)
            return
        begin = self.token_start
        end = min(begin + self.tokens, len(self.current["input_ids"]))
        self._store_window(f"{name}/tp-{self.tp_rank:02d}", value[begin:end].detach().clone(), begin)

    def attach_gdn_detail(self, module, name):
        prefix = name + "/gdn"
        self.handles.append(
            module.in_proj.register_forward_pre_hook(lambda _module, args: self.capture(prefix + "/input", args[0]))
        )
        self.handles.append(
            module.in_proj.register_forward_hook(
                lambda _module, _args, output: self.capture_gdn_partition(prefix + "/projection", output)
            )
        )
        for part in ("qkvz", "ba"):
            self.handles.append(
                getattr(module.in_proj, part).register_forward_hook(
                    lambda _module, _args, output, part=part: self.capture_gdn_partition(
                        prefix + "/projection/" + part, output
                    )
                )
            )

        def prepared(output):
            for part, value in zip(("q", "k", "v", "z", "b", "a"), output, strict=True):
                self.capture_gdn_partition(prefix + "/prepared/" + part, value, batch_first=True)

        self.method_hook(
            module,
            "_prepare_input_for_gated_delta_rule",
            prepared,
            lambda args, _kwargs: self.capture_gdn_partition(prefix + "/conv", args[0], batch_first=True),
        )

        def gates(output):
            self.capture_gdn_partition(prefix + "/g", output[0], batch_first=True)
            self.capture_gdn_partition(prefix + "/beta", output[1], batch_first=True)

        self.method_hook(module, "_compute_g_and_beta", gates)
        self.method_hook(
            module,
            "gated_delta_rule",
            lambda output: self.capture_gdn_partition(prefix + "/core", output[0], batch_first=True),
        )
        self.method_hook(
            module,
            "_apply_gated_norm",
            lambda output: self.capture_gdn_partition(prefix + "/norm_gate", output, flatten_head_rows=True),
        )
        self.handles.append(
            module.out_proj.register_forward_pre_hook(
                lambda _module, args: self.capture_gdn_partition(prefix + "/out_proj_input", args[0])
            )
        )

    def attach_megatron(self, model, gdn_layers=()):
        decoder = model.language_model.decoder
        for layer in decoder.layers:
            name = f"layers/{layer.layer_number - 1:02d}"
            if layer.layer_number - 1 in gdn_layers:
                if type(layer.self_attention).__name__ != "Qwen38NextGatedDeltaNet":
                    raise ValueError(f"Requested detailed GDN trace on non-GDN layer: {name}")
                self.attach_gdn_detail(layer.self_attention, name)
            for site, attr in (("attn", "self_attention_hyper_connection"), ("mlp", "mlp_hyper_connection")):
                prefix = f"{name}/{site}_hc"

                def hook(_module, _args, output, prefix=prefix):
                    # PLE has already been applied to this returned residual.
                    self.capture(prefix + "/input", output[3])
                    self.capture(prefix + "/mixed", output[0])

                self.handles.append(getattr(layer, attr).register_forward_hook(hook))
            self.output_hook(name + "/attention/output", layer.self_attention)
            if self.trace_qsa and type(layer.self_attention).__name__ == "Qwen38NextAttention":
                attention = layer.self_attention
                indexer = attention.indexer
                self.output_hook(name + "/qsa/index_qk", indexer.index_qk_proj, all_tp=True)
                self.qsa_layers[name] = dict(
                    token_topk=indexer.token_topk,
                    compress_ratio=indexer.compress_ratio,
                    indexer_dtype=str(indexer.q_layernorm.dtype),
                    head_dim=indexer.head_dim,
                    score_scale=1 / math.sqrt(indexer.head_dim),
                    selection_format="token_indices_with_negative_padding",
                )
                self.method_hook(
                    indexer,
                    "score_blocks",
                    lambda output, name=name: self.capture(name + "/qsa/scores", output, all_tp=True),
                )

                def selection_hook(_module, _args, attention=attention, name=name):
                    self.capture(name + "/qsa/selection", attention._qsa_selection, all_tp=True)

                self.handles.append(attention.core_attention.register_forward_pre_hook(selection_hook))
            self.output_hook(name + "/mlp/output", layer.mlp)
            self.method_hook(
                layer.mlp.router, "gating", lambda output, name=name: self.capture(name + "/router/logits", output)
            )
            ple = getattr(layer.self_attention_hyper_connection, "ple", None)
            if ple is not None:
                self.output_hook(name + "/ple/output", ple)
        mixer = getattr(decoder, "final_layernorm", None)
        if mixer is not None:
            self.handles.append(
                mixer.register_forward_pre_hook(lambda _module, args: self.capture("final_mixer/input", args[0]))
            )
            self.output_hook("final_mixer/output", mixer)

    def attach_vllm_gdn_detail(self, module, name, ops):
        """Observe native GDN calls, including chunks, without replacing kernels."""
        if module.gqa_interleaved_layout or module.enable_fused_gdn_spec_decode:
            raise ValueError("Detailed trace requires native non-interleaved, non-speculative GDN")
        prefix = name + "/gdn"
        self.method_hook(
            module,
            "forward",
            lambda _output: setattr(self, "active_gdn", None),
            lambda _args, _kwargs: setattr(self, "active_gdn", prefix),
        )
        self.handles.append(
            module.in_proj_qkvz.register_forward_pre_hook(
                lambda _module, args: self.capture(prefix + "/input", args[0])
            )
        )

        def qkvz_projection(_module, _args, output):
            value = output[0]
            self.capture_gdn_partition(prefix + "/projection/qkvz", value)
            qkv_size = (module.key_dim * 2 + module.value_dim) // module.tp_size
            self.capture_gdn_partition(prefix + "/prepared/z", value[:, qkv_size:])

        def ba_projection(_module, _args, output):
            value = output[0]
            self.capture_gdn_partition(prefix + "/projection/ba", value)
            for part, tensor in zip(("b", "a"), value.chunk(2, dim=-1), strict=True):
                self.capture_gdn_partition(prefix + "/prepared/" + part, tensor)

        self.handles.append(module.in_proj_qkvz.register_forward_hook(qkvz_projection))
        self.handles.append(module.in_proj_ba.register_forward_hook(ba_projection))

        def conv(output):
            if self.active_gdn == prefix:
                self.capture_gdn_partition(prefix + "/conv", output.transpose(0, 1))

        def prepared(output):
            if self.active_gdn != prefix:
                return
            q, k, v, g, beta = output
            # Training repeats key heads before its recurrent kernel. Repeat
            # only the detached diagnostic view to compare semantic head order.
            repeat = module.num_v_heads // module.num_k_heads
            for part, tensor in (("q", q), ("k", k)):
                self.capture_gdn_partition(
                    prefix + "/prepared/" + part, tensor.detach().repeat_interleave(repeat, dim=1)
                )
            self.capture_gdn_partition(prefix + "/prepared/v", v)
            self.capture_gdn_partition(prefix + "/g", g)
            self.capture_gdn_partition(prefix + "/beta", beta)

        self.method_hook(ops, "causal_conv1d_fn", conv)
        self.method_hook(ops, "fused_post_conv_prep", prepared)
        self.handles.append(
            module.chunk_gated_delta_rule.register_forward_hook(
                lambda _module, _args, output: self.capture_gdn_partition(prefix + "/core", output[0], batch_first=True)
            )
        )
        self.handles.append(
            module.norm.register_forward_hook(
                lambda _module, _args, output: self.capture_gdn_partition(prefix + "/norm_gate", output)
            )
        )
        self.handles.append(
            module.out_proj.register_forward_pre_hook(
                lambda _module, args: self.capture_gdn_partition(prefix + "/out_proj_input", args[0])
            )
        )

    def attach_vllm(self, model, gdn_layers=()):
        layers = [module for module in model.modules() if type(module).__name__ == "Qwen4ExpDecoderLayer"]
        if not layers:
            raise ValueError("Expected vLLM's native Qwen4Exp decoder")
        self.attach_vllm_positions(model)
        for layer in layers:
            name = f"layers/{layer.layer_idx:02d}"
            for site in ("attn", "mlp"):
                hc = getattr(layer, site + "_hyper_connection")
                self._vllm_hc(hc, f"{name}/{site}_hc/input", f"{name}/{site}_hc/mixed")
            attention = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
            if layer.layer_idx in gdn_layers:
                if layer.layer_type != "linear_attention":
                    raise ValueError(f"Requested detailed vLLM GDN trace on non-GDN layer: {name}")
                from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

                self.attach_vllm_gdn_detail(attention, name, qwen_gdn_linear_attn)
            self.output_hook(name + "/attention/output", attention)
            self.output_hook(name + "/mlp/output", layer.mlp)
            self.output_hook(name + "/router/logits", layer.mlp.gate)
            if self.trace_qsa and layer.layer_type != "linear_attention":
                self.output_hook(name + "/qsa/index_qk", attention.indexer.index_qk_proj)
                # The last column is a valid-entry count, not a token index.
                # Clone immediately: this buffer is reused by later requests.
                self.output_hook(name + "/qsa/selection", attention.indexer, all_tp=True)
                self.qsa_layers[name] = dict(
                    token_topk=attention.indexer.token_topk,
                    compress_ratio=attention.indexer.compress_ratio,
                    indexer_dtype=str(attention.indexer.indexer_dtype),
                    head_dim=attention.indexer.index_head_dim,
                    score_scale=1.0,
                )
                self.handles.append(
                    attention.indexer.register_forward_pre_hook(
                        lambda _module, _args, name=name: setattr(self, "active_qsa", name)
                    )
                )
                self.handles.append(
                    attention.indexer.register_forward_hook(
                        lambda _module, _args, _output: setattr(self, "active_qsa", None)
                    )
                )
                self.handles.append(
                    attention.o_proj.register_forward_pre_hook(
                        lambda _module, args, name=name: self.capture(name + "/qsa/o_proj_input_tp0", args[0])
                    )
                )
            # vLLM PLE returns residual+increment, whereas Megatron returns the
            # increment. The matching post-PLE residual is attn_hc/input.
        mixers = [
            module for module in model.modules() if type(module).__name__ == "GatedResidual" and not module.use_combine
        ]
        if len(mixers) != 1:
            raise ValueError("Expected one final vLLM HC mixer")
        self._vllm_hc(mixers[0], "final_mixer/input", "final_mixer/output")
        if self.trace_qsa:
            from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

            original = qsa_indexer._topk

            def topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, workspace):
                result = original(logits, visible_blocks, token_topk, compress_ratio, block_indices, workspace)
                self._capture_qsa_scores(logits, visible_blocks)
                return result

            self.methods.append((qsa_indexer, "_topk", original))
            qsa_indexer._topk = topk

    def _capture_qsa_scores(self, logits, visible_blocks):
        if not self.enabled or self.current is None or self.active_qsa is None:
            return
        length = len(self.current["input_ids"])
        offset, count = self.vllm_window if self.vllm_window is not None else (0, length)
        if logits.shape[0] != count or visible_blocks.shape != (count,):
            raise ValueError("QSA score rows differ from absolute forward positions")
        start, end = max(self.token_start, offset), min(self.token_start + self.tokens, length, offset + count)
        if start >= end:
            return
        scores = logits[start - offset : end - offset].detach().clone()
        visible = visible_blocks[start - offset : end - offset].detach().clone()
        # Slots beyond visible_blocks are intentionally uninitialized in vLLM.
        # The valid lengths plus zero-filled snapshots preserve all real scores
        # without treating unused workspace bytes as numerical divergence.
        valid = torch.arange(scores.shape[1], device=scores.device)[None, :] < visible[:, None]
        scores.masked_fill_(~valid, 0)
        self._store_window(f"{self.active_qsa}/qsa/scores/tp-{self.tp_rank:02d}", scores, start)
        self._store_window(f"{self.active_qsa}/qsa/visible_blocks/tp-{self.tp_rank:02d}", visible[:, None], start)

    def _vllm_hc(self, module, input_name, output_name):
        def capture(output):
            # combine_and_mix materializes the delayed residual before reading
            # it, giving the same semantic boundary as the training backend.
            self.capture(input_name, output[0])
            self.capture(output_name, output[1])

        for name in ("mix", "combine_and_mix"):
            self.method_hook(module, name, capture)

    def finish(self):
        if self.current is None:
            raise RuntimeError("No backend trace is active")
        if (
            self.vllm_position_hooks
            and sum(chunk["tokens"] for chunk in self.forward_chunks) != self.current["trace_query_length"]
        ):
            raise ValueError("Incomplete vLLM prefill/decode token coverage")
        if self.enabled:
            metadata = dict(
                self.current,
                backend=self.backend,
                rank=self.rank,
                prefix_tokens=self.tokens,
                token_start=self.token_start,
            )
            if self.trace_qsa:
                metadata["qsa_layers"] = self.qsa_layers
                metadata["qsa_tp_size"] = self.tp_size
            if self.vllm_position_hooks:
                metadata["forward_chunks"] = self.forward_chunks
            filename = hashlib.sha256(metadata["id"].encode()).hexdigest()[:16] + ".pt"
            path = self.directory / filename
            if path.exists():
                raise FileExistsError(path)
            stages = {
                name: dict(offset=item["offset"], value=torch.cat([piece.cpu() for piece in item["pieces"]]))
                for name, item in self.values.items()
            }
            torch.save(dict(metadata=metadata, stages=stages), path)
            self.records.append(dict(metadata, file=filename, stages=list(stages)))
            (self.directory / "index.json").write_text(json.dumps(self.records, indent=2) + "\n")
        self.current = None
        self.values.clear()
        self.bytes = 0

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, method, original in reversed(self.methods):
            setattr(module, method, original)
        self.handles.clear()
        self.methods.clear()


class BackendTraceWorkerExtension:
    """Named RPC with JSON arguments; no pickle/callable serialization required."""

    def qwen38_trace(self, **kwargs):
        return vllm_trace_action(self.get_model(), **kwargs)

    def qwen38_production_trace(self, *, action, plan=None, directory=None):
        """Scope native batch/PLE hooks to one diagnostic generation phase."""
        import os

        from verl.models.mcore.qwen3_8_next.production_trace import (
            OUTPUT_ENV,
            PLAN_ENV,
            install_vllm_production_trace,
        )

        trace = getattr(self, "_qwen38_production_trace", None)
        if action == "install":
            if trace is not None or plan is None or directory is None:
                raise ValueError("A new production trace requires a plan, directory and no installed trace")
            keys = {PLAN_ENV: plan, OUTPUT_ENV: directory, "VERL_REPLICA_RANK": "0"}
            saved = {key: os.environ.get(key) for key in keys}
            try:
                os.environ.update(keys)
                install_vllm_production_trace(self)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        elif action == "close" and trace is not None:
            trace.flush()
            trace.close()
            del self._qwen38_production_trace
        else:
            raise ValueError(f"Invalid production trace action {action}")
        return dict(action=action, complete=True)

    def qwen38_ple_context_state(self, *, repair=False):
        """Audit the V2 PLE history offsets; repair only in an isolated A/B."""
        state = self.model_runner.model_state
        if type(state).__name__ != "Qwen4ExpModelState" or not state.uses_ngram_embedding:
            raise ValueError("PLE state audit requires native V2 Qwen4Exp")
        offsets = state.ngram_context_offsets
        before = offsets.detach().cpu().tolist()
        expected = list(range(-state.ngram_context_len, 0))
        if repair:
            offsets.copy_(torch.tensor(expected, dtype=offsets.dtype, device=offsets.device))
        return dict(
            before=before,
            expected=expected,
            after=offsets.detach().cpu().tolist(),
            repaired=repair,
            dtype=str(offsets.dtype),
            device=str(offsets.device),
        )

    def qwen38_sleep_weight_audit(self, *, stage, directory):
        """Check all named model tensors around this isolated level-2 sleep."""
        from verl.models.mcore.qwen3_8_next.runtime_audit import (
            compare_fingerprints,
            model_fingerprints,
            write_immutable,
        )

        if stage not in ("before", "after"):
            raise ValueError(stage)
        values = model_fingerprints(self.get_model())
        comparison = None
        if stage == "before":
            if hasattr(self, "_qwen38_sleep_fingerprints"):
                raise ValueError("An isolated sleep audit already exists")
            self._qwen38_sleep_fingerprints = values
        else:
            comparison = compare_fingerprints(self._qwen38_sleep_fingerprints, values)
        rank = torch.distributed.get_rank()
        result = dict(stage=stage, rank=rank, tensors=len(values), comparison=comparison)
        write_immutable(Path(directory) / f"rank-{rank:03d}" / f"{stage}.json", dict(result, fingerprints=values))
        return result

    def qwen38_qsa_order(self, *, policy):
        """Switch this dedicated diagnostic worker between native and ordered QSA."""
        if policy not in ("native", "block-id"):
            raise ValueError(policy)
        model = self.get_model()
        if hasattr(model, "_qwen38_prefix_trace"):
            raise ValueError("Close passive trace hooks before switching QSA order")
        layers = sum(type(module).__name__ == "QSAIndexer" for module in model.modules())
        if not layers:
            raise ValueError("Expected the Qwen4Exp QSA indexer")
        from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

        from .qsa_order import QsaBlockOrderProbe

        probe = getattr(self, "_qwen38_qsa_order_probe", None)
        if probe is None:
            probe = QsaBlockOrderProbe(qsa_indexer._topk)
            self._qwen38_qsa_order_probe = probe
        if qsa_indexer._topk is not probe and qsa_indexer._topk is not probe.native:
            raise ValueError("QSA selector was replaced by an unrelated probe")
        qsa_indexer._topk = probe if policy == "block-id" else probe.native
        self._qwen38_qsa_order_policy = policy
        return dict(policy=policy, qsa_layers=layers, canonical_calls=probe.calls, membership_changed=False)

    def qwen38_runtime_snapshot(self):
        import os

        import vllm
        from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual

        modes = [getattr(m, "use_fp32", False) for m in self.get_model().modules() if isinstance(m, GatedResidual)]
        gdn_dtypes = [
            str(m.conv1d.weight.dtype)
            for m in self.get_model().modules()
            if type(m).__name__ in ("QwenGatedDeltaNetAttention", "Qwen4ExpGatedDeltaNetAttention")
        ]
        gdn_backends = [
            m.gdn_prefill_backend for m in self.get_model().modules() if type(m).__name__ == "ChunkGatedDeltaRule"
        ]
        moe_backends = []
        for name, module in self.get_model().named_modules():
            if type(module).__name__ != "Qwen4ExpSparseMoeBlock":
                continue
            experts = module.experts
            routed = getattr(experts, "routed_experts", experts)
            method = routed.quant_method
            config = experts.moe_config
            parallel = config.moe_parallel_config
            backend = getattr(method, "unquantized_backend", None)
            kernel = getattr(method, "experts_cls", None)
            moe_backends.append(
                dict(
                    module=name,
                    experts_type=type(experts).__name__,
                    method_type=type(method).__name__,
                    backend=str(backend) if backend is not None else None,
                    kernel_type=kernel.__name__ if kernel is not None else None,
                    router_weight_dtype=str(module.gate.weight.dtype),
                    router_output_dtype=str(module.gate.out_dtype),
                    tensor_parallel_size=parallel.tp_size,
                    expert_parallel_size=parallel.ep_size,
                    expert_parallel_rank=parallel.ep_rank,
                    expert_parallel_enabled=parallel.use_ep,
                    all2all_backend=parallel.all2all_backend,
                    local_experts=config.num_local_experts,
                    intermediate_size=config.intermediate_size,
                    intermediate_size_per_partition=config.intermediate_size_per_partition,
                )
            )
        return dict(
            qsa_order_policy=getattr(self, "_qwen38_qsa_order_policy", "native"),
            hc_fp32_modes=modes,
            gdn_conv_weight_dtypes=gdn_dtypes,
            gdn_prefill_backends=gdn_backends,
            moe_backends=moe_backends,
            vllm_version=vllm.__version__,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
            tf32_environment={
                name: os.environ.get(name) for name in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
            },
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )


def vllm_trace_action(
    model,
    *,
    action,
    directory=None,
    tokens=32,
    token_start=0,
    trace_qsa=False,
    gdn_layers=(),
    prompt=None,
    config_sha256=None,
):
    """Worker-local action; no tensors are returned over RPC."""
    if action == "install":
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        rank = torch.distributed.get_rank()
        tp_rank = get_tensor_model_parallel_rank()
        trace = BackendPrefixTrace(
            directory,
            "vllm",
            rank,
            tp_rank,
            get_tensor_model_parallel_world_size(),
            enabled=tp_rank == 0 or trace_qsa or bool(gdn_layers),
            tokens=tokens,
            token_start=token_start,
            trace_qsa=trace_qsa,
        )
        trace.attach_vllm(model, gdn_layers=gdn_layers)
        model._qwen38_prefix_trace = trace
    else:
        trace = model._qwen38_prefix_trace
        if action == "start":
            trace.start(prompt, len(prompt["input_ids"]), config_sha256)
        elif action == "finish":
            trace.finish()
        elif action == "close":
            trace.close()
            del model._qwen38_prefix_trace
        else:
            raise ValueError(action)
    return action


def load_backend_trace(directory, prompt_id):
    pieces = defaultdict(list)
    metadata = None
    for index in sorted(Path(directory).glob("rank-*/index.json")):
        for record in json.loads(index.read_text()):
            if record["id"] != prompt_id:
                continue
            if metadata is not None:
                for field in ("input_ids", "config_sha256", "backend", "prefix_tokens"):
                    if metadata[field] != record[field]:
                        raise ValueError(f"Inconsistent shard metadata: {field}")
                if metadata.get("token_start", 0) != record.get("token_start", 0):
                    raise ValueError("Inconsistent shard metadata: token_start")
                if metadata.get("trace_query_length", len(metadata["input_ids"])) != record.get(
                    "trace_query_length", len(record["input_ids"])
                ):
                    raise ValueError("Inconsistent shard metadata: trace_query_length")
            metadata = record
            snapshot = torch.load(index.parent / record["file"], map_location="cpu", weights_only=True)
            for name, item in snapshot["stages"].items():
                pieces[name].append(item)
    if metadata is None:
        raise ValueError(f"Missing backend trace: {prompt_id}")
    start = metadata.get("token_start", 0)
    count = min(metadata["prefix_tokens"], len(metadata["input_ids"]) - start)
    values = {}
    for name, shards in pieces.items():
        joined, end = [], start
        for shard in sorted(shards, key=lambda item: item["offset"]):
            if shard["offset"] != end:
                raise ValueError(f"Missing or overlapping token coordinates at {name}, offset {end}")
            joined.append(shard["value"])
            end += shard["value"].shape[0]
        if end != start + count:
            raise ValueError(f"Incomplete prefix at {name}: {end - start} != {count}")
        values[name] = torch.cat(joined)
    return metadata, values


def compare_backend_traces(reference, candidate, prompt_id, *, max_tokens=None):
    from examples.grpo_trainer.qwen3_8_next.activation_trace import activation_differences

    left_meta, left = load_backend_trace(reference, prompt_id)
    right_meta, right = load_backend_trace(candidate, prompt_id)
    for field in ("input_ids", "config_sha256", "prefix_tokens"):
        if left_meta[field] != right_meta[field]:
            raise ValueError(f"Backend trace coordinates differ: {field}")
    if left_meta.get("token_start", 0) != right_meta.get("token_start", 0):
        raise ValueError("Backend trace coordinates differ: token_start")
    count = min(left_meta["prefix_tokens"], len(left_meta["input_ids"]) - left_meta.get("token_start", 0))
    if max_tokens is not None:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Compared token count must be a positive integer")
        count = min(count, max_tokens)
        left = {name: value[:count] for name, value in left.items()}
        right = {name: value[:count] for name, value in right.items()}
    common = set(left) & set(right)
    required = {name for name in left if name.endswith(("_hc/input", "_hc/mixed", "/attention/output", "/mlp/output"))}
    if not required or not required <= common:
        raise ValueError(f"Missing backend stages: {sorted(required - common)}")

    # Each rank file has execution order, but load_backend_trace reads whole
    # rank files in succession. Group TP feature shards at their shared stage
    # so rank-7 Q/K differences cannot appear after rank-0 attention output.
    def stage_name(name):
        base, separator, suffix = name.rpartition("/tp-")
        return base if separator and suffix.isdigit() else name

    stage_order = {}
    for name in left:
        stage_order.setdefault(stage_name(name), len(stage_order))
    # Decision buffers have backend-specific packing/padding and invalid score
    # workspace. Compare them through qsa_trace's causal-set analysis instead.
    decisions = ("/qsa/selection/", "/qsa/scores/", "/qsa/visible_blocks/")
    ordered = sorted(
        (name for name in left if name in common and not any(field in name for field in decisions)),
        key=lambda name: stage_order[stage_name(name)],
    )
    stages = activation_differences({name: left[name] for name in ordered}, {name: right[name] for name in ordered})
    for name, metrics in stages.items():
        metrics.update(
            shape=list(left[name].shape),
            reference_dtype=str(left[name].dtype),
            candidate_dtype=str(right[name].dtype),
        )
    return dict(
        acceptance_gate=False,
        prompt_id=prompt_id,
        token_start=left_meta.get("token_start", 0),
        compared_tokens=count,
        reference_backend=left_meta["backend"],
        candidate_backend=right_meta["backend"],
        first_nonzero_stage=next((name for name, values in stages.items() if values["max_abs"] > 0), None),
        reference_only_stages=sorted(set(left) - common),
        candidate_only_stages=sorted(set(right) - common),
        stages=stages,
    )
