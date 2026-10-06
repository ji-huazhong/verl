# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Opt-in snapshots of actual packed actor and batched rollout forwards.

Coordinates are request/document-local causal query positions. These hooks
observe native calls; they do not replay tokens or change routes or kernels.
"""

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from types import MethodType

import torch

PLAN_ENV = "VERL_QWEN38_LAYER_TRACE_PLAN"
OUTPUT_ENV = "VERL_QWEN38_LAYER_TRACE_DIR"


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


class ProductionTrace:
    def __init__(self, plan, directory, backend, tp_rank, tp_size):
        self.prompts = plan["prompts"]
        self.limit = plan["response_queries"]
        self.ple_substages = plan.get("ple_substages", False)
        self.layer_limit = plan.get("layer_limit", 48)
        self.prompt_tail = plan.get("prompt_tail", 1)
        self.ple_cache_queries = plan.get("ple_cache_queries", 0)
        if not 1 <= self.prompt_tail <= 128 or not 0 <= self.ple_cache_queries <= 32:
            raise ValueError("Invalid bounded PLE history capture")
        if not 1 <= self.layer_limit <= 48:
            raise ValueError("Invalid production layer limit")
        if not 1 <= self.limit <= 2048 or not self.prompts or any(not ids for ids in self.prompts):
            raise ValueError("Invalid production layer trace plan")
        self.directory = Path(directory)
        self.backend, self.tp_rank, self.tp_size = backend, tp_rank, tp_size
        self.rows, self.records, self.handles, self.methods = [], {}, [], []
        self.total_rows = 0
        self.bytes = 0

    def match_prompt(self, ids):
        matches = [prompt for prompt in self.prompts if ids[: len(prompt)] == prompt]
        if len(matches) > 1:
            raise ValueError("Trace prompt prefixes must be unambiguous")
        return matches[0] if matches else None

    def register(self, identity, prompt, input_ids=None):
        if identity in self.records:
            raise ValueError("A traced request was registered twice")
        self.records[identity] = dict(prompt_ids=list(prompt), input_ids=input_ids, stages=defaultdict(list))

    def select_rows(self, identity, start, positions, ids):
        """Select scored causal queries, including the last prompt query."""
        if len(positions) != len(ids):
            raise ValueError("Query positions and token IDs differ")
        prompt = self.records[identity]["prompt_ids"]
        for index, (position, token) in enumerate(zip(positions, ids, strict=True)):
            if position < len(prompt) and token != prompt[position]:
                raise ValueError("Traced input differs from the selected prompt")
            if max(0, len(prompt) - self.prompt_tail) <= position < len(prompt) - 1 + self.limit:
                self.rows.append((start + index, identity, position, token))

    def capture(self, stage, value, *, feature_shard=False):
        if not self.rows:
            return
        if isinstance(value, tuple):
            value = value[0]
        if value.ndim == 3 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 2:
            raise ValueError(f"Unsupported production activation shape at {stage}: {tuple(value.shape)}")
        offset = 0
        if value.shape[0] == self.total_rows:
            if self.tp_rank != 0 and not feature_shard:
                return
        elif self.backend == "megatron" and value.shape[0] * self.tp_size == self.total_rows:
            offset = self.tp_rank * value.shape[0]
        else:
            raise ValueError(f"Unmapped production activation rows at {stage}")
        selected = [row for row in self.rows if offset <= row[0] < offset + value.shape[0]]
        if not selected:
            return
        indices = torch.tensor([row[0] - offset for row in selected], device=value.device)
        saved = value.detach().index_select(0, indices).to("cpu", copy=True)
        if feature_shard:
            stage += f"/tp-{self.tp_rank:02d}"
        self.store_rows(stage, saved, selected)

    def store_rows(self, stage, saved, selected):
        """Store already selected CPU rows, including request-indexed state."""
        self.bytes += saved.numel() * saved.element_size()
        if self.bytes > 16 * 1024**3:
            raise RuntimeError("Production trace exceeded 16 GiB of host snapshots on this rank")
        by_request = defaultdict(list)
        for i, row in enumerate(selected):
            by_request[row[1]].append((i, row[2], row[3]))
        for identity, rows in by_request.items():
            # Clone so a completed request releases its share of the batch.
            self.records[identity]["stages"][stage].append(
                dict(
                    positions=[row[1] for row in rows],
                    token_ids=[row[2] for row in rows],
                    value=saved[[row[0] for row in rows]].clone(),
                )
            )

    def capture_ple_cache(self, prefix, ple):
        if self.tp_rank != 0 or not self.ple_cache_queries or not self.rows:
            return
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        metadata = get_forward_context().attn_metadata[ple.prefix]
        if metadata.num_spec_decodes:
            raise ValueError("PLE cache tracing currently requires non-speculative decode")
        selected = [
            row
            for row in self.rows
            if row[0] < metadata.num_decode_tokens
            and len(self.records[row[1]]["prompt_ids"])
            <= row[2]
            < len(self.records[row[1]]["prompt_ids"]) + self.ple_cache_queries
        ]
        if not selected:
            return
        # Non-speculative decode rows are first, with exactly one token per
        # request. Select native slots before the in-place state update.
        indices = torch.tensor([row[0] for row in selected], device=metadata.state_indices_tensor.device)
        slots = metadata.state_indices_tensor.index_select(0, indices).long()
        if (slots < 0).any():
            raise ValueError("Selected real PLE decode has no state slot")
        state = ple.kv_cache[0].index_select(0, slots)
        if not is_conv_state_dim_first():
            state = state.transpose(-1, -2)
        state = state[..., -ple.conv_state_len :]
        self.store_rows(prefix + "cache_before", state.flatten(1).detach().to("cpu", copy=True), selected)
        flags = metadata.has_initial_states_d.index_select(0, indices).long()
        self.store_rows(prefix + "cache_slot_and_valid", torch.stack([slots, flags], dim=1).cpu(), selected)

    def output_hook(self, stage, module, *, feature_shard=False):
        self.handles.append(
            module.register_forward_hook(
                lambda _m, _a, output: self.capture(stage, output, feature_shard=feature_shard)
            )
        )

    def method_hook(self, module, name, callback, *, before=None):
        original = getattr(module, name)

        def call(_module, *args, **kwargs):
            if before is not None:
                before(args, kwargs)
            output = original(*args, **kwargs)
            callback(output)
            return output

        self.methods.append((module, name, original))
        setattr(module, name, MethodType(call, module))

    def attach_ple(self, name, ple, hc=None):
        """Bracket the PLE increment separately from the following HC mix.

        Hash IDs and embedding rows distinguish history/lookup errors from
        propagation through the gate. No additional model forward is run.
        """
        prefix = name + "/ple/"
        self.handles.append(ple.register_forward_pre_hook(lambda _m, args: self.capture(prefix + "input", args[0])))
        self.output_hook(prefix + "embedding", ple.ple_embedding)
        if self.backend == "megatron":
            self.handles.append(
                ple.register_forward_pre_hook(lambda _m, args: self.capture(prefix + "ngram_ids", args[1]))
            )
            self.output_hook(prefix + "key", ple.key_proj)
            self.output_hook(prefix + "value", ple.value_proj)
            # Megatron PLE returns only the increment; _apply_ple adds it to
            # the original residual, matching the native vLLM PLE output.
            self.output_hook(prefix + "increment", ple)
            self.method_hook(hc, "_apply_ple", lambda output: self.capture(prefix + "output", output))
        else:
            self.method_hook(
                ple.ple_embedding,
                "compute_ngram_ids",
                lambda output: self.capture(prefix + "ngram_ids", output),
            )

            def projection(_module, _args, output):
                key, value = output[0].split(ple.kv_proj.output_sizes, dim=-1)
                self.capture(prefix + "key", key)
                self.capture(prefix + "value", value)

            self.handles.append(ple.kv_proj.register_forward_hook(projection))

            def before_conv(args, kwargs):
                self.capture(prefix + "conv_input", args[0])
                # _short_conv updates this buffer in place: copy the gate
                # output before the native call, not in its post-hook.
                self.capture(prefix + "gated", args[1])
                self.capture_ple_cache(prefix, ple)

            self.method_hook(ple, "_short_conv", lambda output: None, before=before_conv)
            self.output_hook(prefix + "output", ple)

    def attach_vllm(self, model):
        layers = [m for m in model.modules() if type(m).__name__ == "Qwen4ExpDecoderLayer"]
        if len(layers) != 48:
            raise ValueError("Expected all 48 rollout layers")
        for layer in layers:
            if layer.layer_idx >= self.layer_limit:
                continue
            name = f"layers/{layer.layer_idx:02d}"
            if self.ple_substages and layer.ple is not None:
                self.attach_ple(name, layer.ple)
            for site in ("attn", "mlp"):
                hc = getattr(layer, site + "_hyper_connection")
                for method in ("mix", "combine_and_mix"):
                    self.method_hook(
                        hc,
                        method,
                        lambda output, name=name, site=site: self.capture(f"{name}/{site}_hc/mixed", output[1]),
                    )
            attention = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
            self.output_hook(name + "/attention/output", attention)
            self.output_hook(name + "/mlp/output", layer.mlp)
            self.output_hook(name + "/router/logits", layer.mlp.gate)
            if layer.layer_idx == 0:
                self.output_hook(name + "/gdn/projection/qkvz", attention.in_proj_qkvz, feature_shard=True)
                self.output_hook(name + "/gdn/projection/ba", attention.in_proj_ba, feature_shard=True)
                self.handles.append(
                    attention.out_proj.register_forward_pre_hook(
                        lambda _m, args: self.capture("layers/00/gdn/out_proj_input", args[0], feature_shard=True)
                    )
                )
        mixers = [m for m in model.modules() if type(m).__name__ == "GatedResidual" and not m.use_combine]
        if len(mixers) != 1:
            raise ValueError("Expected one final HC mixer")
        for method in ("mix", "combine_and_mix"):
            self.method_hook(mixers[0], method, lambda output: self.capture("final_mixer/output", output[1]))

    def attach_megatron(self, model):
        decoder = model.language_model.decoder
        for layer in decoder.layers:
            if layer.layer_number > self.layer_limit:
                continue
            name = f"layers/{layer.layer_number - 1:02d}"
            hc = layer.self_attention_hyper_connection
            if self.ple_substages and hasattr(hc, "ple"):
                self.attach_ple(name, hc.ple, hc)
            for site, attr in (("attn", "self_attention_hyper_connection"), ("mlp", "mlp_hyper_connection")):
                self.handles.append(
                    getattr(layer, attr).register_forward_hook(
                        lambda _m, _args, output, name=name, site=site: self.capture(
                            f"{name}/{site}_hc/mixed", output[0]
                        )
                    )
                )
            self.output_hook(name + "/attention/output", layer.self_attention)
            self.output_hook(name + "/mlp/output", layer.mlp)
            self.method_hook(
                layer.mlp.router, "gating", lambda output, name=name: self.capture(name + "/router/logits", output)
            )
            self.handles.append(
                layer.mlp.router.register_forward_hook(
                    lambda _m, _args, output, name=name: self.capture(name + "/router/selected", output[1])
                )
            )
            if layer.layer_number == 1:
                for part in ("qkvz", "ba"):
                    self.output_hook(
                        name + "/gdn/projection/" + part,
                        getattr(layer.self_attention.in_proj, part),
                        feature_shard=True,
                    )
                self.handles.append(
                    layer.self_attention.out_proj.register_forward_pre_hook(
                        lambda _m, args: self.capture("layers/00/gdn/out_proj_input", args[0], feature_shard=True)
                    )
                )
        if decoder.final_layernorm is not None:
            self.output_hook("final_mixer/output", decoder.final_layernorm)

    def flush(self, identities=None):
        for identity in list(self.records) if identities is None else identities:
            record = self.records.pop(identity, None)
            if record is None or not record["stages"]:
                continue
            for stage, pieces in record["stages"].items():
                positions = [position for piece in pieces for position in piece["positions"]]
                if positions != sorted(set(positions)):
                    raise ValueError(f"Duplicate or reordered activation coordinates at {stage}")
            name = token_digest([identity])
            folder = self.directory / name
            folder.mkdir(parents=True, exist_ok=False)
            metadata = dict(
                complete=True,
                backend=self.backend,
                identity=identity,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prompt_ids=record["prompt_ids"],
                input_ids=record["input_ids"],
                stages=list(record["stages"]),
            )
            with tempfile.TemporaryDirectory(prefix="qwen38-layer-trace-") as local:
                path = Path(local) / "activations.pt"
                torch.save(dict(metadata=metadata, stages=dict(record["stages"])), path)
                with path.open("rb") as handle:
                    metadata["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
                shutil.copyfile(path, folder / path.name)
            (folder / "complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
            self.bytes -= sum(
                piece["value"].numel() * piece["value"].element_size()
                for pieces in record["stages"].values()
                for piece in pieces
            )

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, name, original in reversed(self.methods):
            setattr(module, name, original)


def install_vllm_production_trace(worker):
    if not os.environ.get(PLAN_ENV):
        return
    if not worker.use_v2_model_runner or not worker.model_config.enforce_eager:
        raise ValueError("Production trace currently supports the native V2 eager runner only")
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    rank = torch.distributed.get_rank()
    directory = (
        Path(os.environ[OUTPUT_ENV])
        / "vllm"
        / f"replica-{int(os.environ['VERL_REPLICA_RANK']):03d}"
        / f"rank-{rank:03d}"
    )
    trace = ProductionTrace(
        json.loads(Path(os.environ[PLAN_ENV]).read_text()),
        directory,
        "vllm",
        get_tensor_model_parallel_rank(),
        get_tensor_model_parallel_world_size(),
    )
    runner = worker.model_runner
    trace.attach_vllm(worker.get_model())
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "installed.json").open("x") as handle:
        json.dump(
            dict(
                complete=True,
                runner_type=type(runner).__module__ + "." + type(runner).__name__,
                model_type=type(worker.get_model()).__name__,
                tp_rank=trace.tp_rank,
                tp_size=trace.tp_size,
                response_queries=trace.limit,
                prompt_sha256=[token_digest(prompt) for prompt in trace.prompts],
                eager=worker.model_config.enforce_eager,
                native_v2=worker.use_v2_model_runner,
            ),
            handle,
            indent=2,
        )
    original_execute, original_prepare, original_sleep = runner.execute_model, runner.prepare_inputs, worker.sleep

    def prepare(_runner, *args, **kwargs):
        batch = original_prepare(*args, **kwargs)
        trace.rows = []
        trace.total_rows = batch.num_tokens_after_padding
        if any(identity in trace.records for identity in batch.req_ids):
            positions = batch.positions.detach().cpu().tolist()
            ids = batch.input_ids.detach().cpu().tolist()
            for i, identity in enumerate(batch.req_ids):
                if identity in trace.records:
                    begin, end = map(int, batch.query_start_loc_np[i : i + 2])
                    trace.select_rows(identity, begin, positions[begin:end], ids[begin:end])
        return batch

    def execute(_runner, scheduler_output, *args, **kwargs):
        trace.rows = []
        if scheduler_output is not None:
            trace.flush(scheduler_output.finished_req_ids)
            for request in scheduler_output.scheduled_new_reqs:
                prompt = None if request.prompt_token_ids is None else trace.match_prompt(request.prompt_token_ids)
                if prompt:
                    trace.register(request.req_id, prompt)
        try:
            return original_execute(scheduler_output, *args, **kwargs)
        finally:
            trace.rows = []

    def sleep(_worker, *args, **kwargs):
        trace.flush()
        return original_sleep(*args, **kwargs)

    runner.prepare_inputs = MethodType(prepare, runner)
    runner.execute_model = MethodType(execute, runner)
    worker.sleep = MethodType(sleep, worker)
    trace.methods.extend(
        [
            (runner, "prepare_inputs", original_prepare),
            (runner, "execute_model", original_execute),
            (worker, "sleep", original_sleep),
        ]
    )
    worker._qwen38_production_trace = trace


def wrap_megatron_production_forward(forward, model, input_ids, *, enabled):
    if not enabled or model.training or not os.environ.get(PLAN_ENV):
        return forward
    from megatron.core import parallel_state as ps

    if ps.get_context_parallel_world_size() != 1:
        raise ValueError("Production layer tracing currently requires CP1")
    counter = getattr(model, "_qwen38_trace_forward_count", 0)
    model._qwen38_trace_forward_count = counter + 1
    directory = (
        Path(os.environ[OUTPUT_ENV])
        / "megatron"
        / f"rank-{torch.distributed.get_rank():03d}"
        / f"forward-{counter:04d}"
    )
    trace = ProductionTrace(
        json.loads(Path(os.environ[PLAN_ENV]).read_text()),
        directory,
        "megatron",
        ps.get_tensor_model_parallel_rank(),
        ps.get_tensor_model_parallel_world_size(),
    )
    documents = [row.detach().cpu().tolist() for row in input_ids.unbind()]

    def start(_model, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        packed = kwargs.get("packed_seq_params")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or packed is None:
            raise ValueError("Expected the production THD language-model input")
        tokens = ids[0].detach().cpu().tolist()
        cu = packed.cu_seqlens_q.detach().cpu().tolist()
        if len(cu) != len(documents) + 1 or cu[0] != 0 or cu[-1] != len(tokens):
            raise ValueError("Packed document boundaries differ from the original microbatch")
        trace.total_rows = len(tokens)
        for i, (begin, end) in enumerate(zip(cu[:-1], cu[1:], strict=True)):
            document = documents[i]
            if len(document) > end - begin or tokens[begin : begin + len(document)] != document:
                raise ValueError("Packed physical tokens differ from the original document")
            prompt = trace.match_prompt(document)
            if prompt:
                identity = f"document-{i}-{token_digest(document)}"
                trace.register(identity, prompt, document)
                # The final token has no sampled successor.
                trace.select_rows(identity, begin, list(range(len(document) - 1)), document[:-1])

    def run(*args, **kwargs):
        trace.attach_megatron(model)
        trace.handles.append(model.language_model.register_forward_pre_hook(start, with_kwargs=True))
        processor = kwargs.get("logits_processor")
        if processor is not None:

            def capture_scores(*processor_args, **processor_kwargs):
                result = processor(*processor_args, **processor_kwargs)
                if "log_probs" in result:
                    scores = result["log_probs"]
                    if scores.ndim != 2 or scores.shape[0] != 1:
                        raise ValueError("Expected original packed actor scores")
                    trace.capture("log_probs", scores.transpose(0, 1))
                return result

            kwargs = dict(kwargs, logits_processor=capture_scores)
        try:
            result = forward(*args, **kwargs)
            trace.flush()
            return result
        finally:
            trace.close()

    return run


def audit_selected_response(*, replica, request_id, prompt_ids, final_res):
    """Keep exact generated tokens, scores and actual R3 routes for traced prompts."""
    if not os.environ.get(PLAN_ENV):
        return
    plan = json.loads(Path(os.environ[PLAN_ENV]).read_text())
    if list(prompt_ids) not in plan["prompts"]:
        return
    answer = final_res.outputs[0]
    tokens = list(answer.token_ids)
    if answer.routed_experts is None or answer.logprobs is None:
        raise ValueError("Production layer trace requires actual rollout routes and scores")
    routes = torch.as_tensor(answer.routed_experts).to("cpu", copy=True)
    if routes.ndim != 3 or routes.shape[0] not in (len(prompt_ids) + len(tokens) - 1, len(prompt_ids) + len(tokens)):
        raise ValueError("Rollout routes do not match the traced causal trajectory")
    payload = dict(
        complete=True,
        request_id=request_id,
        replica=replica,
        prompt_ids=list(prompt_ids),
        response_ids=tokens,
        input_ids=list(prompt_ids) + tokens,
        logprobs=[entry[token].logprob for token, entry in zip(tokens, answer.logprobs, strict=True)],
        cached_tokens=getattr(final_res, "num_cached_tokens", None),
    )
    folder = Path(os.environ[OUTPUT_ENV]) / "responses" / f"replica-{replica:03d}" / token_digest([request_id])
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(routes, folder / "routes.pt")
    with (folder / "routes.pt").open("rb") as handle:
        payload["routes_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    (folder / "complete.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
