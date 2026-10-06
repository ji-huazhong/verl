# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Isolate the production vocabulary mask on fixed real response tokens."""

import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from examples.grpo_trainer.qwen3_8_next.backend_trace import BackendTraceWorkerExtension
from examples.grpo_trainer.qwen3_8_next.production_decode_replay import select_probe_samples
from examples.grpo_trainer.qwen3_8_next.real_prompt_trace import (
    response_logprobs,
    reuse_production_capture,
    write_json,
)

PHASES = ("native0", "tail0", "production0", "production1", "native1")


def mask_statistics(logits, vocab_size, banned_token_ids):
    """Measure normalization changes from one unchanged logits tensor."""
    values = logits.float()
    native = values.logsumexp(-1)
    allowed = values[..., :vocab_size].clone()
    tail = allowed.logsumexp(-1)
    ids = [i for i in banned_token_ids if i < allowed.shape[-1]]
    if ids:
        allowed[..., ids] = -torch.inf
    production = allowed.logsumexp(-1)
    return {
        "native_logsumexp": native.tolist(),
        "tail_logprob_shift": (native - tail).tolist(),
        "production_logprob_shift": (native - production).tolist(),
        "native_argmax": values.argmax(-1).tolist(),
    }


class VocabularyMaskProbe:
    def __init__(self, model, vocab_size, banned_token_ids, patch):
        self.model, self.original = model, model.compute_logits
        self.vocab_size, self.banned_token_ids = vocab_size, banned_token_ids
        self.patch, self.records = patch, []
        self.versions = {name: value._version for name, value in model.named_parameters()}

    def select(self, policy):
        if policy not in ("native", "tail", "production"):
            raise ValueError(f"Unknown vocabulary mask policy: {policy}")
        if self.records:
            raise ValueError("Drain the previous probe before changing policy")

        def compute_logits(model, *args, **kwargs):
            logits = self.original(*args, **kwargs)
            # Non-output TP ranks may have no gathered logits.
            if logits is not None:
                hidden = args[0] if args else kwargs["hidden_states"]
                self.records.append(
                    dict(
                        shape=list(logits.shape),
                        hidden_sha256=hashlib.sha256(
                            hidden.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                        ).hexdigest(),
                        **mask_statistics(logits, self.vocab_size, self.banned_token_ids),
                    )
                )
            return logits

        self.model.compute_logits = MethodType(compute_logits, self.model)
        if policy != "native":
            # Use the exact production function, not a duplicate sampler mask.
            self.patch(self.model, self.vocab_size, self.banned_token_ids if policy == "production" else [])

    def drain(self):
        assert self.versions == {name: value._version for name, value in self.model.named_parameters()}
        records, self.records = self.records, []
        return records

    def close(self):
        self.model.compute_logits = self.original


class VocabularyMaskWorkerExtension(BackendTraceWorkerExtension):
    def qwen38_vocab_mask(self, *, action, vocab_size=None, banned_token_ids=None, policy=None):
        if action == "install":
            from verl.workers.rollout.vllm_rollout.utils import monkey_patch_compute_logits

            assert not hasattr(self, "_vocab_mask_probe")
            self._vocab_mask_probe = VocabularyMaskProbe(
                self.get_model(), vocab_size, banned_token_ids, monkey_patch_compute_logits
            )
            return {"installed": True}
        probe = self._vocab_mask_probe
        if action == "select":
            probe.select(policy)
            return {"policy": policy}
        if action == "drain":
            return probe.drain()
        if action == "close":
            probe.close()
            del self._vocab_mask_probe
            return {"restored": True}
        raise ValueError(action)


def assess(output, samples, metadata):
    phases = {phase: json.loads((output / f"{phase}.json").read_text()) for phase in PHASES}
    for left, right in (("native0", "native1"), ("production0", "production1")):
        assert phases[left] == phases[right], f"Non-repeatable mask control: {left} vs {right}"
    records, gaps = [], {}
    for sample in samples:
        name = sample["id"]
        base = phases["native0"][name]
        for phase in PHASES:
            row = phases[phase][name]
            assert all(row[key] == base[key] for key in ("routes_sha256", "cached_tokens", "workers"))
        native = np.asarray(base["logprobs"])
        actor = np.asarray(sample["production_actor_logprobs"])
        rollout = np.asarray(sample["generation_logprobs"])
        worst = int(np.abs(actor - rollout).argmax())
        record = dict(id=name, worst_response_index=worst, actor=float(actor[worst]), rollout=float(rollout[worst]))
        for phase in ("native0", "tail0", "production0"):
            scores = np.asarray(phases[phase][name]["logprobs"])
            assert scores.shape == rollout.shape and np.isfinite(scores).all()
            assert np.all(scores >= native - 1e-5), "Masking allowed tokens cannot lower their probabilities"
            delta = np.abs(scores - rollout)
            gaps.setdefault(phase, []).append(delta)
            record[phase] = dict(
                mean_abs=float(delta.mean()),
                max_abs=float(delta.max()),
                worst_token_logprob=float(scores[worst]),
                worst_token_shift=float(scores[worst] - native[worst]),
            )
        records.append(record)
    report = dict(
        complete=True,
        controls_valid=True,
        full_model_acceptance=False,
        production_changed=False,
        samples=len(samples),
        metadata=metadata,
        records=records,
        summaries={
            phase: dict(
                response_mean_abs=float(np.mean([x.mean() for x in rows])),
                token_mean_abs=float(np.concatenate(rows).mean()),
                max_abs=float(np.concatenate(rows).max()),
            )
            for phase, rows in gaps.items()
        },
        scope="Same loaded HF vLLM, fixed real tokens, native expert routes. Exact production compute_logits mask. "
        "Hidden states, normalization statistics and route hashes must stay identical across all policies. "
        "No production refit or full-batch acceptance.",
    )
    write_json(output / "report.json", report)
    print("QWEN38_REAL_PROMPT_MASK_COMPLETE", json.dumps(report["summaries"]), flush=True)


def run(output, model, source, reference):
    from vllm import LLM, SamplingParams

    from verl.utils import hf_processor, hf_tokenizer
    from verl.workers.rollout.utils import get_vision_placeholder_token_ids

    _, captured = reuse_production_capture(source, output, 16)
    samples = select_probe_samples(captured)
    reference_report = json.loads((reference / "report.json").read_text())
    assert reference_report["complete"] and reference_report["controls_valid"]
    assert (reference / "samples.json").read_bytes() == (output / "samples.json").read_bytes()
    tokenizer = hf_tokenizer(str(model), trust_remote_code=True)
    processor = hf_processor(str(model), trust_remote_code=True)
    vocab_size, banned = len(tokenizer), get_vision_placeholder_token_ids(processor)
    for sample in samples:
        assert all(
            0 <= token < vocab_size and token not in banned for token in sample["input_ids"][sample["prompt_length"] :]
        )
    metadata = dict(
        tokenizer_type=type(tokenizer).__name__,
        tokenizer_length=vocab_size,
        processor_type=type(processor).__name__,
        banned_token_ids=banned,
        config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
    )
    write_json(output / "mask-metadata.json", metadata)
    write_json(output / "decode-samples.json", samples)
    engine = LLM(
        model=str(model),
        skip_tokenizer_init=True,
        language_model_only=True,
        dtype="bfloat16",
        tensor_parallel_size=8,
        load_format="safetensors",
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enable_return_routed_experts=True,
        engram_config={"cpu_offload": True, "dp_shared_memory": False},
        gpu_memory_utilization=0.55,
        logprobs_mode="processed_logprobs",
        worker_extension_cls="examples.grpo_trainer.qwen3_8_next.vocab_mask_ab.VocabularyMaskWorkerExtension",
    )
    try:
        policies = engine.collective_rpc("qwen38_qsa_order", kwargs={"policy": "block-id"})
        assert len(policies) == 8 and all(x["policy"] == "block-id" for x in policies)
        write_json(
            output / "runtime.json", dict(metadata=metadata, workers=engine.collective_rpc("qwen38_runtime_snapshot"))
        )
        engine.collective_rpc(
            "qwen38_vocab_mask", kwargs=dict(action="install", vocab_size=vocab_size, banned_token_ids=banned)
        )
        for phase in PHASES:
            engine.collective_rpc("qwen38_vocab_mask", kwargs=dict(action="select", policy=phase.rstrip("01")))
            rows = {}
            for sample in samples:
                assert engine.reset_prefix_cache(reset_connector=True)
                result = engine.generate(
                    [{"prompt_token_ids": sample["input_ids"]}],
                    SamplingParams(
                        temperature=1,
                        top_p=1,
                        top_k=-1,
                        repetition_penalty=1,
                        max_tokens=1,
                        ignore_eos=True,
                        logprobs=0,
                        prompt_logprobs=1,
                        detokenize=False,
                        seed=20261004,
                    ),
                    use_tqdm=False,
                )[0]
                logs = [result.prompt_logprobs[i][token].logprob for i, token in enumerate(sample["input_ids"]) if i]
                routes = result.outputs[0].routed_experts
                assert routes.shape == (len(sample["input_ids"]), 48, 10)
                rows[sample["id"]] = dict(
                    logprobs=response_logprobs(sample, logs),
                    routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
                    cached_tokens=result.num_cached_tokens,
                    workers=engine.collective_rpc("qwen38_vocab_mask", kwargs={"action": "drain"}),
                )
                assert len(rows[sample["id"]]["workers"]) == 8 and any(rows[sample["id"]]["workers"])
            write_json(output / f"{phase}.json", rows)
            print("QWEN38_REAL_PROMPT_MASK_PHASE", phase, flush=True)
        engine.collective_rpc("qwen38_vocab_mask", kwargs={"action": "close"})
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    assess(output, samples, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "model", "source", "reference"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.model, args.source, args.reference)
