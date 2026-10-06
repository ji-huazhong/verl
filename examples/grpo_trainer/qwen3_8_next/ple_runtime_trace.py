# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Trace PLE in native batched prefill/decode before and after sleep/wake."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.grpo_trainer.qwen3_8_next.real_prompt_trace import write_json


def probe_samples(samples):
    candidates = [s for s in samples if len(s["input_ids"]) <= 1024 and all(s["response_mask"])]
    if len(candidates) < 3:
        raise ValueError("Need at least three complete short production responses")
    candidates.sort(
        key=lambda s: float(np.abs(np.asarray(s["production_actor_logprobs"]) - s["generation_logprobs"]).max()),
        reverse=True,
    )
    selected = candidates[:7]
    if len(candidates) > 7:
        selected.append(candidates[-1])
    return [dict(s, response_length=min(s["response_length"], 128)) for s in selected]


def run(output, model):
    from vllm import LLM, SamplingParams

    samples = probe_samples(json.loads((output / "samples.json").read_text()))
    # Four copies occupy different live request/cache slots, like the four
    # responses per prompt in GRPO. Tokens are forced, with native model scores.
    batch = [dict(s, repeat=repeat) for repeat in range(4) for s in samples]
    prompts = list({tuple(s["prompt_ids"]): s["prompt_ids"] for s in samples}.values())
    plan = output / "ple-plan.json"
    write_json(
        plan,
        dict(
            prompts=prompts,
            response_queries=128,
            layer_limit=2,
            ple_substages=True,
            prompt_tail=32,
            ple_cache_queries=16,
        ),
    )
    write_json(output / "ple-samples.json", samples)
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
        enable_sleep_mode=True,
        engram_config={"cpu_offload": True, "dp_shared_memory": False},
        gpu_memory_utilization=0.55,
        logprobs_mode="processed_logprobs",
        enable_trace_replay=True,
        worker_extension_cls="examples.grpo_trainer.qwen3_8_next.backend_trace.BackendTraceWorkerExtension",
    )
    results = {}
    try:
        policies = engine.collective_rpc("qwen38_qsa_order", kwargs={"policy": "block-id"})
        assert len(policies) == 8 and all(p["policy"] == "block-id" for p in policies)
        write_json(
            output / "runtime.json",
            dict(
                config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
                workers=engine.collective_rpc("qwen38_runtime_snapshot"),
            ),
        )
        for phase in ("decode_control", "decode_trace", "prefill_trace", "wake_trace"):
            if phase == "wake_trace":
                engine.sleep(level=1)
                engine.wake_up()
            assert engine.reset_prefix_cache(reset_connector=True)
            traced = phase != "decode_control"
            prefill = phase == "prefill_trace"
            if traced:
                engine.collective_rpc(
                    "qwen38_production_trace",
                    kwargs=dict(
                        action="install",
                        plan=str(plan),
                        directory=str(output / phase),
                    ),
                )
            try:
                params, inputs = [], []
                for sample in batch:
                    p, n = sample["prompt_length"], sample["response_length"]
                    tokens = sample["input_ids"][p : p + n]
                    params.append(
                        SamplingParams(
                            temperature=1.0,
                            top_p=1.0,
                            top_k=-1,
                            repetition_penalty=1.0,
                            max_tokens=1 if prefill else n,
                            ignore_eos=True,
                            logprobs=0,
                            prompt_logprobs=1 if prefill else None,
                            detokenize=False,
                            seed=20261004,
                            trace_decode_token_ids=None if prefill else tokens,
                        )
                    )
                    inputs.append(
                        {"prompt_token_ids": sample["input_ids"][: p + n] if prefill else sample["prompt_ids"]}
                    )
                generated = engine.generate(inputs, params, use_tqdm=False)
                rows = []
                for sample, result in zip(batch, generated, strict=True):
                    p, n = sample["prompt_length"], sample["response_length"]
                    tokens = sample["input_ids"][p : p + n]
                    if prefill:
                        scores = [result.prompt_logprobs[p + i][t].logprob for i, t in enumerate(tokens)]
                    else:
                        answer = result.outputs[0]
                        assert list(answer.token_ids) == tokens
                        scores = [row[t].logprob for t, row in zip(tokens, answer.logprobs, strict=True)]
                    assert len(scores) == n and np.isfinite(scores).all()
                    rows.append(
                        dict(
                            id=sample["id"],
                            repeat=sample["repeat"],
                            request_id=result.request_id,
                            input_ids=sample["input_ids"][: p + n],
                            prompt_ids=sample["prompt_ids"],
                            logprobs=scores,
                            cached_tokens=result.num_cached_tokens,
                        )
                    )
                results[phase] = rows
                write_json(output / (phase + ".json"), rows)
            finally:
                if traced:
                    engine.collective_rpc("qwen38_production_trace", kwargs={"action": "close"})
            print("QWEN38_REAL_PROMPT_PLE_PHASE", phase, flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    differences = {}
    for phase in ("decode_trace", "prefill_trace", "wake_trace"):
        deltas = [
            np.abs(np.asarray(a["logprobs"]) - b["logprobs"])
            for a, b in zip(results[phase], results["decode_control"], strict=True)
        ]
        differences[phase] = dict(
            token_mean_abs=float(np.concatenate(deltas).mean()), max_abs=float(np.concatenate(deltas).max())
        )
    write_json(
        output / "report.json",
        dict(
            complete=True,
            full_precision_accepted=False,
            production_changed=False,
            samples=len(samples),
            batch_requests=len(batch),
            queries_per_response=128,
            differences_vs_decode_control=differences,
            trace_control_passed=differences["decode_trace"]["max_abs"] == 0,
            scope="Native V2 TP8 batched forced tokens, first two layers plus PLE substages. "
            "HF-loaded weights and level-1 sleep/wake; no production refit and no actor update.",
        ),
    )
    print("QWEN38_REAL_PROMPT_PLE_COMPLETE", json.dumps(differences), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.model)
