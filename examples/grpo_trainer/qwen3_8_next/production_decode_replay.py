# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Same-weight vLLM prefill/decode/batching comparison on captured production tokens."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.grpo_trainer.qwen3_8_next.real_prompt_trace import (
    response_logprobs,
    reuse_production_capture,
    write_json,
)

PHASES = (
    "prefill0",
    "decode0",
    "batch0",
    "batch1",
    "prefill_trace0",
    "prefill_trace1",
    "decode_trace0",
    "decode_trace1",
    "decode1",
    "prefill1",
)


def select_probe_samples(samples):
    """Two largest short-context outliers and the smallest-error short control."""
    candidates = [s for s in samples if len(s["input_ids"]) <= 1024 and all(s["response_mask"])]
    if len(candidates) < 3:
        raise ValueError("Need three distinct, fully scored real short responses")

    def delta(sample):
        return np.abs(np.asarray(sample["production_actor_logprobs"]) - sample["generation_logprobs"])

    ordered = sorted(candidates, key=lambda s: float(delta(s).max()), reverse=True)
    selected = ordered[:2]
    selected.append(min(ordered[2:], key=lambda s: float(delta(s).mean())))
    return [
        dict(
            sample,
            trace_token_start=max(
                sample["prompt_length"] - 1, sample["prompt_length"] - 1 + int(delta(sample).argmax()) - 8
            ),
        )
        for sample in selected
    ]


def assess(output, reference):
    from examples.grpo_trainer.qwen3_8_next.backend_trace import compare_backend_traces

    samples = json.loads((output / "decode-samples.json").read_text())
    values = {phase: json.loads((output / f"{phase}.json").read_text()) for phase in PHASES}
    mega = json.loads((reference / "packing-control/manual0.json").read_text())
    for phase, rows in values.items():
        base = "prefill0" if phase.startswith("prefill") else "batch0" if phase.startswith("batch") else "decode0"
        assert rows.keys() == values[base].keys()
        assert all(
            all(row[key] == values[base][name][key] for key in ("logprobs", "routes_sha256", "cached_tokens"))
            for name, row in rows.items()
        ), f"Scores/route hashes changed in repeat/control {phase}"
    differences, records, traces = {}, [], []
    for sample in samples:
        name = sample["id"]
        scores = {phase: np.asarray(values[phase][name]["logprobs"]) for phase in ("prefill0", "decode0", "batch0")}
        scores.update(
            production_rollout=np.asarray(sample["generation_logprobs"]),
            production_actor=np.asarray(sample["production_actor_logprobs"]),
            hf_megatron=np.asarray(response_logprobs(sample, mega[name]["log_probs"])),
        )
        pairs = [
            (left, right)
            for left in ("prefill0", "decode0", "batch0")
            for right in ("production_rollout", "production_actor", "hf_megatron")
        ]
        pairs += [("decode0", "prefill0"), ("batch0", "decode0")]
        record = dict(id=name, prompt_length=sample["prompt_length"], response_length=sample["response_length"])
        for left, right in pairs:
            delta = np.abs(scores[left] - scores[right])
            assert delta.shape == (sample["response_length"],) and np.isfinite(delta).all()
            key = left + "_vs_" + right
            differences.setdefault(key, []).append(delta)
            record[key] = dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()))
        production_routes = np.load(output / "generation" / sample["generation_routes"], allow_pickle=False)
        record["route_disagreements_vs_production"] = {}
        for phase in ("prefill0", "decode0", "batch0"):
            routes = np.load(output / values[phase][name]["routes"], allow_pickle=False)[: len(production_routes)]
            record["route_disagreements_vs_production"][phase] = int(
                np.any(np.sort(routes, axis=-1) != np.sort(production_routes, axis=-1), axis=-1).sum()
            )
        records.append(record)
        for mode in ("prefill", "decode"):
            repeat = compare_backend_traces(output / (mode + "_trace0"), output / (mode + "_trace1"), name)
            assert repeat["first_nonzero_stage"] is None, f"{mode} activation trace is not repeatable"
        traces.append(compare_backend_traces(output / "prefill_trace0", output / "decode_trace0", name))
    summaries = {
        key: dict(
            response_mean_abs=float(np.mean([x.mean() for x in rows])),
            token_mean_abs=float(np.concatenate(rows).mean()),
            max_abs=float(np.concatenate(rows).max()),
        )
        for key, rows in differences.items()
    }
    report = dict(
        complete=True,
        controls_valid=True,
        full_model_acceptance=False,
        production_changed=False,
        samples=len(samples),
        summaries=summaries,
        records=records,
        diagnostics=traces,
        same_loaded_vllm_weights=True,
        native_v2_trace_replay=True,
        logprobs_mode="processed_logprobs",
        logits_masked=False,
        routes_forced=False,
        scope="Two short production outliers and one short control. Native expert routes; fixed recorded tokens. "
        "HF-loaded vLLM prefill/single decode/three-request decode; not a production refit or acceptance run.",
    )
    write_json(output / "report.json", report)
    print("QWEN38_REAL_PROMPT_DECODE_COMPLETE", json.dumps(summaries), flush=True)


def run(output, model, source, reference):
    from vllm import LLM, SamplingParams

    _, samples = reuse_production_capture(source, output, 16)
    reference_report = json.loads((reference / "report.json").read_text())
    assert reference_report["complete"] and reference_report["controls_valid"]
    assert (reference / "samples.json").read_bytes() == (output / "samples.json").read_bytes()
    config_sha = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    assert all(
        json.loads((reference / f"packing-audit-rank-{rank:02d}.json").read_text())["config_sha256"] == config_sha
        for rank in range(8)
    )
    samples = select_probe_samples(samples)
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
        enable_trace_replay=True,
        worker_extension_cls="examples.grpo_trainer.qwen3_8_next.backend_trace.BackendTraceWorkerExtension",
    )
    policies = engine.collective_rpc("qwen38_qsa_order", kwargs={"policy": "block-id"})
    assert len(policies) == 8 and all(x["policy"] == "block-id" for x in policies)
    write_json(
        output / "runtime.json",
        dict(config_sha256=config_sha, workers=engine.collective_rpc("qwen38_runtime_snapshot")),
    )
    try:
        for phase in PHASES:
            prefill, traced = phase.startswith("prefill"), "trace" in phase
            rows = {}
            if traced:
                engine.collective_rpc(
                    "qwen38_trace", kwargs=dict(action="install", directory=str(output / phase), tokens=16)
                )
            groups = [samples] if phase.startswith("batch") else [[s] for s in samples]
            try:
                for group in groups:
                    assert engine.reset_prefix_cache(reset_connector=True)
                    if traced:
                        # The final generated token is sampled but never fed back
                        # to the model. All captured activation windows precede it.
                        prompt = dict(group[0], trace_query_length=len(group[0]["input_ids"]) - (not prefill))
                        engine.collective_rpc(
                            "qwen38_trace", kwargs=dict(action="start", prompt=prompt, config_sha256=config_sha)
                        )
                    params = [
                        SamplingParams(
                            temperature=1.0,
                            top_p=1.0,
                            top_k=-1,
                            repetition_penalty=1.0,
                            max_tokens=1 if prefill else s["response_length"],
                            ignore_eos=True,
                            logprobs=0,
                            prompt_logprobs=1 if prefill else None,
                            detokenize=False,
                            seed=20261004,
                            trace_decode_token_ids=None if prefill else s["input_ids"][s["prompt_length"] :],
                        )
                        for s in group
                    ]
                    outputs = engine.generate(
                        [{"prompt_token_ids": s["input_ids"] if prefill else s["prompt_ids"]} for s in group],
                        params,
                        use_tqdm=False,
                    )
                    if traced:
                        engine.collective_rpc("qwen38_trace", kwargs={"action": "finish"})
                    for sample, result in zip(group, outputs, strict=True):
                        answer = result.outputs[0]
                        if prefill:
                            logs = [
                                result.prompt_logprobs[i][token].logprob
                                for i, token in enumerate(sample["input_ids"])
                                if i
                            ]
                            logs = response_logprobs(sample, logs)
                        else:
                            assert list(answer.token_ids) == sample["input_ids"][sample["prompt_length"] :]
                            logs = [
                                entry[token].logprob
                                for token, entry in zip(answer.token_ids, answer.logprobs, strict=True)
                            ]
                        assert len(logs) == sample["response_length"] and np.isfinite(logs).all()
                        routes = answer.routed_experts
                        assert routes.shape == (len(sample["input_ids"]) - (not prefill), 48, 10)
                        route_file = f"{phase}-{sample['id']}-routes.npy"
                        np.save(output / route_file, routes)
                        rows[sample["id"]] = dict(
                            logprobs=logs,
                            routes=route_file,
                            routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
                            cached_tokens=result.num_cached_tokens,
                        )
                # Route filenames necessarily differ between repeated phases.
                write_json(output / f"{phase}.json", rows)
            finally:
                if traced:
                    engine.collective_rpc("qwen38_trace", kwargs={"action": "close"})
            print("QWEN38_REAL_PROMPT_DECODE_PHASE", phase, flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    assess(output, reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "model", "source", "reference"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.model, args.source, args.reference)
