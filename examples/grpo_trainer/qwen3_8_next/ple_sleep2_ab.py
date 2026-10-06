# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Isolate PLE context offsets across native V2 level-2 sleep and restoration."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from examples.grpo_trainer.qwen3_8_next.ple_runtime_trace import probe_samples
from examples.grpo_trainer.qwen3_8_next.real_prompt_trace import write_json


def score_difference(left, right):
    deltas = []
    for a, b in zip(left, right, strict=True):
        if (a["id"], a["repeat"], a["input_ids"]) != (b["id"], b["repeat"], b["input_ids"]):
            raise ValueError("Sleep A/B phases differ in sample identity or tokens")
        delta = np.abs(np.asarray(a["logprobs"]) - b["logprobs"])
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite sleep A/B scores")
        deltas.append(delta)
    return dict(token_mean_abs=float(np.concatenate(deltas).mean()), max_abs=float(np.concatenate(deltas).max()))


def safe_context_states(states):
    """Do not execute a corrupted gather if offsets could address outside history."""
    return len(states) == 8 and all(
        len(row["after"]) == len(row["expected"])
        and all(min(row["expected"]) <= offset <= 0 for offset in row["after"])
        for row in states
    )


def run(output, model, *, synchronous_schedule=False):
    if synchronous_schedule:
        # Enqueue the whole batch before stepping, so arrival timing cannot
        # change the GEMM shapes between the traced and untraced phases.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    samples = probe_samples(json.loads((output / "samples.json").read_text()))
    batch = [dict(sample, repeat=repeat) for repeat in range(4) for sample in samples]
    plan = output / "ple-plan.json"
    write_json(
        plan,
        dict(
            prompts=list({tuple(s["prompt_ids"]): s["prompt_ids"] for s in samples}.values()),
            response_queries=16,
            layer_limit=2,
            ple_substages=True,
            prompt_tail=16,
            ple_cache_queries=8,
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
        # Native worker backs up model parameters to CPU for this experiment.
        # ModelState tensors are excluded from parameter backup; the fixed
        # wheel preserves them independently through its runtime pool.
        sleep_preserve_parameter_names=["*"],
        engram_config={"cpu_offload": True, "dp_shared_memory": False},
        gpu_memory_utilization=0.55,
        logprobs_mode="processed_logprobs",
        enable_trace_replay=True,
        worker_extension_cls="examples.grpo_trainer.qwen3_8_next.backend_trace.BackendTraceWorkerExtension",
    )
    results, context, audits, skipped = {}, {}, {}, {}
    schedules, active_schedule = {}, []
    if synchronous_schedule:
        client = engine.llm_engine.engine_core
        assert type(client).__name__ == "InprocClient"
        scheduler = client.engine_core.scheduler
        original_schedule = scheduler.schedule

        def record_schedule(*args, **kwargs):
            scheduled = original_schedule(*args, **kwargs)
            if scheduled.num_scheduled_tokens:
                active_schedule.append(dict(scheduled.num_scheduled_tokens))
            return scheduled

        scheduler.schedule = record_schedule

    def generate(phase, traced):
        active_schedule.clear()
        assert engine.reset_prefix_cache(reset_connector=True)
        if traced:
            engine.collective_rpc(
                "qwen38_production_trace",
                kwargs=dict(action="install", plan=str(plan), directory=str(output / phase)),
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
                        max_tokens=n,
                        ignore_eos=True,
                        logprobs=0,
                        detokenize=False,
                        seed=20261004,
                        trace_decode_token_ids=tokens,
                    )
                )
                inputs.append({"prompt_token_ids": sample["prompt_ids"]})
            generated = engine.generate(inputs, params, use_tqdm=False)
            rows = []
            for sample, result in zip(batch, generated, strict=True):
                p, n = sample["prompt_length"], sample["response_length"]
                tokens = sample["input_ids"][p : p + n]
                answer = result.outputs[0]
                assert list(answer.token_ids) == tokens
                scores = [row[token].logprob for token, row in zip(tokens, answer.logprobs, strict=True)]
                assert len(scores) == n and np.isfinite(scores).all() and result.num_cached_tokens == 0
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
            if synchronous_schedule:
                assert active_schedule, "Scheduler recorder observed no execution"
                slots = {row["request_id"]: i for i, row in enumerate(rows)}
                schedules[phase] = [
                    sorted((slots[identity.split("-", 1)[0]], count) for identity, count in step.items())
                    for step in active_schedule
                ]
                assert {slot for step in schedules[phase] for slot, _ in step} == set(range(len(rows)))
                write_json(output / "schedules.json", schedules)
        finally:
            if traced:
                engine.collective_rpc("qwen38_production_trace", kwargs={"action": "close"})
        print("QWEN38_PLE_SLEEP2_PHASE", phase, flush=True)

    try:
        policies = engine.collective_rpc("qwen38_qsa_order", kwargs={"policy": "block-id"})
        assert len(policies) == 8 and all(p["policy"] == "block-id" for p in policies)
        write_json(
            output / "runtime.json",
            dict(
                workers=engine.collective_rpc("qwen38_runtime_snapshot"),
                synchronous_schedule=synchronous_schedule,
                engine_core_client=type(engine.llm_engine.engine_core).__name__,
            ),
        )
        context["cold"] = engine.collective_rpc("qwen38_ple_context_state")
        assert len(context["cold"]) == 8 and all(row["after"] == row["expected"] for row in context["cold"])
        generate("decode_control", False)
        if synchronous_schedule:
            generate("decode_control_repeat", False)
        generate("decode_trace", True)
        audits["before"] = engine.collective_rpc(
            "qwen38_sleep_weight_audit", kwargs=dict(stage="before", directory=str(output / "weight-audit"))
        )
        engine.sleep(level=2)
        engine.wake_up()
        context["after_wake"] = engine.collective_rpc("qwen38_ple_context_state")
        write_json(output / "context-states.json", context)
        print("QWEN38_PLE_SLEEP2_CONTEXT", json.dumps(context), flush=True)
        audits["after"] = engine.collective_rpc(
            "qwen38_sleep_weight_audit", kwargs=dict(stage="after", directory=str(output / "weight-audit"))
        )
        write_json(output / "weight-audit.json", audits)
        assert len(audits["after"]) == 8 and all(
            row["comparison"] == dict(missing=[], added=[], changed={}) for row in audits["after"]
        ), "Model tensors changed across sleep: cannot isolate context offsets"
        if safe_context_states(context["after_wake"]):
            generate("wake2_trace", True)
        else:
            skipped["wake2_trace"] = "Corrupt offsets could address outside request history; unsafe gather skipped"
        context["repair"] = engine.collective_rpc("qwen38_ple_context_state", kwargs=dict(repair=True))
        write_json(output / "context-states.json", context)
        assert all(row["after"] == row["expected"] for row in context["repair"])
        generate("repair_trace", True)
        generate("repair_repeat", False)
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    differences = {
        phase: score_difference(rows, results["decode_control"])
        for phase, rows in results.items()
        if phase != "decode_control"
    }
    references = {}
    for field in ("production_actor_logprobs", "generation_logprobs"):
        reference = [
            dict(row, logprobs=sample[field][: len(row["logprobs"])])
            for row, sample in zip(results["decode_control"], batch, strict=True)
        ]
        references[field] = {phase: score_difference(rows, reference) for phase, rows in results.items()}
    fixed_state_wheel = os.environ.get("QWEN38_PRECISION_PROFILE") == "hc-gdn-fp32-ple-state"
    report = dict(
        complete=True,
        samples=len(samples),
        batch_requests=len(batch),
        model_tensors_preserved=True,
        context_offsets_changed=any(row["after"] != row["expected"] for row in context["after_wake"]),
        trace_control_passed=differences["decode_trace"]["max_abs"] == 0,
        differences_vs_decode_control=differences,
        differences_vs_production=references,
        repaired_repeat_difference=score_difference(results["repair_trace"], results["repair_repeat"]),
        skipped=skipped,
        full_precision_accepted=False,
        production_changed=False,
        fixed_state_wheel=fixed_state_wheel,
        synchronous_schedule=synchronous_schedule,
        schedules_equal=(
            all(value == schedules["decode_control"] for value in schedules.values()) if synchronous_schedule else None
        ),
        scope="Native V2 TP8; same real tokens and retained model tensors, level-2 sleep. "
        "An offset-only diagnostic repair follows the untouched wake phase; no training/refit. "
        "The fixed-state wheel is recorded separately from the original baseline.",
    )
    write_json(output / "report.json", report)
    if synchronous_schedule:
        assert report["schedules_equal"], "Diagnostic phases used different batch schedules"
        assert differences["decode_control_repeat"]["max_abs"] == 0, "Untraced fixed-schedule repeat drifted"
    if fixed_state_wheel:
        assert not report["context_offsets_changed"], "Fixed wheel still loses PLE context offsets"
        assert report["trace_control_passed"], "Tracing changed the cold control"
        assert not skipped, "Fixed wheel could not safely evaluate every phase"
    print("QWEN38_PLE_SLEEP2_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--synchronous-schedule", action="store_true")
    args = parser.parse_args()
    run(args.output, args.model, synchronous_schedule=args.synchronous_schedule)
