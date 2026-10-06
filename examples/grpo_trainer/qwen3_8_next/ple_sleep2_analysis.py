# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Verify the PLE offset intervention using native, checksum-verified snapshots."""

import argparse
import json
from pathlib import Path

import torch

from examples.grpo_trainer.qwen3_8_next.ple_runtime_analysis import (
    PLE_STAGES,
    PREFIX,
    difference,
    load_request,
    verify_decode_cache,
)


def assess(root, *, allow_trace_control_drift=False):
    report = json.loads((root / "report.json").read_text())
    if (
        not report["complete"]
        or not report["model_tensors_preserved"]
        or (not report["trace_control_passed"] and not allow_trace_control_drift)
    ):
        raise ValueError("Sleep intervention lacks completed trace and weight controls")
    audits = json.loads((root / "weight-audit.json").read_text())
    if len(audits["after"]) != 8 or any(
        row["comparison"] != dict(missing=[], added=[], changed={}) for row in audits["after"]
    ):
        raise ValueError("Named model tensors differ across sleep")
    phases = ["decode_trace"]
    if "wake2_trace" not in report["skipped"]:
        phases.append("wake2_trace")
    phases.append("repair_trace")
    responses = {phase: json.loads((root / (phase + ".json")).read_text()) for phase in phases}
    keys = [(row["id"], row["repeat"]) for row in responses["decode_trace"]]
    if len(keys) != len(set(keys)) or len(keys) != report["batch_requests"]:
        raise ValueError("Invalid response membership")
    if any([(row["id"], row["repeat"]) for row in responses[phase]] != keys for phase in phases):
        raise ValueError("Response ordering differs across phases")
    names = [
        "layers/00/attn_hc/mixed",
        "layers/00/attention/output",
        "layers/00/mlp/output",
        *(PREFIX + name for name in PLE_STAGES),
        "layers/01/attn_hc/mixed",
    ]
    records = []
    for index, (sample_id, repeat) in enumerate(keys):
        reference = responses["decode_trace"][index]
        if any(
            responses[phase][index]["input_ids"] != reference["input_ids"]
            or responses[phase][index]["cached_tokens"] != 0
            for phase in phases
        ):
            raise ValueError("Input tokens or fresh-prefix conditions differ")
        native = {phase: load_request(root, phase, responses[phase][index]) for phase in phases}
        p = len(reference["prompt_ids"])
        n = min(16, len(reference["logprobs"]))
        positions = list(range(p - 1, p - 1 + n))
        comparisons = {}
        for phase in phases[1:]:
            stages = {}
            for name in names:
                if not all(
                    position in native[phase][name] and position in native["decode_trace"][name]
                    for position in positions
                ):
                    raise ValueError(f"Missing causal query in {phase}/{name}")
                left = torch.stack([native["decode_trace"][name][position] for position in positions])
                right = torch.stack([native[phase][name][position] for position in positions])
                stages[name] = dict(
                    aggregate=difference(left, right),
                    last_prefill_query=difference(left[0], right[0]),
                    first_decode_query=difference(left[1], right[1]),
                    changed_queries=[
                        position for position, a, b in zip(positions, left, right, strict=True) if not torch.equal(a, b)
                    ],
                )
            comparisons[phase] = stages
        records.append(
            dict(
                id=sample_id,
                repeat=repeat,
                prompt_length=p,
                captured_queries=n,
                comparisons_vs_cold=comparisons,
                cache_checks={phase: verify_decode_cache(native[phase], p, min(8, n - 1)) for phase in phases},
            )
        )
    return dict(
        complete=True,
        samples=report["samples"],
        requests=len(keys),
        runtime_report=report,
        trace_control_passed=report["trace_control_passed"],
        isolated_numerical_attribution_valid=report["trace_control_passed"],
        native_cache_history_all_equal=all(c["bitwise_equal"] for r in records for c in r["cache_checks"].values()),
        records=records,
        full_precision_accepted=False,
        scope="Native V2 level-2 lifecycle, same model tensors and tokens; compare traced phases. "
        "First 16 causal response queries, rank-0 replicated PLE and HC values. "
        "Native cache history checked independently of embedding changes. If the untraced control drifts, "
        "these measurements cannot attribute numerical differences solely to sleep or the offset repair.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-trace-control-drift", action="store_true")
    args = parser.parse_args()
    result = assess(args.root, allow_trace_control_drift=args.allow_trace_control_drift)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))
