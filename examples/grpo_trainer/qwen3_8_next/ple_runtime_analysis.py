# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Compare native PLE boundaries and verify decode cache against observed history."""

import argparse
import json
import re
from pathlib import Path

import torch

from examples.grpo_trainer.qwen3_8_next.production_trace_analysis import load_verified, merge_stages

PREFIX = "layers/01/ple/"
PLE_STAGES = ("input", "ngram_ids", "embedding", "key", "value", "gated", "conv_input", "output")


def difference(left, right):
    if left.shape != right.shape or not left.isfinite().all() or not right.isfinite().all():
        raise ValueError("PLE tensors differ in shape or contain nonfinite values")
    delta = right.double() - left.double()
    return dict(
        relative_l2=(delta.norm() / left.double().norm().clamp_min(1e-30)).item(),
        max_abs=delta.abs().max().item(),
        mean_abs=delta.abs().mean().item(),
        differing_elements=int(torch.count_nonzero(delta)),
    )


def verify_decode_cache(stages, prompt_length, expected_queries):
    """A decode cache must be the previous W native conv-input vectors.

    This test uses the same backend's recorded inputs, independently of
    Megatron, logprob drift, or changes propagated from earlier layers.
    """
    inputs = stages[PREFIX + "conv_input"]
    states, flags = stages[PREFIX + "cache_before"], stages[PREFIX + "cache_slot_and_valid"]
    positions = list(range(prompt_length, prompt_length + expected_queries))
    records = []
    for position in positions:
        if position not in states or position not in flags or position not in inputs:
            raise ValueError(f"Missing decode state/input at query {position}")
        width, remainder = divmod(states[position].numel(), inputs[position].numel())
        if not width or remainder:
            raise ValueError("PLE cache width does not match its channel count")
        if flags[position].tolist()[1] != 1:
            raise ValueError("A real decode after the prompt lost its initial PLE state")
        history = list(range(position - width, position))
        if not all(index in inputs for index in history):
            raise ValueError("PLE prompt-tail capture does not cover the complete convolution history")
        expected = torch.stack([inputs[index] for index in history], dim=-1).flatten()
        actual = states[position]
        if actual.dtype != expected.dtype:
            raise ValueError("PLE state dtype differs from the native conv input")
        records.append(
            dict(
                query_position=position,
                slot=int(flags[position][0]),
                history_tokens=width,
                **difference(expected, actual),
            )
        )
    return dict(
        queries_checked=len(records), bitwise_equal=all(r["differing_elements"] == 0 for r in records), records=records
    )


def load_request(root, phase, response):
    pattern = re.compile(re.escape(response["request_id"]) + r"(?:-[0-9a-f]{8})?")
    matches = []
    for path in (root / phase / "vllm/replica-000/rank-000").glob("*/complete.json"):
        metadata = json.loads(path.read_text())
        if pattern.fullmatch(metadata["identity"]):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous native request {phase}/{response['request_id']}")
    record = load_verified(matches[0])
    if record["metadata"]["prompt_ids"] != response["prompt_ids"]:
        raise ValueError("Trace prompt differs from scored response")
    return merge_stages([record], response["input_ids"])


def assess(root):
    control = json.loads((root / "report.json").read_text())
    if not control["complete"] or not control["trace_control_passed"]:
        raise ValueError("Passive trace has not passed its untraced score control")
    phases = ("decode_trace", "prefill_trace", "wake_trace")
    responses = {phase: json.loads((root / (phase + ".json")).read_text()) for phase in phases}
    keys = [(row["id"], row["repeat"]) for row in responses[phases[0]]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicated diagnostic sample identity")
    if any([(row["id"], row["repeat"]) for row in responses[phase]] != keys for phase in phases):
        raise ValueError("Diagnostic phases differ in sample order or membership")
    records = []
    for index, (sample_id, repeat) in enumerate(keys):
        native = {phase: load_request(root, phase, responses[phase][index]) for phase in phases}
        reference = responses["decode_trace"][index]
        if any(responses[phase][index]["input_ids"] != reference["input_ids"] for phase in phases):
            raise ValueError("Diagnostic phases fed different token sequences")
        p, n = len(reference["prompt_ids"]), len(reference["logprobs"])
        positions = list(range(p - 1, p - 1 + n))
        comparisons = {}
        for phase in ("prefill_trace", "wake_trace"):
            comparisons[phase + "_vs_decode"] = {}
            for stage in PLE_STAGES:
                name = PREFIX + stage
                if not all(
                    position in native[phase][name] and position in native["decode_trace"][name]
                    for position in positions
                ):
                    raise ValueError(f"Incomplete native PLE query coverage at {stage}")
                left = torch.stack([native["decode_trace"][name][position] for position in positions])
                right = torch.stack([native[phase][name][position] for position in positions])
                metrics = difference(left, right)
                metrics["last_prefill_query"] = difference(left[0], right[0])
                metrics["first_decode_query"] = difference(left[1], right[1])
                comparisons[phase + "_vs_decode"][stage] = metrics
        caches = {
            phase: verify_decode_cache(native[phase], p, min(16, n - 1)) for phase in ("decode_trace", "wake_trace")
        }
        records.append(
            dict(id=sample_id, repeat=repeat, prompt_length=p, queries=n, comparisons=comparisons, cache_checks=caches)
        )
    return dict(
        complete=True,
        samples=len(set(key[0] for key in keys)),
        requests=len(keys),
        native_cache_history_all_equal=all(c["bitwise_equal"] for r in records for c in r["cache_checks"].values()),
        records=records,
        full_precision_accepted=False,
        scope="Same loaded vLLM weights and actual native PLE inputs; rank-0 replicated PLE activations. "
        "Cache checks test state history independently of cross-backend propagation. "
        "An unequal downstream output alone does not identify a faulty operator.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.root)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))
