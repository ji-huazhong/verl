# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""A native PLE cache must contain the exact preceding convolution inputs."""

import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[3] / "examples/grpo_trainer/qwen3_8_next/ple_runtime_analysis.py"
_spec = importlib.util.spec_from_file_location("qwen38_ple_runtime_analysis", _path)
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def cache_fixture():
    inputs = {p: torch.tensor([p * 2.0, p * p + 1.0], dtype=torch.bfloat16) for p in range(10)}
    caches = {p: torch.stack([inputs[q] for q in range(p - 3, p)], dim=-1).flatten() for p in [6, 7]}
    return {
        analysis.PREFIX + "conv_input": inputs,
        analysis.PREFIX + "cache_before": caches,
        analysis.PREFIX + "cache_slot_and_valid": {p: torch.tensor([42, 1]) for p in [6, 7]},
    }


def test_correct_native_cache_matches_previous_vectors_exactly():
    report = analysis.verify_decode_cache(cache_fixture(), 6, 2)
    assert report["bitwise_equal"] and report["queries_checked"] == 2
    assert [row["history_tokens"] for row in report["records"]] == [3, 3]


def test_shifted_or_cross_request_cache_is_a_local_state_failure():
    stages = cache_fixture()
    stages[analysis.PREFIX + "cache_before"][6] = stages[analysis.PREFIX + "cache_before"][7].clone()
    report = analysis.verify_decode_cache(stages, 6, 2)
    assert not report["bitwise_equal"]
    assert report["records"][0]["differing_elements"] == 6
    assert report["records"][1]["differing_elements"] == 0


@pytest.mark.parametrize("missing", ["history", "state", "valid"])
def test_missing_or_invalid_state_is_not_reported_as_equal(missing):
    stages = cache_fixture()
    if missing == "history":
        del stages[analysis.PREFIX + "conv_input"][3]
    elif missing == "state":
        del stages[analysis.PREFIX + "cache_before"][6]
    else:
        stages[analysis.PREFIX + "cache_slot_and_valid"][6][1] = 0
    with pytest.raises(ValueError):
        analysis.verify_decode_cache(stages, 6, 2)
