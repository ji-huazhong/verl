# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Controls must retain differences beyond the captured activation prefix."""

from copy import deepcopy

import pytest

from examples.grpo_trainer.qwen3_8_next.repetition import compare_repeated_logprobs


def result():
    return dict(
        complete=True,
        config_sha256="config",
        prompt_sha256="prompts",
        backend="megatron",
        dtype="bfloat16",
        parallelism=dict(tp=8, pp=4, ep=8),
        records=[dict(id="long", input_ids=list(range(2304)), logprobs=[-1.0] * 2303)],
    )


def test_control_keeps_tail_failure_separate_from_exact_prefix():
    reference = result()
    candidate = deepcopy(reference)
    candidate["records"][0]["logprobs"][2051] = -1.125
    report = compare_repeated_logprobs(reference, candidate)
    assert report["all"]["max_abs"] == 0.125
    assert report["records"][0]["all"]["first_changed_index"] == 2051
    assert report["records"][0]["prefix31"]["changed"] == 0
    assert report["records"][0]["prefix2048"]["changed"] == 0


@pytest.mark.parametrize("failure", ["incomplete", "missing_token", "different_tokens", "nonfinite"])
def test_control_rejects_incomparable_or_invalid_results(failure):
    reference = result()
    candidate = deepcopy(reference)
    if failure == "incomplete":
        candidate["complete"] = False
    elif failure == "missing_token":
        candidate["records"][0]["logprobs"].pop()
    elif failure == "different_tokens":
        candidate["records"][0]["input_ids"][0] += 1
    else:
        candidate["records"][0]["logprobs"][10] = float("nan")
    with pytest.raises(ValueError):
        compare_repeated_logprobs(reference, candidate)
