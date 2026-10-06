# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Check normalization shifts and probe restoration using the production mask."""

import ast
from pathlib import Path
from types import MethodType
from typing import Optional

import pytest
import torch

from examples.grpo_trainer.qwen3_8_next.vocab_mask_ab import VocabularyMaskProbe, mask_statistics


def production_patch():
    path = Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout/utils.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if getattr(node, "name", "") == "monkey_patch_compute_logits")
    scope = dict(torch=torch, MethodType=MethodType, Optional=Optional)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[function.name]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 8.0, 9.0]]))

    def compute_logits(self, hidden_states):
        return hidden_states @ self.weight


def test_actual_mask_shift_explains_allowed_token_probability_without_changing_forward():
    model = FakeModel()
    hidden = torch.ones(2, 1)
    original = model.compute_logits
    native = original(hidden)
    statistics = mask_statistics(native, 3, [2])
    probe = VocabularyMaskProbe(model, 3, [2], production_patch())
    records = []
    try:
        for policy in ("native", "tail", "production", "production", "native"):
            probe.select(policy)
            output = model.compute_logits(hidden)
            shift = output.log_softmax(-1)[:, 0] - native.log_softmax(-1)[:, 0]
            expected = torch.zeros(2) if policy == "native" else torch.tensor(statistics[policy + "_logprob_shift"])
            torch.testing.assert_close(shift, expected)
            assert torch.equal(output[:, :2], native[:, :2])
            records.append(probe.drain())
        assert all(record == records[0] for record in records)
        assert statistics["production_logprob_shift"][0] > 7
    finally:
        probe.close()
    assert model.compute_logits == original
    assert torch.equal(model.compute_logits(hidden), native)


def test_no_mask_has_zero_shift_and_statistics_do_not_mutate_logits():
    values = torch.randn(4, 8).bfloat16()
    saved = values.clone()
    result = mask_statistics(values, 8, [])
    assert result["tail_logprob_shift"] == result["production_logprob_shift"] == [0] * 4
    assert torch.equal(values, saved)


def test_probe_detects_weight_changes_and_rejects_undrained_phase():
    model = FakeModel()
    probe = VocabularyMaskProbe(model, 3, [], production_patch())
    try:
        probe.select("native")
        model.compute_logits(torch.ones(1, 1))
        with pytest.raises(ValueError, match="Drain"):
            probe.select("tail")
        with torch.no_grad():
            model.weight.add_(1)
        with pytest.raises(AssertionError):
            probe.drain()
    finally:
        probe.close()
