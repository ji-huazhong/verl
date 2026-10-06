# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Validate the isolated history-state intervention and score pairing."""

from types import SimpleNamespace

import pytest
import torch

from examples.grpo_trainer.qwen3_8_next.backend_trace import BackendTraceWorkerExtension
from examples.grpo_trainer.qwen3_8_next.ple_sleep2_ab import safe_context_states, score_difference


def test_context_audit_is_read_only_then_repairs_in_place():
    state = type("Qwen4ExpModelState", (), {})()
    state.uses_ngram_embedding = True
    state.ngram_context_len = 2
    state.ngram_context_offsets = torch.zeros(2, dtype=torch.int64)
    state.ngram_context = torch.tensor([[42, 43]], dtype=torch.int32)
    original = state.ngram_context_offsets
    pointer = original.data_ptr()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model_state=state))
    audit = BackendTraceWorkerExtension.qwen38_ple_context_state(worker)
    assert audit["before"] == audit["after"] == [0, 0]
    assert audit["expected"] == [-2, -1] and not audit["repaired"]
    repaired = BackendTraceWorkerExtension.qwen38_ple_context_state(worker, repair=True)
    assert repaired["before"] == [0, 0] and repaired["after"] == [-2, -1]
    assert state.ngram_context_offsets is original and original.data_ptr() == pointer
    assert state.ngram_context.tolist() == [[42, 43]]


@pytest.mark.parametrize("offsets", [[-3, -1], [0, 100000], [0]])
def test_unsafe_context_cannot_launch_a_gather(offsets):
    assert not safe_context_states([dict(after=offsets, expected=[-2, -1])] * 8)


def test_score_comparison_rejects_different_tokens_or_sample_slots():
    row = dict(id="production-013", repeat=0, input_ids=[7, 9], logprobs=[-2.0])
    assert score_difference([row], [row]) == dict(token_mean_abs=0.0, max_abs=0.0)
    for changed in [dict(row, repeat=1), dict(row, input_ids=[7, 8])]:
        with pytest.raises(ValueError, match="identity or tokens"):
            score_difference([row], [changed])
