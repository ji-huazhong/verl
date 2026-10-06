# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from examples.grpo_trainer.qwen3_8_next.qsa_trace import (
    canonical_qsa_selection,
    qsa_score_boundaries,
    qsa_selection_differences,
)


def test_cross_backend_selection_formats_keep_tail_tokens_and_membership():
    megatron = torch.tensor([[2, 3, -1, -1, 0, 1, 4, -1], [0, 1, 2, 3, -1, -1, -1, -1]])
    vllm = torch.tensor([[0, 1, 2, 3, 4, 5], [2, 3, 0, 1, -1, 4]])
    left = canonical_qsa_selection(megatron, token_topk=4, compress_ratio=2, has_count_column=False)
    right = canonical_qsa_selection(vllm, token_topk=4, compress_ratio=2, has_count_column=True)
    report = qsa_selection_differences(left, right, token_start=4)
    assert report["order_changed_rows"] == 2
    assert report["selected_set_changed_rows"] == report["valid_count_changed_rows"] == 0


def test_selection_order_is_distinct_from_membership_and_count_column():
    left = torch.tensor([[0, 1, 2, -1, 3], [0, 2, 3, -1, 3]], dtype=torch.int32)
    right = torch.tensor([[2, 0, 1, -1, 3], [0, 2, 4, -1, 3]], dtype=torch.int32)
    report = qsa_selection_differences(left, right, token_start=3)
    assert report["actual_selection_recorded"]
    assert report["order_changed_rows"] == 2
    assert report["selected_set_changed_rows"] == 1
    assert report["valid_count_changed_rows"] == 0
    assert report["first_order_changed_token"] == 3
    assert report["first_set_changed_token"] == 4


@pytest.mark.parametrize("invalid", [[0, 1, 2, -1, 5], [0, 1, 9, -1, 3], [0, 1, 2, 0, 3]])
def test_selection_rejects_bad_counts_future_indices_or_padding(invalid):
    values = torch.tensor([invalid], dtype=torch.int32)
    with pytest.raises(ValueError):
        qsa_selection_differences(values, values, token_start=3)


def test_score_boundary_distinguishes_valid_ties_from_wrong_selection():
    scores = torch.tensor([[5.0, 4.0, 3.0, 3.0, 3.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    selections = torch.tensor([[0, 1, 2, 3, 4, 5, 8, 9, -1, 8], [2, 3, 4, 5, 6, 7, 8, 9, -1, 8]])
    rows = qsa_score_boundaries(
        scores,
        torch.tensor([5, 5]),
        selections,
        token_start=9,
        compress_ratio=2,
        token_topk=8,
    )
    assert rows[0]["boundary_tie_count"] == 3
    assert rows[0]["next_score_gap"] == 0
    assert not rows[0]["selected_below_threshold"]
    assert rows[1]["next_score_gap"] == 1
    assert rows[1]["selected_below_threshold"]
