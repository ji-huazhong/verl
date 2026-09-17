# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the physical packed CP layout; no model-support claim."""

import pytest
import torch

from verl.models.mcore.qwen3_8_next.ops.context_parallel import PackedContextParallelLayout


@pytest.mark.parametrize("size,bounds", [(1, [0, 3, 8]), (2, [0, 12, 32]), (4, [0, 0, 24, 64, 64])])
def test_packed_cp_roundtrip(size, bounds):
    global_rows = torch.arange(bounds[-1] * 3).reshape(-1, 3)
    layouts = [PackedContextParallelLayout(torch.tensor(bounds), size, rank) for rank in range(size)]
    gathered = torch.cat([layout.local(global_rows) for layout in layouts])
    torch.testing.assert_close(gathered[layouts[0].natural_order], global_rows)
    assert sorted(layouts[0].rank_order.tolist()) == list(range(bounds[-1]))
    if size == 2:
        assert layouts[0].local_indices.tolist() == [0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 27, 28, 29, 30, 31]
        assert layouts[1].local_indices.tolist() == [3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


@pytest.mark.parametrize(
    "bounds,size,rank",
    [([1, 8], 2, 0), ([0, 8, 4], 2, 0), ([0, 7], 2, 0), ([0, 8], 0, 0), ([0, 8], 2, 2)],
)
def test_packed_cp_rejects_invalid_physical_layout(bounds, size, rank):
    with pytest.raises(ValueError):
        PackedContextParallelLayout(torch.tensor(bounds), size, rank)


def test_packed_cp_requires_integer_boundaries_and_exact_global_length():
    with pytest.raises(ValueError, match="integer"):
        PackedContextParallelLayout(torch.tensor([0.0, 8.0]), 2, 0)
    layout = PackedContextParallelLayout(torch.tensor([0, 8]), 2, 0)
    with pytest.raises(ValueError, match="Global token"):
        layout.local(torch.zeros(4, 3))


@pytest.mark.parametrize("size,bounds", [(1, [0, 0, 3, 8]), (2, [0, 0, 4, 16, 60, 60]), (4, [0, 8, 32, 96])])
@pytest.mark.parametrize("depth", [0, 1, 3, 9, 32])
def test_halo_plans_keep_core_order_and_exchange_only_remote_unique_rows(size, bounds, depth):
    from verl.models.mcore.qwen3_8_next.ops.ple_context_parallel import PackedCPHalo

    layouts = [PackedContextParallelLayout(torch.tensor(bounds), size, rank) for rank in range(size)]
    plans = [PackedCPHalo(layout, depth) for layout in layouts]
    for rank, plan in enumerate(plans):
        torch.testing.assert_close(plan.global_indices[plan.core_indices], layouts[rank].local_indices)
        received = []
        for owner, source in enumerate(plans):
            start = sum(source.send_splits[:rank])
            count = source.send_splits[rank]
            rows = layouts[owner].local_indices[source.send_indices[start : start + count]]
            expected = sorted(set(plan.global_indices.tolist()) & set(layouts[owner].local_indices.tolist()))
            if owner == rank:
                expected = []
            assert rows.tolist() == expected
            assert len(rows) == plan.recv_splits[owner]
            received.extend(rows.tolist())
        available = torch.cat((layouts[rank].local_indices, torch.tensor(received, dtype=torch.long)))
        torch.testing.assert_close(available[plan.expand_indices], plan.global_indices)
        assert sum(plan.recv_splits) <= 2 * (len(bounds) - 1) * depth
        # Each expanded chunk is contiguous within one original document.
        for start, end in zip(plan.cu_seqlens.tolist(), plan.cu_seqlens.tolist()[1:], strict=False):
            rows = plan.global_indices[start:end]
            assert rows.tolist() == list(range(int(rows[0]), int(rows[-1]) + 1))
            assert any(lo <= int(rows[0]) <= int(rows[-1]) < hi for lo, hi in zip(bounds, bounds[1:], strict=False))


def test_halo_empty_layout_and_invalid_depth():
    from verl.models.mcore.qwen3_8_next.ops.ple_context_parallel import PackedCPHalo

    layout = PackedContextParallelLayout(torch.tensor([0, 0, 0]), 2, 0)
    plan = PackedCPHalo(layout, 9)
    assert plan.expanded_count == 0 and plan.send_splits == plan.recv_splits == [0, 0]
    for depth in (-1, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            PackedCPHalo(layout, depth)
