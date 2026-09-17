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
