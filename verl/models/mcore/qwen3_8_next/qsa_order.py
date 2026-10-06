# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Canonical QSA block traversal without changing native top-k membership."""

import torch


def canonical_block_order(indices):
    """Sort valid block ids ascending and retain negative padding at the end.

    Native QSA block ids are bounded by the score matrix width, far below the
    signed integer maximum used for padding during this diagnostic sort.
    """
    if indices.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA block indices must be a two-dimensional signed integer tensor")
    sentinel = torch.iinfo(indices.dtype).max
    ordered = indices.masked_fill(indices < 0, sentinel).sort(dim=-1).values
    return ordered.masked_fill(ordered == sentinel, -1)


class QsaBlockOrderProbe:
    """Call the original selector, then only permute its resulting block ids."""

    def __init__(self, native):
        self.native = native
        self.calls = 0

    def __call__(self, logits, visible_blocks, token_topk, compress_ratio, block_indices, workspace):
        result = self.native(logits, visible_blocks, token_topk, compress_ratio, block_indices, workspace)
        block_indices.copy_(canonical_block_order(block_indices))
        self.calls += 1
        return result
