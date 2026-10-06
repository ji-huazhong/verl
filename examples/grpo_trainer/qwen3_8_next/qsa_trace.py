# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Compare actual QSA packed selections, separating order from membership."""

import torch


def canonical_qsa_selection(selection, *, token_topk, compress_ratio, has_count_column):
    """Keep the original ordering while putting both backend formats on CPU."""
    if selection.device.type != "cpu" or selection.ndim != 2 or selection.dtype not in (torch.int32, torch.int64):
        raise ValueError("Normalize QSA selections on CPU after the original forward")
    width = token_topk + compress_ratio - 1
    packed = torch.full((selection.shape[0], width + 1), -1, dtype=selection.dtype)
    for row, values in enumerate(selection):
        if has_count_column:
            count = int(values[-1])
            if not 0 <= count < values.numel() or (values[count:-1] != -1).any():
                raise ValueError("Invalid packed QSA selection")
            indices = values[:count]
            if (indices < 0).any():
                raise ValueError("Negative token in the valid QSA selection")
        else:
            indices = values[values >= 0]
        if indices.numel() > width:
            raise ValueError("QSA selection exceeded the token budget plus local tail")
        packed[row, : indices.numel()] = indices
        packed[row, -1] = indices.numel()
    return packed


def qsa_selection_differences(left, right, token_start):
    if left.device.type != "cpu" or right.device.type != "cpu" or left.shape != right.shape or left.ndim != 2:
        raise ValueError("QSA selections require equal CPU token coordinates")
    if left.dtype not in (torch.int32, torch.int64) or right.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA selections must contain integer indices")
    ordered, membership, counts = [], [], []
    duplicate_rows = [0, 0]
    for row, (a, b) in enumerate(zip(left, right, strict=True)):
        selections = []
        for side, value in enumerate((a, b)):
            count = int(value[-1])
            if not 0 <= count < value.numel():
                raise ValueError("Invalid QSA valid-entry count")
            indices = value[:count]
            if (indices < 0).any() or (indices > token_start + row).any() or (value[count:-1] != -1).any():
                raise ValueError("Invalid causal QSA selection or padding")
            if indices.unique().numel() != count:
                duplicate_rows[side] += 1
            selections.append(indices)
        a, b = selections
        counts.append(a.numel() != b.numel())
        ordered.append(not torch.equal(a, b))
        membership.append(not torch.equal(a.unique(sorted=True), b.unique(sorted=True)))

    def first(values):
        return next((token_start + i for i, changed in enumerate(values) if changed), None)

    return dict(
        actual_selection_recorded=True,
        token_start=token_start,
        rows=len(ordered),
        order_changed_rows=sum(ordered),
        selected_set_changed_rows=sum(membership),
        valid_count_changed_rows=sum(counts),
        first_order_changed_token=first(ordered),
        first_set_changed_token=first(membership),
        duplicate_rows_reference=duplicate_rows[0],
        duplicate_rows_candidate=duplicate_rows[1],
    )


def qsa_score_boundaries(scores, visible_blocks, selections, *, token_start, compress_ratio, token_topk):
    """Check real selected blocks against the original FP32 ranking threshold."""
    if scores.device.type != "cpu" or selections.device.type != "cpu" or visible_blocks.device.type != "cpu":
        raise ValueError("Analyze QSA scores on CPU")
    if scores.shape[0] != selections.shape[0] or visible_blocks.numel() != scores.shape[0]:
        raise ValueError("QSA score and selection coordinates differ")
    block_topk = token_topk // compress_ratio
    rows = []
    for index, (score, visible, selected) in enumerate(zip(scores, visible_blocks.flatten(), selections, strict=True)):
        visible = int(visible)
        if not 0 < visible <= score.numel() or not torch.isfinite(score[:visible]).all():
            raise ValueError("Invalid visible QSA scores")
        count = min(visible, block_topk)
        groups = selected[: count * compress_ratio].reshape(count, compress_ratio)
        blocks = groups[:, 0] // compress_ratio
        if not torch.equal(groups, blocks[:, None] * compress_ratio + torch.arange(compress_ratio)[None, :]):
            raise ValueError("QSA indices are not complete compressed blocks")
        if (blocks < 0).any() or (blocks >= visible).any():
            raise ValueError("QSA selected an invisible compressed block")
        ordered = score[:visible].topk(min(count + 1, visible)).values
        threshold = ordered[count - 1]
        gap = float(threshold - ordered[count]) if count < visible else None
        selected_min = score[blocks].min()
        rows.append(
            dict(
                token=token_start + index,
                visible_blocks=visible,
                threshold=float(threshold),
                next_score_gap=gap,
                boundary_tie_count=int((score[:visible] == threshold).sum()),
                selected_below_threshold=bool(selected_min < threshold),
            )
        )
    return rows
