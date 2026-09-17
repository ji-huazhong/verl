# SPDX-License-Identifier: Apache-2.0
"""Packed-sequence metadata shared by PLE and QSA (no Megatron/CUDA imports)."""

import torch


def packed_token_segments(cu_seqlens, total):
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain a start and an end")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be integer offsets")
    if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != total:
        raise ValueError("cu_seqlens must cover the complete token buffer")
    if bool((cu_seqlens[1:] < cu_seqlens[:-1]).any()):
        raise ValueError("cu_seqlens must be nondecreasing")
    # right=True skips zero-length segments, including duplicate starts and a
    # trailing empty segment. Marker/cumsum implementations mis-handle both.
    token = torch.arange(total, device=cu_seqlens.device)
    segments = torch.searchsorted(cu_seqlens[1:].contiguous(), token, right=True)
    return segments, token - cu_seqlens[:-1][segments]


def apply_indexer_rope(x, frequencies):
    """Use the attention's already-composed absolute mRoPE angles on QSA heads."""
    while frequencies.ndim > 2 and frequencies.shape[1] == 1:
        frequencies = frequencies.squeeze(1)
    if frequencies.ndim != 2 or frequencies.shape[0] != x.shape[0]:
        raise ValueError("QSA requires one composed rotary angle vector per token")
    dim = frequencies.shape[-1]
    if dim % 2 or dim > x.shape[-1]:
        raise ValueError("Invalid QSA rotary dimension")
    rotated, rest = x[..., :dim], x[..., dim:]
    half = dim // 2
    rotate_half = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
    angles = frequencies.unsqueeze(1)
    result = rotated.float() * angles.cos() + rotate_half.float() * angles.sin()
    return torch.cat((result.to(x.dtype), rest), dim=-1)
