# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fused CUDA fake-quant kernel for integer INT4 QAT."""

import torch
import triton
import triton.language as tl


@triton.jit
def _round_to_nearest_even(value):
    lower = tl.floor(value)
    fraction = value - lower
    lower_i32 = lower.to(tl.int32)
    tie_rounded = tl.where((lower_i32 & 1) == 0, lower, lower + 1.0)
    return tl.where(fraction < 0.5, lower, tl.where(fraction > 0.5, lower + 1.0, tie_rounded))


@triton.jit
def _int4_fake_quant_kernel(
    input_ptr,
    output_ptr,
    numel,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE_BF16: tl.constexpr,
):
    group_id = tl.program_id(0)
    offsets = group_id * GROUP_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (tl.arange(0, BLOCK_SIZE) < GROUP_SIZE) & (offsets < numel)
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(values), axis=0)
    # Keep division IEEE round-to-nearest. Triton's default floating-point
    # division may use an approximate reciprocal, which can move exact .5
    # quantization ties across the boundary for large BF16 expert matrices.
    scale = tl.maximum(tl.div_rn(amax, 7.0), 1e-5)
    if SCALE_BF16:
        scale = scale.to(tl.bfloat16).to(tl.float32)
    else:
        scale = scale.to(tl.float16).to(tl.float32)

    scaled = tl.div_rn(values, scale)
    # Explicit round-to-nearest-even to match torch.round/rintf, including
    # negative ties, without relying on backend-specific libdevice symbols.
    rounded = _round_to_nearest_even(scaled)
    quantized = tl.minimum(tl.maximum(rounded, -7.0), 7.0)
    tl.store(output_ptr + offsets, quantized * scale, mask=mask)


@triton.jit
def _int4_quantize_pack_kernel(
    input_ptr,
    packed_ptr,
    scale_ptr,
    numel,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PACKED_BLOCK_SIZE: tl.constexpr,
    SCALE_BF16: tl.constexpr,
):
    # One program handles several independent quantization groups. The
    # exporter invokes this kernel for every individual Qwen expert projection;
    # using one tiny program per 128-value group created hundreds of millions
    # of programs for a full actor-to-rollout refresh. This two-dimensional
    # tile preserves the exact per-group quantization grid while amortizing
    # program scheduling across 1,024 input values.
    group_ids = tl.program_id(0) * GROUPS_PER_PROGRAM + tl.arange(0, GROUPS_PER_PROGRAM)
    group_mask = group_ids < tl.cdiv(numel, GROUP_SIZE)
    local_offsets = tl.arange(0, BLOCK_SIZE)
    input_offsets = group_ids[:, None] * GROUP_SIZE + local_offsets[None, :]
    input_mask = group_mask[:, None] & (local_offsets[None, :] < GROUP_SIZE) & (input_offsets < numel)
    values = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(values), axis=1)
    scale = tl.maximum(tl.div_rn(amax, 7.0), 1e-5)
    if SCALE_BF16:
        scale = scale.to(tl.bfloat16).to(tl.float32)
    else:
        scale = scale.to(tl.float16).to(tl.float32)
    tl.store(scale_ptr + group_ids, scale, mask=group_mask)

    packed_local = tl.arange(0, PACKED_BLOCK_SIZE)
    packed_offsets = group_ids[:, None] * (GROUP_SIZE // 8) + packed_local[None, :]
    packed_mask = group_mask[:, None] & (packed_local[None, :] < (GROUP_SIZE // 8))
    packed = tl.zeros((GROUPS_PER_PROGRAM, PACKED_BLOCK_SIZE), dtype=tl.int32)
    for nibble in tl.static_range(0, 8):
        value_offsets = group_ids[:, None] * GROUP_SIZE + packed_local[None, :] * 8 + nibble
        current = tl.load(input_ptr + value_offsets, mask=packed_mask, other=0.0).to(tl.float32)
        rounded = _round_to_nearest_even(tl.div_rn(current, scale[:, None]))
        quantized = tl.minimum(tl.maximum(rounded, -7.0), 7.0).to(tl.int32) + 8
        packed = packed | (quantized << (nibble * 4))

    tl.store(packed_ptr + packed_offsets, packed, mask=packed_mask)


def fake_quant_int4_cuda(weight: torch.Tensor, group_size: int, scale_dtype: str) -> torch.Tensor:
    """Run fused group-wise INT4 QDQ on a contiguous CUDA weight tensor."""
    if not weight.is_cuda:
        raise ValueError("fake_quant_int4_cuda requires a CUDA tensor")
    if not weight.is_contiguous():
        raise ValueError("fake_quant_int4_cuda requires a contiguous tensor")
    if weight.numel() % group_size != 0:
        raise ValueError("weight numel must be divisible by group_size")

    output = torch.empty_like(weight)
    block_size = triton.next_power_of_2(group_size)
    grid = (triton.cdiv(weight.numel(), group_size),)
    _int4_fake_quant_kernel[grid](
        weight,
        output,
        weight.numel(),
        GROUP_SIZE=group_size,
        BLOCK_SIZE=block_size,
        SCALE_BF16=scale_dtype.lower() in {"bfloat16", "bf16"},
        num_warps=4,
    )
    return output


def quantize_pack_int4_cuda(
    weight: torch.Tensor,
    group_size: int,
    scale_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused absmax, stored-scale rounding, INT4 quantization, and packing."""
    if not weight.is_cuda:
        raise ValueError("quantize_pack_int4_cuda requires a CUDA tensor")
    if not weight.is_contiguous():
        raise ValueError("quantize_pack_int4_cuda requires a contiguous tensor")
    if weight.shape[-1] % group_size != 0:
        raise ValueError("weight input dimension must be divisible by group_size")

    packed_shape = (*weight.shape[:-1], weight.shape[-1] // 8)
    scale_shape = (*weight.shape[:-1], weight.shape[-1] // group_size)
    packed = torch.empty(packed_shape, dtype=torch.int32, device=weight.device)
    stored_dtype = torch.bfloat16 if scale_dtype.lower() in {"bfloat16", "bf16"} else torch.float16
    scale = torch.empty(scale_shape, dtype=stored_dtype, device=weight.device)

    block_size = triton.next_power_of_2(group_size)
    packed_block_size = triton.next_power_of_2(group_size // 8)
    groups_per_program = max(1, 1024 // block_size)
    grid = (triton.cdiv(weight.numel(), group_size * groups_per_program),)
    _int4_quantize_pack_kernel[grid](
        weight,
        packed,
        scale,
        weight.numel(),
        GROUP_SIZE=group_size,
        GROUPS_PER_PROGRAM=groups_per_program,
        BLOCK_SIZE=block_size,
        PACKED_BLOCK_SIZE=packed_block_size,
        SCALE_BF16=scale_dtype.lower() in {"bfloat16", "bf16"},
        num_warps=4,
    )
    return packed, scale


__all__ = ["fake_quant_int4_cuda", "quantize_pack_int4_cuda"]
