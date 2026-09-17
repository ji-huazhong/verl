# SPDX-License-Identifier: Apache-2.0
"""Packed zigzag CP indexing and differentiable projected-KV gathering.

Only projected keys/values (or small index metadata) are gathered. Decoder
hidden states, queries and attention outputs stay sequence-sharded. This is a
building block, not permission to enable CP before PLE boundaries are handled.
"""

import torch
import torch.distributed as dist


class PackedContextParallelLayout:
    """Core/Bridge THD order: each rank owns two chunks of every document.

    ``cu_seqlens`` describes the physical, padded global token buffer. Padding
    must already follow Bridge's per-document 2*CP alignment. Logical document
    lengths must not be substituted for those physical offsets.
    """

    def __init__(self, cu_seqlens, size, rank):
        if size < 1 or not 0 <= rank < size:
            raise ValueError("Invalid context-parallel size or rank")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("CP requires packed sequence boundaries")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("CP boundaries must be integer offsets")
        bounds = cu_seqlens.tolist()
        lengths = [hi - lo for lo, hi in zip(bounds, bounds[1:], strict=False)]
        if bounds[0] != 0 or any(length < 0 for length in lengths):
            raise ValueError("CP boundaries must start at zero and be nondecreasing")
        if size > 1 and any(length % (2 * size) for length in lengths):
            raise ValueError("Each packed physical sequence must be aligned to 2*CP")
        self.size, self.rank = size, rank
        self.total = bounds[-1]
        self.cu_seqlens = cu_seqlens
        indices = []
        for member in range(size):
            chunks = []
            for lo, hi in zip(bounds, bounds[1:], strict=False):
                if size == 1:
                    chunks.append(torch.arange(lo, hi, device=cu_seqlens.device))
                else:
                    width = (hi - lo) // (2 * size)
                    for chunk in (member, 2 * size - 1 - member):
                        chunks.append(
                            torch.arange(lo + chunk * width, lo + (chunk + 1) * width, device=cu_seqlens.device)
                        )
            indices.append(torch.cat(chunks))
        self.local_indices = indices[rank]
        self.rank_order = torch.cat(indices)
        self.natural_order = torch.argsort(self.rank_order)

    def local(self, global_tensor):
        if global_tensor.shape[0] != self.total:
            raise ValueError("Global token buffer does not match packed CP boundaries")
        return global_tensor.index_select(0, self.local_indices)

    def gather(self, local_tensor, group, *, fp32_output=False):
        if dist.get_world_size(group) != self.size or dist.get_rank(group) != self.rank:
            raise ValueError("CP process group does not match the packed layout")
        if local_tensor.shape[0] != self.local_indices.numel():
            raise ValueError("Local token buffer does not match packed CP boundaries")
        if self.size == 1:
            return local_tensor.float() if fp32_output else local_tensor
        return _GatherPackedCP.apply(local_tensor, self.rank_order, self.natural_order, group, fp32_output)


class _GatherPackedCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, rank_order, natural_order, group, fp32_output):
        ctx.group = group
        ctx.save_for_backward(rank_order)
        ctx.local_shape = local.shape
        ctx.input_dtype = local.dtype
        gathered = local.new_empty((local.shape[0] * dist.get_world_size(group), *local.shape[1:]))
        dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
        result = gathered.index_select(0, natural_order)
        # KV values are still communicated in their original dtype. A FP32
        # output edge lets the attention kernel return unrounded partial dKV,
        # reducing them before casting to the owner's activation dtype.
        return result.float() if fp32_output else result

    @staticmethod
    def backward(ctx, global_grad):
        (rank_order,) = ctx.saved_tensors
        # Every rank's LOCAL queries contribute to global KV. Sum those
        # contributions before returning each owner's zigzag shard; averaging
        # here would silently shrink gradients by CP. DDP handles loss scaling.
        rank_order_grad = global_grad.index_select(0, rank_order).contiguous()
        local_grad = global_grad.new_empty(ctx.local_shape)
        dist.reduce_scatter_tensor(local_grad, rank_order_grad, group=ctx.group)
        return local_grad.to(ctx.input_dtype), None, None, None, None
