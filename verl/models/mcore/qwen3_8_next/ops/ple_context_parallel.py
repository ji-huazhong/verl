# SPDX-License-Identifier: Apache-2.0
"""Packed CP causal-convolution halos, without gathering full hidden states."""

import torch
import torch.distributed as dist


class PackedCPHalo:
    """Expand each owned chunk by its document-clipped left convolution halo.

    Plans depend only on physical packed boundaries, not model type or values.
    Each peer sends a requested row at most once; local rows never traverse the
    network. Backward reverses the exchange and sums repeated consumers in FP32.
    The normalized PLE activations themselves use FP32 in both CP1 and CP>1.
    """

    def __init__(self, layout, depth):
        if not isinstance(depth, int) or depth < 0:
            raise ValueError("Halo depth must be a nonnegative integer")
        self.layout = layout
        self.depth = depth
        bounds = layout.cu_seqlens.tolist()
        local_count = layout.local_indices.numel()
        self.local_count = local_count
        natural = layout.natural_order.tolist()
        rank_order = layout.rank_order.tolist()
        owners = [index // local_count for index in natural] if local_count else []
        local_rows = [index % local_count for index in natural] if local_count else []

        expanded, cores, cuts = [], [], []
        for member in range(layout.size):
            rows, core, cu = [], [], [0]
            for lo, hi in zip(bounds, bounds[1:], strict=False):
                width = (hi - lo) // (2 * layout.size) if layout.size > 1 else hi - lo
                chunks = (member, 2 * layout.size - 1 - member) if layout.size > 1 else (0,)
                for chunk in chunks:
                    start, end = lo + chunk * width, lo + (chunk + 1) * width
                    if start == end:
                        continue
                    halo_start = max(lo, start - depth)
                    core.extend(range(len(rows) + start - halo_start, len(rows) + end - halo_start))
                    rows.extend(range(halo_start, end))
                    cu.append(len(rows))
            expanded.append(rows)
            cores.append(core)
            cuts.append(cu)

        # requests[consumer][owner] are unique global rows, in stable order.
        requests = [
            [
                sorted({row for row in rows if owners[row] == owner and owner != consumer})
                for owner in range(layout.size)
            ]
            for consumer, rows in enumerate(expanded)
        ]
        rank = layout.rank
        self.send_splits = [len(requests[peer][rank]) for peer in range(layout.size)]
        self.recv_splits = [len(requests[rank][peer]) for peer in range(layout.size)]
        send_rows = [local_rows[row] for peer in range(layout.size) for row in requests[peer][rank]]
        received = [row for peer in range(layout.size) for row in requests[rank][peer]]
        available = rank_order[rank * local_count : (rank + 1) * local_count] + received
        available_index = {row: index for index, row in enumerate(available)}
        device = layout.cu_seqlens.device

        def indices(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        self.send_indices = indices(send_rows)
        self.expand_indices = indices([available_index[row] for row in expanded[rank]])
        self.core_indices = indices(cores[rank])
        self.global_indices = indices(expanded[rank])
        self.cu_seqlens = torch.tensor(cuts[rank], dtype=torch.int32, device=device)
        self.expanded_count = len(expanded[rank])

    def _validate(self, tensor, group, length):
        if dist.get_world_size(group) != self.layout.size or dist.get_rank(group) != self.layout.rank:
            raise ValueError("CP process group does not match the halo plan")
        if tensor.shape[0] != length or tensor.device != self.expand_indices.device:
            raise ValueError("Tensor rows/device do not match the halo plan")
        if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("Halo exchange requires FP32/BF16/FP16 activations")

    def expand(self, local, group):
        """Forward exchange. Call ``apply`` when an autograd edge is needed."""
        self._validate(local, group, self.local_count)
        recv = local.new_empty((sum(self.recv_splits), *local.shape[1:]))
        if self.layout.size > 1:
            dist.all_to_all_single(
                recv,
                local.index_select(0, self.send_indices).contiguous(),
                output_split_sizes=self.recv_splits,
                input_split_sizes=self.send_splits,
                group=group,
            )
        return torch.cat((local, recv)).index_select(0, self.expand_indices)

    def reduce(self, expanded_grad, group):
        """Reverse exchange plus FP32 accumulation into each unique owner."""
        self._validate(expanded_grad, group, self.expanded_count)
        merged = torch.zeros(
            (self.local_count + sum(self.recv_splits), *expanded_grad.shape[1:]),
            dtype=torch.float32,
            device=expanded_grad.device,
        )
        merged.index_add_(0, self.expand_indices, expanded_grad.float())
        returned = merged.new_empty((sum(self.send_splits), *expanded_grad.shape[1:]))
        if self.layout.size > 1:
            dist.all_to_all_single(
                returned,
                merged[self.local_count :].contiguous(),
                output_split_sizes=self.send_splits,
                input_split_sizes=self.recv_splits,
                group=group,
            )
        local = merged[: self.local_count].clone()
        local.index_add_(0, self.send_indices, returned)
        return local

    def apply(self, local, group):
        return _HaloExchange.apply(local, self, group)


class _HaloExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, plan, group):
        ctx.plan, ctx.group, ctx.dtype = plan, group, local.dtype
        return plan.expand(local, group)

    @staticmethod
    def backward(ctx, grad):
        return ctx.plan.reduce(grad, ctx.group).to(ctx.dtype), None, None
