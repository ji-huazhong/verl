# Copyright 2026 Individual Contributor
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PLE shard conversion adapted from NVIDIA Megatron-Bridge PR #6123.

The table is row-parallel during training and streamed as HF shards for rollout.
"""

from collections.abc import Callable, Mapping
from typing import Optional

import torch
from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
from torch import nn


def ple_local_rows_from_shards(
    load_shard: Callable[[int], torch.Tensor],
    num_shards: int,
    num_rows_total: int,
    shard_rows: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Assemble one tensor-parallel rank's rows of a row-sharded HF table.

    Rank ``r`` owns rows ``[r * rows_per_rank, (r + 1) * rows_per_rank)`` (the
    ``VocabParallelEmbedding`` layout); only the HF shards overlapping that range are loaded.

    Args:
        load_shard: ``callable(shard_index) -> Tensor`` loading one HF shard.
        num_shards: Number of HF shards.
        num_rows_total: Total (padded) row count of the table.
        shard_rows: Rows per HF shard (all but possibly the last shard).
        tp_rank / tp_size: Tensor-parallel coordinates.
    """
    if tp_size < 1 or not 0 <= tp_rank < tp_size or num_shards < 1 or shard_rows < 1 or num_rows_total < 1:
        raise ValueError("Invalid PLE shard layout or tensor-parallel coordinates")
    if not (num_shards - 1) * shard_rows < num_rows_total <= num_shards * shard_rows:
        raise ValueError("PLE shard count/height does not cover the table")
    if num_rows_total % tp_size != 0:
        raise ValueError(f"PLE table rows ({num_rows_total}) are not divisible by tp_size={tp_size}")
    rows_per_rank = num_rows_total // tp_size
    start, end = tp_rank * rows_per_rank, (tp_rank + 1) * rows_per_rank
    pieces = []
    for shard_index in range(start // shard_rows, min(num_shards, (end - 1) // shard_rows + 1)):
        shard = load_shard(shard_index)
        shard_start = shard_index * shard_rows
        expected_rows = min(shard_rows, num_rows_total - shard_start)
        if shard.ndim != 2 or shard.shape[0] != expected_rows:
            raise ValueError(f"PLE shard {shard_index} has shape {tuple(shard.shape)}, expected {expected_rows} rows")
        lo = max(start, shard_start) - shard_start
        hi = min(end, shard_start + shard.shape[0]) - shard_start
        pieces.append(shard[lo:hi])
    rows = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
    if rows.shape[0] != rows_per_rank:
        raise ValueError(f"PLE shards cover {rows.shape[0]} rows for TP rank {tp_rank}, expected {rows_per_rank}")
    return rows


class PLENGramEmbeddingMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
    """Vocab-parallel PLE n-gram table <-> ``split_ngram_parts`` HF shards.

    The HF checkpoint stores the table row-split into ``ngram_embedding.shard_<i>.weight`` tensors
    (concatenated along dim 0 they form the padded table). Megatron keeps the table as a
    :class:`~megatron.core.tensor_parallel.layers.VocabParallelEmbedding`, i.e. row-sharded across
    the tensor-parallel group.

    Import is rank-local: :meth:`Qwen38NextBridge.maybe_modify_loaded_hf_weight` loads only the shards
    that overlap this rank's row range (:func:`ple_local_rows_from_shards`), so the 51B-parameter
    table is never materialized in full on any rank. Export streams the table back out one HF shard
    at a time (:class:`_PLEShardExport`): each shard is assembled from the TP ranks' rows when the
    export loop reaches it, so peak memory is one shard (~0.8 GB for Qwen3.8-Flash-Next), not the
    ~100 GB table.
    """

    def __init__(self, megatron_param: str, hf_param: str | dict[str, str], num_shards: Optional[int] = None):
        """
        Args:
            megatron_param: Megatron parameter name pattern.
            hf_param: Either the HF shard-name pattern with a ``{}`` placeholder for the shard index
                (then ``num_shards`` is required), or the already expanded ``{"shard_<i>": name}``
                dict (the form :meth:`resolve` re-instantiates the mapping with).
            num_shards: Number of HF shards when ``hf_param`` is a pattern.
        """
        if isinstance(hf_param, str):
            if num_shards is None:
                raise ValueError("num_shards is required when hf_param is a shard-name pattern")
            hf_param = {f"shard_{i}": hf_param.format(i) for i in range(num_shards)}
        super().__init__(megatron_param, hf_param)
        self.num_shards = len(hf_param)

    def hf_to_megatron(self, hf_weights: dict[str, torch.Tensor], megatron_module: nn.Module) -> torch.Tensor:
        """Return this rank's rows.

        ``hf_weights`` is either the rank-local ``{"local_rows": Tensor}`` produced by
        :meth:`Qwen38NextBridge.maybe_modify_loaded_hf_weight`, or (fallback) the full shard dict.
        """
        if "local_rows" in hf_weights:
            rows = hf_weights["local_rows"]
        else:
            shards = [hf_weights[f"shard_{i}"] for i in range(self.num_shards)]
            total = sum(s.shape[0] for s in shards)
            rows = ple_local_rows_from_shards(
                lambda i: shards[i], self.num_shards, total, shards[0].shape[0], self.tp_rank, self.tp_size
            )
        target = megatron_module.weight
        if rows.shape != target.shape:
            raise ValueError(f"PLE local weight shape {tuple(rows.shape)} does not match {tuple(target.shape)}")
        return rows.to(device=target.device, dtype=target.dtype)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Mapping[str, torch.Tensor]:
        """Re-split the TP row shards into the checkpoint's shard layout, one shard at a time.

        Returns a lazy mapping: shard ``i`` is assembled (a tensor-parallel collective) when the
        export loop iterates to it, so all TP ranks must consume the shards in the same order --
        which the export does, and which a ``dict(...)`` / ``.items()`` traversal preserves.
        """
        # Keep the TP-local table on its owning pipeline stage. Broadcasting it
        # here would allocate a whole local table on every non-owning PP stage.
        if megatron_weights is None and self.pp_size == 1:
            raise ValueError(f"Missing PLE parameter {self.megatron_param}")
        return _PLEShardExport(self, megatron_weights)

    @torch.no_grad()
    def gather_shard(self, local_rows: Optional[torch.Tensor], shard_index: int) -> torch.Tensor:
        shard = self._gather_tp_shard(local_rows.detach(), shard_index) if local_rows is not None else None
        return self.broadcast_from_pp_rank(shard, cache_key=f"{self.megatron_param}:shard_{shard_index}")

    def _gather_tp_shard(self, local_rows: torch.Tensor, shard_index: int) -> torch.Tensor:
        """Assemble HF shard ``shard_index`` from the tensor-parallel row shards.

        Rank ``r`` owns rows ``[r * rows_per_rank, (r + 1) * rows_per_rank)``; each rank writes
        its overlap with the shard's row range into a zero buffer and the buffers are summed over
        the TP group (exact: every row has exactly one owner).
        """
        rows_per_rank = local_rows.shape[0]
        total_rows = rows_per_rank * self.tp_size
        shard_rows = -(-total_rows // self.num_shards)
        start = shard_index * shard_rows
        end = min(total_rows, start + shard_rows)
        if self.tp_size == 1:
            return local_rows[start:end]
        rank_start = self.tp_rank * rows_per_rank
        lo = max(start, rank_start)
        hi = min(end, rank_start + rows_per_rank)
        shard = torch.zeros(end - start, local_rows.shape[1], dtype=local_rows.dtype, device=local_rows.device)
        if hi > lo:
            shard[lo - start : hi - start] = local_rows[lo - rank_start : hi - rank_start]
        torch.distributed.all_reduce(shard, group=self.tp_group)
        return shard


class _PLEShardExport(Mapping[str, torch.Tensor]):
    """Lazy ``{hf_shard_name: tensor}`` view over a TP-sharded PLE table (see ``gather_shard``)."""

    def __init__(self, mapping: PLENGramEmbeddingMapping, local_rows: Optional[torch.Tensor], dtype=None):
        self._mapping = mapping
        self._local_rows = local_rows
        self._dtype = dtype
        self._names = [mapping.hf_param[f"shard_{i}"] for i in range(mapping.num_shards)]
        self._index = {name: i for i, name in enumerate(self._names)}

    def __getitem__(self, name: str) -> torch.Tensor:
        shard = self._mapping.gather_shard(self._local_rows, self._index[name])
        return shard.to(self._dtype) if self._dtype is not None and shard.is_floating_point() else shard

    def with_dtype(self, dtype: Optional[torch.dtype]):
        """Apply Bridge's export dtype without materializing any shard yet."""
        return type(self)(self._mapping, self._local_rows, dtype)

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)
