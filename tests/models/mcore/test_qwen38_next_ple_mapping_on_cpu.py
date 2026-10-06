# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""PLE conversion checks without importing the CUDA-only model provider."""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Generic, TypeVar

import pytest
import torch

T = TypeVar("T")
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def mapping_module(monkeypatch):
    class MappingBase(Generic[T]):
        def __init__(self, megatron_param, hf_param):
            self.megatron_param = megatron_param
            self.hf_param = hf_param
            self.tp_rank, self.tp_size, self.pp_size = 0, 1, 1
            self.tp_group = None
            self.broadcast_shapes = []

        def broadcast_from_pp_rank(self, tensor, cache_key=None):
            self.broadcast_shapes.append(tuple(tensor.shape))
            return tensor

    stub = types.ModuleType("megatron.bridge.models.conversion.param_mapping")
    stub.MegatronParamMapping = MappingBase
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    spec = importlib.util.spec_from_file_location(
        "qwen38_ple_mapping_test", ROOT / "verl/models/mcore/qwen3_8_next/param_mapping.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 8])
def test_rank_local_import_only_reads_overlapping_shards(mapping_module, tp_size):
    full = torch.arange(24 * 5).reshape(24, 5)
    shards = full.split(6)
    ranks = []
    for rank in range(tp_size):
        loaded = []

        def load(index, loaded=loaded):
            loaded.append(index)
            return shards[index]

        result = mapping_module.ple_local_rows_from_shards(load, 4, 24, 6, rank, tp_size)
        lo, hi = rank * 24 // tp_size, (rank + 1) * 24 // tp_size
        assert loaded == list(range(lo // 6, (hi - 1) // 6 + 1))
        torch.testing.assert_close(result, full[lo:hi])
        ranks.append(result)
    torch.testing.assert_close(torch.cat(ranks), full)


def test_partial_final_hf_shard(mapping_module):
    full = torch.arange(10 * 3).reshape(10, 3)
    shards = full.split(4)
    rows = mapping_module.ple_local_rows_from_shards(lambda i: shards[i], 3, 10, 4, 1, 2)
    torch.testing.assert_close(rows, full[5:])


@pytest.mark.parametrize(
    "rank,size,total,height,count",
    [(0, 0, 12, 4, 3), (2, 2, 12, 4, 3), (0, 5, 12, 4, 3), (0, 1, 13, 4, 3)],
)
def test_invalid_layout_fails_before_reading(mapping_module, rank, size, total, height, count):
    def load(_):
        raise AssertionError("Invalid layout must fail before loading tensor payloads")

    with pytest.raises(ValueError):
        mapping_module.ple_local_rows_from_shards(load, count, total, height, rank, size)


def test_truncated_shard_is_rejected(mapping_module):
    with pytest.raises(ValueError, match="expected 4 rows"):
        mapping_module.ple_local_rows_from_shards(lambda _: torch.zeros(3, 2), 2, 8, 4, 0, 2)


def test_export_is_lazy_casts_per_shard_and_detaches(mapping_module):
    mapping = mapping_module.PLENGramEmbeddingMapping("ple.weight", "ple.shard_{}.weight", 4)
    table = torch.nn.Parameter(torch.arange(48, dtype=torch.float32).reshape(12, 4))
    exported = mapping.megatron_to_hf(table, None).with_dtype(torch.bfloat16)
    assert len(exported) == 4
    assert len(list(exported)) == 4
    assert mapping.broadcast_shapes == []
    for i, (_, value) in enumerate(exported.items()):
        assert mapping.broadcast_shapes == [(3, 4)] * (i + 1)
        assert value.dtype == torch.bfloat16
        assert not value.requires_grad
        torch.testing.assert_close(value.float(), table[i * 3 : (i + 1) * 3].detach())
    assert torch.is_grad_enabled()


def test_nonowning_pipeline_stage_receives_only_one_hf_shard(mapping_module):
    mapping = mapping_module.PLENGramEmbeddingMapping("ple.weight", "ple.shard_{}.weight", 4)
    mapping.pp_size = 2
    calls = []

    def receive(tensor, cache_key):
        assert tensor is None
        calls.append(cache_key)
        return torch.ones(3, 4)

    mapping.broadcast_from_pp_rank = receive
    exported = mapping.megatron_to_hf(None, None)
    assert calls == []
    assert exported["ple.shard_2.weight"].shape == (3, 4)
    assert calls == ["ple.weight:shard_2"]


def test_import_checks_target_shape(mapping_module):
    mapping = mapping_module.PLENGramEmbeddingMapping("ple.weight", "ple.shard_{}.weight", 2)
    target = torch.nn.Embedding(4, 3)
    with pytest.raises(ValueError, match="does not match"):
        mapping.hf_to_megatron({"local_rows": torch.zeros(4, 2)}, target)


def test_export_missing_single_pipeline_owner_is_an_error(mapping_module):
    mapping = mapping_module.PLENGramEmbeddingMapping("ple.weight", "ple.shard_{}.weight", 2)
    with pytest.raises(ValueError, match="Missing PLE parameter"):
        mapping.megatron_to_hf(None, None)
