# SPDX-License-Identifier: Apache-2.0
"""Checkpoint factory merges must not participate in the training graph.

The CPU storage stand-in exercises real Core factories and PyTorch view rules;
the separate multi-rank trainer resume gate covers actual distributed I/O.
"""

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.dist_checkpointing.mapping import (  # noqa: E402
    ShardedTensor,
    ShardedTensorFactory,
    apply_factory_merges,
)

from verl.utils.megatron import dist_checkpointing as checkpoint_utils  # noqa: E402


def _factory(parameter):
    @torch.no_grad()
    def build(key, tensor, replica_id, flattened_range):
        assert flattened_range is None
        return [
            ShardedTensor.from_rank_offsets(key, part, (0, index, 2), replica_id=replica_id)
            for index, part in enumerate(tensor.chunk(2, dim=0))
        ]

    def merge(parts):
        return torch.cat(parts, dim=0)

    return ShardedTensorFactory("adapter", parameter, build, merge, replica_id=0)


def _restore_into_live_views(state, _path, **_kwargs):
    factory = state["model"]["adapter"]
    shards = factory.build()
    for index, shard in enumerate(shards):
        # Loading into detached storage still increments the live parameter's
        # shared version counter, as in a replicated distributed load.
        shard.data.detach().copy_(torch.full_like(shard.data, index + 2))
    return apply_factory_merges({"model": {"adapter": [s.data for s in shards]}}, state)


def _stub_strategies(monkeypatch):
    monkeypatch.setattr(checkpoint_utils, "TorchDistLoadShardedStrategy", lambda: object())
    monkeypatch.setattr(checkpoint_utils, "FullyParallelLoadStrategyWrapper", lambda strategy, group: strategy)
    monkeypatch.setattr(checkpoint_utils.mpu, "get_data_parallel_group", lambda **kwargs: None)


def test_grad_enabled_factory_merge_reproduces_view_error():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    state = {"model": {"adapter": _factory(parameter)}}
    with torch.enable_grad(), pytest.raises(RuntimeError, match="view was created in no_grad mode"):
        _restore_into_live_views(state, "unused")


@pytest.mark.parametrize("caller_grad_enabled", [True, False])
def test_load_factory_views_without_autograd_and_restore_caller(monkeypatch, caller_grad_enabled):
    _stub_strategies(monkeypatch)
    monkeypatch.setattr(checkpoint_utils.dist_checkpointing, "load", _restore_into_live_views)
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    state = {"model": {"adapter": _factory(parameter)}}
    with torch.set_grad_enabled(caller_grad_enabled):
        loaded = checkpoint_utils.load_dist_checkpointing(state, "unused")
        assert torch.is_grad_enabled() == caller_grad_enabled
    restored = loaded["model"]["adapter"]
    expected = torch.tensor([2, 2, 3, 3], dtype=torch.float32)[:, None].expand(4, 3)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert restored.grad_fn is None and not restored.requires_grad
    assert parameter.requires_grad
    with torch.enable_grad():
        parameter.square().sum().backward()
    torch.testing.assert_close(parameter.grad, 2 * expected, rtol=0, atol=0)


@pytest.mark.parametrize("caller_grad_enabled", [True, False])
def test_load_error_restores_caller_grad_mode(monkeypatch, caller_grad_enabled):
    _stub_strategies(monkeypatch)

    def fail_load(*args, **kwargs):
        assert not torch.is_grad_enabled()
        raise OSError("synthetic checkpoint read failure")

    monkeypatch.setattr(checkpoint_utils.dist_checkpointing, "load", fail_load)
    with torch.set_grad_enabled(caller_grad_enabled):
        with pytest.raises(OSError, match="synthetic checkpoint read failure"):
            checkpoint_utils.load_dist_checkpointing({}, "unused")
        assert torch.is_grad_enabled() == caller_grad_enabled
