# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-state regression for the production Megatron backward patch."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("already_checkpointing", [False, True])
@pytest.mark.parametrize("fail_recompute", [False, True])
def test_backward_sets_and_restores_checkpoint_state(monkeypatch, already_checkpointing, fail_recompute):
    rd = pytest.importorskip("megatron.core.tensor_parallel.random")
    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    # Only CUDA RNG/stream services are stubbed; call the actual patched
    # backward, actual autograd, and Core's real checkpoint-state functions.
    monkeypatch.setattr(rd, "_fork_rng", nullcontext)
    monkeypatch.setattr(rd, "_set_all_rng_states", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    monkeypatch.setattr(rd.CheckpointFunction, "backward", rd.CheckpointFunction.backward)
    monkeypatch.setattr(rd, "IS_CHECKPOINTING", already_checkpointing)
    apply_patch_megatron_recomputation_backward()
    observed = []

    def run(x):
        observed.append((rd.is_checkpointing(), torch.is_grad_enabled()))
        if fail_recompute:
            raise RuntimeError("injected recompute failure")
        return x.square()

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    ctx = SimpleNamespace(saved_tensors=(x,), distribute_saved_activations=False, rng_states=(), run_function=run)
    if fail_recompute:
        with pytest.raises(RuntimeError, match="injected recompute failure"):
            rd.CheckpointFunction.backward(ctx, torch.ones(3))
    else:
        result = rd.CheckpointFunction.backward(ctx, torch.ones(3))
        torch.testing.assert_close(result[2], torch.tensor([2.0, 4.0, 6.0]), rtol=0, atol=0)
    assert observed == [(True, True)]
    assert rd.is_checkpointing() is already_checkpointing
