# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from verl.workers.rollout.vllm_rollout.sleep_state import RunnerTensorSnapshot


def test_runner_constants_restored_in_place_without_backing_up_model():
    state = SimpleNamespace(offsets=torch.tensor([-2, -1]), model=torch.nn.Linear(256, 256), counter=5)
    snapshot = RunnerTensorSnapshot.capture(state, max_bytes=16)
    alias = state.offsets
    assert snapshot.nbytes == 16
    state.offsets.zero_()
    snapshot.restore(state)
    assert state.offsets is alias
    torch.testing.assert_close(state.offsets, torch.tensor([-2, -1]), rtol=0, atol=0)


def test_backup_bound_and_parameters_fail_closed():
    with pytest.raises(RuntimeError, match="exceeding"):
        RunnerTensorSnapshot.capture(SimpleNamespace(large=torch.ones(32)), max_bytes=16)
    with pytest.raises(RuntimeError, match="model parameters"):
        RunnerTensorSnapshot.capture(SimpleNamespace(weight=torch.nn.Parameter(torch.ones(1))))


@pytest.mark.parametrize("replace_state", [False, True])
def test_state_replacement_rejected_before_restore(replace_state):
    state = SimpleNamespace(a=torch.ones(2), b=torch.ones(2))
    snapshot = RunnerTensorSnapshot.capture(state)
    state.a.zero_()
    if replace_state:
        state = SimpleNamespace(a=state.a, b=state.b)
    else:
        state.b = torch.zeros(2)
    with pytest.raises(RuntimeError, match="changed|replaced"):
        snapshot.restore(state)
    assert not bool(state.a.any())


def test_legacy_runner_without_model_state():
    snapshot = RunnerTensorSnapshot.capture(None)
    snapshot.restore(None)
    assert snapshot.nbytes == 0
