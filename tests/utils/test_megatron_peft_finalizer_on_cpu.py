# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Bridge step-level gradient hook selection without CUDA or model-name checks."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from verl.utils.megatron.peft import bridge_peft_grad_finalizer


def install_peft_utils(monkeypatch, utils):
    for name in ("megatron", "megatron.bridge", "megatron.bridge.peft"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["megatron.bridge.peft"].utils = utils


def test_bridge_owns_finalize_and_forwards_schedule_arguments(monkeypatch):
    events = []
    model = [object(), object()]
    tokens, groups = object(), object()

    def enable(chunks):
        events.append(("enable", chunks))
        return 96

    def finalize(chunks, num_tokens, **kwargs):
        events.append(("finalize", chunks, num_tokens, kwargs))

    install_peft_utils(
        monkeypatch,
        SimpleNamespace(
            enable_expert_parallel_grad_sync_in_finalize=enable,
            finalize_model_grads_with_expert_adapter_sync=finalize,
        ),
    )
    selected = bridge_peft_grad_finalizer(model, lambda *args: pytest.fail("Wrong Core-only finalizer"))
    assert selected is finalize
    assert events == [("enable", model)]
    selected(model, tokens, pg_collection=groups, force_all_reduce=True)
    assert events[-1] == ("finalize", model, tokens, {"pg_collection": groups, "force_all_reduce": True})


def test_older_bridge_retains_original_finalizer(monkeypatch):
    install_peft_utils(monkeypatch, SimpleNamespace())
    original = object()
    assert bridge_peft_grad_finalizer([], original) is original


@pytest.mark.parametrize("present", ["enable", "finalize"])
def test_incomplete_api_rejected_before_removing_fallback(monkeypatch, present):
    def unexpected(*args):
        pytest.fail("Do not remove fallback hooks unless their replacement exists")

    utils = SimpleNamespace()
    if present == "enable":
        utils.enable_expert_parallel_grad_sync_in_finalize = unexpected
    else:
        utils.finalize_model_grads_with_expert_adapter_sync = unexpected
    install_peft_utils(monkeypatch, utils)
    with pytest.raises(RuntimeError, match="incomplete"):
        bridge_peft_grad_finalizer([], unexpected)
