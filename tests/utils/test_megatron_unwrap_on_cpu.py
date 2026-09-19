# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Core's FSDP factory must not enter runtime wrapper-type checks."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from verl.utils.megatron_utils import (
    ALL_MODULE_WRAPPER_CLASSNAMES,
    MCORE_FSDP_WRAPPER_TYPES,
    register_megatron_training_hooks,
    unwrap_model,
)


def test_plain_model_and_list_unwrap_without_factory_type_error():
    model = torch.nn.Linear(2, 3)
    assert all(isinstance(wrapper, type) for wrapper in ALL_MODULE_WRAPPER_CLASSNAMES)
    assert unwrap_model(model) is model
    assert unwrap_model([model, model]) == [model, model]


def test_current_fsdp_concrete_classes_are_included():
    from megatron.core.distributed.fsdp import mcore_fsdp_adapter

    for name in ("FullyShardedDataParallel", "FullyShardedDataParallelV1", "FullyShardedDataParallelV2"):
        wrapper = getattr(mcore_fsdp_adapter, name, None)
        if isinstance(wrapper, type):
            assert wrapper in MCORE_FSDP_WRAPPER_TYPES
        elif wrapper is not None:
            assert wrapper not in MCORE_FSDP_WRAPPER_TYPES


def test_training_hook_registration_does_not_union_a_factory():
    model = torch.nn.Linear(2, 3)
    model.config = SimpleNamespace(no_sync_func=None)
    model.ddp_config = SimpleNamespace(overlap_grad_reduce=False)
    optimizer = SimpleNamespace(config=SimpleNamespace(overlap_param_gather=False), scale_loss=lambda loss: loss)
    register_megatron_training_hooks([model], optimizer)
    assert model.config.grad_scale_func is optimizer.scale_loss
    assert callable(model.config.finalize_model_grads_func)
