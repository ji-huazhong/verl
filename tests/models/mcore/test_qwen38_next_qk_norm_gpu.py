# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Q/K norm forward, gradient, recomputation, and packing checks."""

import inspect
import os
import types
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

pytestmark = pytest.mark.skipif(os.environ.get("RUN_QWEN38_GPU_TESTS") != "1", reason="explicit GPU opt-in required")


@pytest.fixture(scope="module", autouse=True)
def gpu_context():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 4 * 1024**3:
        pytest.skip("Need 4 GiB free headroom; never evict another job")
    torch.cuda.set_per_process_memory_fraction(min(0.10, 4 * 1024**3 / total))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_scale", [1.0, 0.001])
def test_qk_norm_fp64_gradient_and_recomputation(dtype, input_scale):
    from verl.models.mcore.qwen3_8_next.ops.gated_delta_net import fp32_qk_l2norm

    torch.manual_seed(10048)
    values = torch.randn(1, 32, 4, 128, device="cuda", dtype=dtype) * input_scale
    values[:, 0] = 0  # Exercise epsilon and finite gradients at the origin.
    x = values.detach().requires_grad_()
    reference = values.double().requires_grad_()
    actual = fp32_qk_l2norm(x)
    expected = reference / (reference.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    upstream = torch.randn_like(actual)
    (gradient,) = torch.autograd.grad(actual, x, upstream)
    (oracle_gradient,) = torch.autograd.grad(expected, reference, upstream.double())
    for result, oracle in ((actual, expected), (gradient, oracle_gradient)):
        assert result.dtype == dtype and result.isfinite().all()
        rounded = oracle.to(dtype).double()
        relative_error = (result.double() - rounded).norm() / rounded.norm()
        assert relative_error < 1e-3
    assert gradient.abs().sum() > 0
    recomputed_x = values.detach().requires_grad_()
    recomputed = checkpoint(fp32_qk_l2norm, recomputed_x, use_reentrant=False)
    (recomputed_gradient,) = torch.autograd.grad(recomputed, recomputed_x, upstream)
    assert torch.equal(recomputed, actual)
    assert torch.equal(recomputed_gradient, gradient)


@pytest.mark.parametrize("tp,cp", [(8, 1), (2, 2), (1, 4)])
@pytest.mark.parametrize("normalize", [False, True])
def test_qk_packing_preserves_core_values_and_gradients(tp, cp, normalize):
    """Compare with pinned Core packing, using the same norm on both paths."""
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    from verl.models.mcore.qwen3_8_next.ops.gated_delta_net import Qwen38NextGatedDeltaNet, fp32_qk_l2norm

    original = inspect.unwrap(GatedDeltaNet._prepare_input_for_gated_delta_rule)
    assert isinstance(original, types.FunctionType)
    namespace = dict(original.__globals__, l2norm=fp32_qk_l2norm)
    reference = types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    reference.__kwdefaults__ = original.__kwdefaults__
    block = SimpleNamespace(
        qk_dim_local_tp=16 * 128 // tp,
        v_dim_local_tp=48 * 128 // tp,
        cp_size=cp,
        key_head_dim=128,
        value_head_dim=128,
        num_value_heads=48,
        num_key_heads=16,
        use_qk_l2norm=normalize,
    )
    torch.manual_seed(10048)
    shapes = (
        (2, 5, (2 * block.qk_dim_local_tp + block.v_dim_local_tp) // cp),
        (2, 5, 48 // tp // cp, 128),
        (2, 5, 48 // tp // cp),
        (2, 5, 48 // tp // cp),
    )
    values = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for shape in shapes]
    outputs, gradients = [], []
    for fn in (reference, Qwen38NextGatedDeltaNet._prepare_input_for_gated_delta_rule):
        inputs = [v.detach().clone().requires_grad_() for v in values]
        output = fn(block, inputs[0], inputs[1], 2, 5, *inputs[2:])
        outputs.append(output)
        torch.manual_seed(99)
        gradients.append(torch.autograd.grad(output, inputs, tuple(torch.randn_like(v) for v in output)))
    assert all(torch.equal(a, b) for a, b in zip(*outputs, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(*gradients, strict=True))
