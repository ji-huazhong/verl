# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from verl.utils.qat.int4 import apply_int4_qat_to_modules, fake_quant_int4_grouped_ste, fake_quant_int4_ste
from verl.utils.qat.int4_profile import int4_qat_profile

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_grouped_exact_bits_gradients_and_main_grad(device, group, dtype):
    # Different shapes and row widths ensure flattening cannot merge scale grids.
    weights = [torch.nn.Parameter(torch.randn(*shape, device=device, dtype=dtype)) for shape in [(3, 128), (2, 256)]]
    scale_dtype = "float16" if dtype == torch.float16 else "bfloat16"
    with torch.no_grad():
        weights[0][0].zero_()
        weights[1][0, :8] = torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 7], device=device, dtype=dtype)
        weights[1][0, 8:group].zero_()
    for weight in weights:
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    for _ in range(3):
        actual = fake_quant_int4_grouped_ste(weights, group, scale_dtype)
        for index, (weight, output) in enumerate(zip(weights, actual, strict=True)):
            assert torch.equal(output, fake_quant_int4_ste(weight, group, scale_dtype))
            assert output.main_grad is weight.main_grad
            (output.float().sum() * (index + 1)).backward()
    for index, weight in enumerate(weights):
        assert torch.equal(weight.grad, torch.full_like(weight, 3 * (index + 1)))


@pytest.mark.parametrize("device", DEVICES)
def test_grouped_no_cache_and_saved_outputs_survive_updates(device):
    weights = [torch.nn.Parameter(torch.randn(2, 128, device=device)) for _ in range(3)]
    old = fake_quant_int4_grouped_ste(weights)
    saved = [output.detach().clone() for output in old]
    inputs = [torch.randn_like(weight, requires_grad=True) for weight in weights]
    loss = sum((output * inp).sum() for output, inp in zip(old, inputs, strict=True))
    with torch.no_grad():
        weights[0].add_(1)
        weights[1].data = torch.full_like(weights[1], 0.25)
    new = fake_quant_int4_grouped_ste(weights)
    for weight, previous, snapshot, output in zip(weights, old, saved, new, strict=True):
        assert torch.equal(previous, snapshot)
        assert torch.equal(output, fake_quant_int4_ste(weight))
        assert previous.untyped_storage().data_ptr() != output.untyped_storage().data_ptr()
    loss.backward()
    for inp, snapshot in zip(inputs, saved, strict=True):
        assert torch.equal(inp.grad, snapshot)


def test_grouped_frozen_and_unused_experts():
    weights = [torch.randn(2, 128, requires_grad=index != 1) for index in range(3)]
    outputs = fake_quant_int4_grouped_ste(weights)
    assert not outputs[1].requires_grad
    outputs[0].sum().backward()
    assert torch.equal(weights[0].grad, torch.ones_like(weights[0]))
    assert weights[1].grad is None
    assert weights[2].grad is None
    assert fake_quant_int4_grouped_ste([]) == []
    assert torch.equal(fake_quant_int4_grouped_ste(weights[:1])[0], fake_quant_int4_ste(weights[0]))


@pytest.mark.parametrize("case", ["noncontiguous", "mixed_dtype", "subclass"])
def test_grouped_falls_back_without_changing_contract(case, monkeypatch):
    class TensorSubclass(torch.Tensor):
        pass

    weights = [torch.randn(128, 128, requires_grad=True) for _ in range(2)]
    if case == "noncontiguous":
        weights[1] = weights[1].t()
    elif case == "mixed_dtype":
        weights[1] = weights[1].to(torch.bfloat16)
    else:
        weights[1] = weights[1].as_subclass(TensorSubclass)
    expected = [fake_quant_int4_ste(weight) for weight in weights]

    def unexpected_cat(*args, **kwargs):
        raise AssertionError("unsupported layouts must use per-expert QDQ")

    monkeypatch.setattr(torch, "cat", unexpected_cat)
    actual = fake_quant_int4_grouped_ste(weights)
    assert all(torch.equal(got, want) for got, want in zip(actual, expected, strict=True))


def test_grouped_validates_each_row_before_flattening():
    # Total size is divisible by 128, but neither row is; accepting this would
    # silently change the quantization grid across row/expert boundaries.
    with pytest.raises(ValueError, match="input dimension"):
        fake_quant_int4_grouped_ste([torch.randn(2, 64), torch.randn(2, 64)], 128)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_hook_switch_and_profile_count_real_qdq_calls(enabled, monkeypatch, capsys):
    class Grouped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(2, 128)) for _ in range(3)])

        def _get_weight_tensors(self):
            return list(self.weights)

    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.Module()
    model.mlp.experts.linear_fc1 = module = Grouped()
    monkeypatch.setenv("VERL_INT4_QAT_GROUPED_QDQ", enabled)
    monkeypatch.setenv("VERL_INT4_QAT_TRAIN_PROFILE", "1")
    apply_int4_qat_to_modules([model], SimpleNamespace(group_size=32))
    with int4_qat_profile("train", 1, enabled=True) as profile:
        outputs = profile.wrap_forward_step(module._get_weight_tensors)()
        sum(output.sum() for output in outputs).backward()
    report = json.loads(capsys.readouterr().err.split("INT4_QAT_TRAIN_PROFILE ", 1)[1])
    assert sum(group["calls"] for group in report["groups"]) == (1 if enabled == "1" else 3)
    assert sum(group["output_bytes"] for group in report["groups"]) == 3 * 2 * 128 * 4
    assert not hasattr(module, "_verl_int4_schedule_cache")
    for weight in module.weights:
        assert torch.equal(weight.grad, torch.ones_like(weight))


def test_hook_rejects_invalid_switch(monkeypatch):
    monkeypatch.setenv("VERL_INT4_QAT_GROUPED_QDQ", "invalid")
    with pytest.raises(ValueError, match="VERL_INT4_QAT_GROUPED_QDQ"):
        apply_int4_qat_to_modules([], SimpleNamespace())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA and Transformer Engine required")
@pytest.mark.parametrize("recompute", [None, True, False])
@pytest.mark.parametrize("fused_grad", [False, True])
@pytest.mark.parametrize("group", [32, 128])
def test_real_te_grouped_matches_per_expert_with_recompute(monkeypatch, recompute, fused_grad, group):
    from transformer_engine.pytorch import GroupedLinear

    def make_model(enabled):
        monkeypatch.setenv("VERL_INT4_QAT_GROUPED_QDQ", enabled)
        model = torch.nn.Module()
        model.mlp = torch.nn.Module()
        model.mlp.experts = torch.nn.Module()
        for name in ("linear_fc1", "linear_fc2"):
            setattr(
                model.mlp.experts,
                name,
                GroupedLinear(
                    3,
                    128,
                    128,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    device="cuda",
                    fuse_wgrad_accumulation=fused_grad,
                ),
            )
        for parameter in model.parameters():
            parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
        apply_int4_qat_to_modules([model], SimpleNamespace(group_size=group))
        return model

    baseline, grouped = make_model("0"), make_model("1")
    grouped.load_state_dict(baseline.state_dict())
    data = torch.randn(6, 128, dtype=torch.bfloat16, device="cuda")

    def run(model):
        inp = data.detach().clone().requires_grad_()

        def forward(x):
            experts = model.mlp.experts
            # Include an expert with zero tokens, as in sparse routing.
            x = experts.linear_fc1(x, m_splits=[2, 0, 4])
            return experts.linear_fc2(torch.nn.functional.silu(x), m_splits=[2, 0, 4])

        for _ in range(2):
            output = forward(inp) if recompute is None else checkpoint(forward, inp, use_reentrant=recompute)
            output.float().square().sum().backward()
        grads = [p.main_grad if fused_grad else p.grad for p in model.parameters()]
        return output, inp.grad, grads

    want, want_dx, want_grads = run(baseline)
    got, got_dx, got_grads = run(grouped)
    assert torch.equal(got, want)
    assert torch.equal(got_dx, want_dx)
    for actual, expected in zip(got_grads, want_grads, strict=True):
        assert actual is not None
        assert torch.equal(actual, expected)
    assert any(grad.abs().sum() > 0 for grad in got_grads)
