# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import logging
from types import SimpleNamespace

import pytest
import torch

from verl.utils.qat.core import QATConfig, load_quantization_config
from verl.workers.config.engine import QATEngineConfig
from verl.utils.qat.int4 import (
    Int4WeightExporter,
    apply_int4_qat_to_modules,
    dequantize_int4_levels,
    fake_quant_int4_ste,
    pack_int4_levels,
    quantize_int4_levels,
    unpack_int4_levels,
)
from verl.utils.qat.int4_vllm import (
    configure_int4_layerwise_reload,
    configure_int4_vllm_backend,
    expand_qwen3_5_fused_int4_weights,
    is_int4_wna16_quant_config,
    own_pending_int4_reload_views,
)


def _write_int4_config(tmp_path, group_size=32):
    path = tmp_path / "int4.json"
    path.write_text(
        json.dumps(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "experts": {
                        "targets": ["re:.*experts.*"],
                        "weights": {
                            "type": "int",
                            "num_bits": 4,
                            "strategy": "group",
                            "symmetric": True,
                            "group_size": group_size,
                        },
                        "input_activations": None,
                    }
                },
            }
        )
    )
    return path


def test_int4_json_contract_matches_trainer_config(tmp_path):
    path = _write_int4_config(tmp_path)
    config = QATConfig(
        enable=True,
        format="INT4",
        group_size=32,
        scope="routed_experts",
        quantization_config_path=str(path),
    )

    loaded = load_quantization_config(config)

    assert config.format == "int4"
    assert loaded["config_groups"]["experts"]["weights"]["group_size"] == 32


def test_int4_json_contract_rejects_group_size_mismatch(tmp_path):
    path = _write_int4_config(tmp_path, group_size=128)
    config = QATConfig(
        enable=True,
        format="int4",
        group_size=32,
        scope="routed_experts",
        quantization_config_path=str(path),
    )

    with pytest.raises(ValueError, match="does not match trainer settings"):
        load_quantization_config(config)


def test_int4_ptq_control_keeps_real_export_contract_without_fake_quant(tmp_path):
    path = _write_int4_config(tmp_path)
    trainer_config = QATEngineConfig(
        enable=True,
        format="int4",
        mode="w4a16",
        group_size=32,
        scope="routed_experts",
        fake_quant=False,
    )
    rollout_config = QATConfig(
        enable=True,
        format="int4",
        mode="w4a16",
        group_size=32,
        scope="routed_experts",
        fake_quant=False,
        quantization_config_path=str(path),
    )

    assert trainer_config.fake_quant is False
    assert rollout_config.fake_quant is False
    assert load_quantization_config(rollout_config)["quant_method"] == "compressed-tensors"


def test_int4_pack_round_trip_uses_uint4b8_bias():
    levels = torch.tensor([[-7, -6, -1, 0, 1, 6, 7, 0]], dtype=torch.int8)

    packed = pack_int4_levels(levels)

    assert packed.dtype == torch.int32
    assert packed.shape == (1, 1)
    assert torch.equal(unpack_int4_levels(packed), levels)
    assert (int(packed.item()) & 0xF) == 1  # -7 + bias 8
    assert ((int(packed.item()) >> 12) & 0xF) == 8  # zero + bias 8


def test_int4_pack_matches_compressed_tensors_format():
    helpers = pytest.importorskip("compressed_tensors.compressors.pack_quantized.helpers")
    levels = torch.arange(-7, 8, dtype=torch.int8).repeat(2, 9)[:, :128]

    assert torch.equal(pack_int4_levels(levels), helpers.pack_to_int32(levels, num_bits=4))


def test_fake_quant_and_export_share_stored_bf16_scale():
    weight = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32).reshape(2, 128)
    levels, scale = quantize_int4_levels(weight, group_size=128, scale_dtype="bfloat16")
    expected = dequantize_int4_levels(levels, scale, 128, weight.dtype)

    actual = fake_quant_int4_ste(weight, group_size=128, scale_dtype="bfloat16")

    assert scale.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_fake_quant_uses_identity_ste():
    weight = torch.randn(2, 128, requires_grad=True)

    fake_quant_int4_ste(weight, group_size=128).sum().backward()

    assert torch.equal(weight.grad, torch.ones_like(weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("shape", [(768, 2048), (2048, 768), (3, 16, 128)])
@pytest.mark.parametrize("group_size", [32, 128])
def test_cuda_fake_quant_and_packer_match_cpu_reference(shape, group_size):
    """Cover Qwen3 individual and Qwen3.5-style fused expert tensors."""
    cpu_weight = torch.randn(*shape, dtype=torch.bfloat16)
    cuda_weight = cpu_weight.cuda()

    expected_levels, expected_scale = quantize_int4_levels(cpu_weight, group_size, "bfloat16")
    expected_fake = dequantize_int4_levels(expected_levels, expected_scale, group_size, torch.bfloat16)
    actual_fake = fake_quant_int4_ste(cuda_weight, group_size, "bfloat16").cpu()

    input_name = (
        "model.layers.0.mlp.experts.0.gate_proj.weight"
        if len(shape) == 2
        else "model.layers.0.mlp.experts.gate_up_proj"
    )
    base_name = input_name.removesuffix(".weight")
    exporter = Int4WeightExporter(group_size=group_size)
    exported = dict(exporter.process_weights_iterator([(input_name, cuda_weight)]))

    assert torch.equal(actual_fake, expected_fake)
    assert torch.equal(
        unpack_int4_levels(exported[f"{base_name}.weight_packed"]).cpu(),
        expected_levels,
    )
    assert torch.equal(
        exported[f"{base_name}.weight_scale"].cpu(),
        expected_scale,
    )
    assert f"{base_name}.weight_shape" not in exported


def test_exporter_keeps_individual_expert_weight_shape_resident():
    weight = torch.randn(16, 128, dtype=torch.bfloat16)
    base_name = "model.layers.0.mlp.experts.0.gate_proj"

    exported = dict(
        Int4WeightExporter(group_size=32).process_weights_iterator([(f"{base_name}.weight", weight)])
    )

    assert f"{base_name}.weight_packed" in exported
    assert f"{base_name}.weight_scale" in exported
    assert f"{base_name}.weight_shape" not in exported


def test_exporter_diagnostics_aggregate_without_per_tensor_host_reads(monkeypatch, caplog):
    monkeypatch.setenv("VERL_INT4_QAT_RELOAD_DIAGNOSTICS", "1")
    weights = [
        (f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight", torch.ones(2, 128))
        for expert_id in range(2)
    ]

    with caplog.at_level(logging.WARNING):
        list(Int4WeightExporter(group_size=32).process_weights_iterator(weights))

    assert "scale_tensors=2 invalid=0" in caplog.text


def test_exporter_quantizes_only_compact_fused_routed_experts():
    gate_up = torch.randn(3, 16, 128, dtype=torch.bfloat16)
    attention = torch.randn(8, 128, dtype=torch.bfloat16)
    exporter = Int4WeightExporter(group_size=32)

    exported = dict(
        exporter.process_weights_iterator(
            [
                ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
                ("model.layers.0.self_attn.q_proj.weight", attention),
            ]
        )
    )

    assert exported["model.layers.0.mlp.experts.gate_up_proj.weight_packed"].shape == (3, 16, 16)
    assert exported["model.layers.0.mlp.experts.gate_up_proj.weight_scale"].shape == (3, 16, 4)
    assert "model.layers.0.mlp.experts.gate_up_proj.weight_shape" not in exported
    assert exported["model.layers.0.self_attn.q_proj.weight"] is attention


def test_qwen3_5_fused_int4_expansion_happens_after_ipc():
    packed = torch.arange(2 * 12 * 4, dtype=torch.int32).reshape(2, 12, 4)
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj.weight_packed"

    expanded = list(expand_qwen3_5_fused_int4_weights([(name, packed)]))

    assert [item[0] for item in expanded] == [
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_packed",
        "model.language_model.layers.0.mlp.experts.0.up_proj.weight_packed",
        "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_packed",
        "model.language_model.layers.0.mlp.experts.1.up_proj.weight_packed",
    ]
    assert all(tensor.shape == (6, 4) for _, tensor in expanded)


def test_wna16_detection_is_semantic_not_class_name_based():
    config = {
        "quant_format": "pack-quantized",
        "target_scheme_map": {
            "Linear": {
                "weights": {"num_bits": 4, "type": "int", "strategy": "group"},
                "input_activations": None,
            }
        },
    }

    assert is_int4_wna16_quant_config(config)
    config["target_scheme_map"]["Linear"]["input_activations"] = {"num_bits": 8}
    assert not is_int4_wna16_quant_config(config)


def test_int4_layerwise_reload_preserves_wna16_derived_tensors():
    skip_tensors = {"_expert_map"}

    configure_int4_layerwise_reload(skip_tensors)

    assert skip_tensors == {
        "_expert_map",
        "w13_weight_shape",
        "w2_weight_shape",
        "w13_weight_g_idx",
        "w2_weight_g_idx",
        "w13_g_idx_sort_indices",
        "w2_g_idx_sort_indices",
    }


def test_int4_vllm_backend_can_force_generic_method(monkeypatch):
    module = SimpleNamespace(check_moe_marlin_supports_layer=lambda *_args, **_kwargs: True)
    monkeypatch.setenv("VERL_INT4_QAT_FORCE_GENERIC_WNA16", "1")

    assert configure_int4_vllm_backend(moe_module=module)
    assert not module.check_moe_marlin_supports_layer(object(), 128)


def test_int4_reload_owns_only_the_layerwise_views_still_pending():
    """Completed layers do not need an extra IPC-buffer clone."""
    source = torch.ones(4)
    retained = source.view_as(source)
    unrelated = torch.zeros(4)
    retained_info = SimpleNamespace(
        loaded_weights=[("weight", SimpleNamespace(arguments={"loaded_weight": retained, "param": unrelated}))]
    )
    complete_info = SimpleNamespace(loaded_weights=[])
    model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
    infos = {module: (retained_info if index == 1 else complete_info) for index, module in enumerate(model.modules())}

    tensors, byte_count = own_pending_int4_reload_views(
        [model],
        [("source.weight", source)],
        get_layerwise_info=infos.__getitem__,
    )

    assert tensors == 1
    assert byte_count == retained.nbytes
    owned = retained_info.loaded_weights[0][1].arguments["loaded_weight"]
    assert owned.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    source.zero_()
    torch.testing.assert_close(owned, torch.ones(4))


def test_vllm_layerwise_reload_completes_without_derived_wna16_updates():
    reload_api = pytest.importorskip("vllm.model_executor.model_loader.reload")
    layerwise_reload = pytest.importorskip("vllm.model_executor.model_loader.reload.layerwise")

    configure_int4_layerwise_reload()

    class _FakeWNA16Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name, size in {
                "w13_weight_packed": 4,
                "w2_weight_packed": 4,
                "w13_weight_scale": 2,
                "w2_weight_scale": 2,
                "w13_weight_shape": 2,
                "w2_weight_shape": 2,
                "w13_weight_g_idx": 3,
                "w2_weight_g_idx": 3,
                "w13_g_idx_sort_indices": 3,
                "w2_g_idx_sort_indices": 3,
            }.items():
                self.register_parameter(name, torch.nn.Parameter(torch.zeros(size), requires_grad=False))

    layer = _FakeWNA16Layer()
    derived_before = {
        name: parameter.clone()
        for name, parameter in layer.named_parameters()
        if name.endswith(("weight_shape", "weight_g_idx", "g_idx_sort_indices"))
    }
    reload_api.record_metadata_for_reloading(layer)
    reload_api.initialize_layerwise_reload(layer)

    info = layerwise_reload.get_layerwise_info(layer)
    assert info.load_numel_total == 12
    for name in ("w13_weight_packed", "w2_weight_packed", "w13_weight_scale", "w2_weight_scale"):
        parameter = getattr(layer, name)
        loaded_weight = torch.ones(parameter.shape, dtype=parameter.dtype, device="cpu")
        parameter.weight_loader(parameter, loaded_weight)

    assert info.load_numel == 0
    assert all(torch.equal(getattr(layer, name), value) for name, value in derived_before.items())


class _FakeGroupedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 128))

    def _get_weight_tensors(self):
        return [self.weight]


class _FakeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = _FakeGroupedLinear()
        self.linear_fc2 = _FakeGroupedLinear()


class _FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _FakeExperts()


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _FakeMLP()


def test_megatron_hook_patches_only_routed_expert_grouped_linears():
    model = _FakeModel()
    config = type("Config", (), {"group_size": 128, "scale_dtype": "bfloat16"})()

    apply_int4_qat_to_modules([model], config)
    quantized_weight = model.mlp.experts.linear_fc1._get_weight_tensors()[0]

    assert not torch.equal(quantized_weight, model.mlp.experts.linear_fc1.weight)
    quantized_weight.sum().backward()
    assert torch.equal(model.mlp.experts.linear_fc1.weight.grad, torch.ones_like(quantized_weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Transformer Engine")
def test_megatron_hook_runs_real_te_grouped_linear_forward_backward():
    """Exercise the exact Transformer Engine hook used by Qwen3 routed experts."""
    from transformer_engine.pytorch import GroupedLinear

    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.Module()
    model.mlp.experts.linear_fc1 = GroupedLinear(
        num_gemms=2,
        in_features=128,
        out_features=64,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    config = type("Config", (), {"group_size": 32, "scale_dtype": "bfloat16"})()
    apply_int4_qat_to_modules([model], config)

    inp = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = model.mlp.experts.linear_fc1(inp, m_splits=[2, 3])
    output.float().square().mean().backward()

    assert output.shape == (5, 64)
    assert inp.grad is not None
    original_weights = model.mlp.experts.linear_fc1._verl_int4_original_get_weight_tensors()
    assert all(weight.grad is not None for weight in original_weights)
