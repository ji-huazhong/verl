# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; these do not substitute for real-weight GPU parity."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.models.mcore.qwen3_8_next.config import validate_rollout, validate_runtime
from verl.models.mcore.qwen3_8_next.ops.sequence import apply_indexer_rope, packed_token_segments


@pytest.mark.parametrize("bounds", [[0, 3, 5], [0, 0, 3, 3, 5, 5], [0, 0, 0]])
def test_segments_include_empty_sequences(bounds):
    cu = torch.tensor(bounds, dtype=torch.int32)
    segment, position = packed_token_segments(cu, bounds[-1])
    expected_segments, expected_positions = [], []
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:], strict=False)):
        expected_segments.extend([i] * (hi - lo))
        expected_positions.extend(range(hi - lo))
    assert segment.tolist() == expected_segments
    assert position.tolist() == expected_positions


@pytest.mark.parametrize("bounds,total", [([1, 3], 3), ([0, 4], 3), ([0, 3, 2, 4], 4)])
def test_invalid_packing_fails(bounds, total):
    with pytest.raises(ValueError):
        packed_token_segments(torch.tensor(bounds), total)


def test_indexer_partial_rope_uses_absolute_composed_angles():
    x = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
    half_angles = torch.tensor([[0.1, 0.3], [1.2, 0.7], [0.4, 1.8]])
    angles = half_angles.repeat(1, 2)
    out = apply_indexer_rope(x, angles[:, None, None, :])
    expected = x.clone()
    for t in range(3):
        for h in range(2):
            for i in range(2):
                a, b = x[t, h, i], x[t, h, i + 2]
                expected[t, h, i] = a * half_angles[t, i].cos() - b * half_angles[t, i].sin()
                expected[t, h, i + 2] = b * half_angles[t, i].cos() + a * half_angles[t, i].sin()
    torch.testing.assert_close(out, expected)
    assert torch.equal(out[..., 4:], x[..., 4:])


@pytest.mark.parametrize("sharded", [False, True])
def test_ple_weight_map_accepts_both_hf_safetensors_layouts(tmp_path, sharded):
    pytest.importorskip("megatron.bridge")
    from safetensors.torch import save_file

    from verl.models.mcore.qwen3_8_next.ops.ple import _weight_map

    name = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    filename = "model-00001-of-00001.safetensors" if sharded else "model.safetensors"
    save_file({name: torch.ones(2, 3)}, str(tmp_path / filename))
    if sharded:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: filename}}))
    assert _weight_map(str(tmp_path)) == {name: filename}


def test_gdn_output_gate_is_sigmoid_not_mlp_silu():
    pytest.importorskip("megatron.bridge")
    from verl.models.mcore.qwen3_8_next.ops.gated_delta_net import Qwen38NextGatedDeltaNet

    config = SimpleNamespace(qwen3_8_next_output_gate_type="sigmoid", activation_func=torch.nn.functional.silu)
    fake = SimpleNamespace(config=config, out_norm=torch.nn.Identity())
    x = torch.ones(1, 2, 1, 3, requires_grad=True)
    gate = torch.tensor([[[[-2.0, 0.0, 2.0]], [[-1.0, 0.5, 1.0]]]], requires_grad=True)
    out = Qwen38NextGatedDeltaNet._apply_gated_norm(fake, x, gate)
    torch.testing.assert_close(out, gate.reshape(-1, 3).sigmoid())
    out.sum().backward()
    torch.testing.assert_close(gate.grad, gate.sigmoid() * (1 - gate.sigmoid()))
    assert config.activation_func is torch.nn.functional.silu


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("virtual_pipeline_model_parallel_size", 2),
        ("mtp_num_layers", 1),
        ("recompute_granularity", "selective"),
        ("cuda_graph_impl", "local"),
        ("fp8", "hybrid"),
        ("cpu_offloading", True),
    ],
)
def test_unsupported_runtime_fails_early(field, value):
    config = SimpleNamespace(num_residual_streams=4, qwen3_8_next_indexer_kv_heads=1)
    setattr(config, field, value)
    with pytest.raises(NotImplementedError):
        validate_runtime(config)


@pytest.mark.parametrize("load_format", ["dummy", "pt", None])
def test_frozen_ple_rejects_non_checkpoint_rollout(load_format):
    hf = SimpleNamespace(model_type="qwen4_exp", text_config=SimpleNamespace(ple_layer_ids=[2]))
    with pytest.raises(ValueError, match="frozen PLE"):
        validate_rollout(hf, SimpleNamespace(load_format=load_format))


@pytest.mark.parametrize("key", ["load_format", "load-format"])
def test_frozen_ple_cannot_bypass_guard_via_engine_kwargs(key):
    hf = SimpleNamespace(model_type="qwen4_exp", text_config=SimpleNamespace(ple_layer_ids=[2]))
    rollout = SimpleNamespace(load_format="auto", engine_kwargs={"vllm": {key: "dummy"}})
    with pytest.raises(ValueError, match="frozen PLE"):
        validate_rollout(hf, rollout)


def test_frozen_ple_allows_real_base_loading():
    hf = SimpleNamespace(model_type="qwen4_exp", text_config=SimpleNamespace(ple_layer_ids=[2]))
    for load_format in ("auto", "safetensors"):
        validate_rollout(hf, SimpleNamespace(load_format=load_format))
    validate_rollout(SimpleNamespace(model_type="different_architecture"), SimpleNamespace(load_format="dummy"))


def test_rollout_calls_explicit_model_plugin_contract(monkeypatch):
    import sys
    from types import ModuleType

    from verl.utils.import_utils import validate_external_model_rollout_config

    plugin = ModuleType("test_model_contract_plugin")
    calls = []
    plugin.validate_verl_rollout = lambda model, rollout: calls.append((model, rollout))
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    model, rollout = SimpleNamespace(external_lib=plugin.__name__), SimpleNamespace()
    validate_external_model_rollout_config(model, rollout)
    assert calls == [(model, rollout)]
    del plugin.validate_verl_rollout
    validate_external_model_rollout_config(model, rollout)  # Existing import-only plugins unchanged.


def test_actual_vllm_server_validation_rejects_dummy_before_engine_allocation():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    # Exercise the real server method without __init__, Ray, or GPU allocation.
    server = object.__new__(vLLMHttpServer)
    server.model_config = SimpleNamespace(
        external_lib="verl.models.mcore.qwen3_8_next.bridge",
        hf_config=SimpleNamespace(model_type="qwen4_exp", text_config=SimpleNamespace(ple_layer_ids=[2])),
    )
    server.config = SimpleNamespace(load_format="dummy")
    with pytest.raises(ValueError, match="frozen PLE"):
        server._validate_configs()


def test_qsa_compression_respects_packed_boundaries():
    pytest.importorskip("megatron.bridge")
    from verl.models.mcore.qwen3_8_next.ops.qsa_indexer import (
        PackedBlockLayout,
        compress_keys_by_mean,
        compress_keys_by_mean_packed,
        packed_block_causal_mask,
    )

    bounds = [0, 0, 301, 301, 512, 512]
    cu = torch.tensor(bounds, dtype=torch.int32)
    segments, positions = packed_token_segments(cu, 512)
    layout = PackedBlockLayout(cu, positions, 4)
    keys = torch.arange(1024, dtype=torch.float32).reshape(512, 2)
    expected = torch.cat([compress_keys_by_mean(keys[:301], 4), compress_keys_by_mean(keys[301:], 4)])
    torch.testing.assert_close(compress_keys_by_mean_packed(keys, layout), expected)
    mask = packed_block_causal_mask(positions, layout, 4)
    assert not bool((mask & (segments[:, None] != layout.block_seq[None, :])).any())


def test_ple_contexts_do_not_cross_documents():
    pytest.importorskip("megatron.bridge")
    from verl.models.mcore.qwen3_8_next.ops.ple import build_ngram_contexts_packed

    tokens = torch.tensor([10, 11, 20, 21, 22])
    contexts = build_ngram_contexts_packed(tokens, torch.tensor([0, 0, 2, 5, 5]), 3, 99)
    assert contexts.tolist() == [[99, 99, 10], [99, 10, 11], [99, 99, 20], [99, 20, 21], [20, 21, 22]]


@pytest.mark.parametrize("tp_size", [1, 2])
def test_lora_gdn_export_preserves_each_projection(tp_size, monkeypatch):
    pytest.importorskip("megatron.bridge")
    from megatron.core import parallel_state

    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge

    # Rank intentionally differs from hidden_size: this catches exporters that
    # reshape a LoRA B tensor as a full base weight. Build the native GDN layout
    # directly instead of using the inverse exporter as our expected result.
    config = SimpleNamespace(
        hidden_size=8, linear_key_head_dim=2, linear_value_head_dim=2, linear_num_key_heads=2, linear_num_value_heads=4
    )
    rank = 3
    generator = torch.Generator().manual_seed(123)
    q, k, v, z, b, a = [torch.randn(rows, rank, generator=generator) for rows in (4, 4, 8, 8, 4, 4)]
    shards = [[part.chunk(tp_size)[i] for part in (q, k, v, z, b, a)] for i in range(tp_size)]
    native_b = torch.cat([torch.cat(shard) for shard in shards])
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: tp_size)
    exported = Qwen38NextBridge()._split_gdn_in_proj_linear_out_weight(SimpleNamespace(config=config), native_b)
    expected = {"in_proj_qkv": torch.cat((q, k, v)), "in_proj_z": z, "in_proj_b": b, "in_proj_a": a}
    lora_a = torch.randn(rank, config.hidden_size, generator=generator)
    for key, value in expected.items():
        torch.testing.assert_close(exported[key], value, rtol=0, atol=0)
        torch.testing.assert_close(exported[key] @ lora_a, value @ lora_a, rtol=0, atol=0)


def test_ple_hook_uses_packed_language_boundary_and_clears_on_error(monkeypatch):
    pytest.importorskip("megatron.bridge")
    from verl.models.mcore.qwen3_8_next import provider
    from verl.models.mcore.qwen3_8_next.ops.ple import current_ple_batch

    class FakeTable(torch.nn.Module):
        ngram_size = 3
        eos_token_id = 99

        def compute_ngram_ids(self, contexts):
            return contexts

    class Language(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = FakeTable()

        def forward(self, input_ids, packed_seq_params=None, fail=False):
            observed, _ = current_ple_batch()
            assert observed.tolist() == [[99, 99, 1], [99, 1, 2], [99, 99, 3]]
            if fail:
                raise RuntimeError("injected forward failure")
            return observed

    model = torch.nn.Module()
    model.language_model = Language()
    monkeypatch.setattr(provider, "Qwen38NextFrozenNGramEmbedding", FakeTable)
    provider.install_ple_context_hooks(model)
    ids = torch.tensor([[1, 2, 3]])
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2, 3]))
    model.language_model(input_ids=ids, packed_seq_params=packed)
    with pytest.raises(RuntimeError, match="no n-gram ids published"):
        current_ple_batch()
    with pytest.raises(RuntimeError, match="injected forward failure"):
        model.language_model(input_ids=ids, packed_seq_params=packed, fail=True)
    with pytest.raises(RuntimeError, match="no n-gram ids published"):
        current_ple_batch()


def test_public_checkpoint_config_and_key_coverage():
    checkpoint = os.environ.get("QWEN38_MODEL_PATH")
    if not checkpoint:
        pytest.skip("Set QWEN38_MODEL_PATH to a local public config + safetensors index")
    from transformers import AutoConfig

    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.provider import build_flash_next_spec

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    bridge = Qwen38NextBridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=config, model_name_or_path=checkpoint))
    assert provider.num_layers == config.text_config.num_hidden_layers
    assert provider.num_residual_streams == config.text_config.hc_count
    assert provider.vision_config is config.vision_config
    assert provider.mtp_num_layers is None
    # No process group or GPU tensor allocation is needed to build a block spec.
    spec = build_flash_next_spec(provider, pp_rank=0)
    from verl.models.mcore.qwen3_8_next.ops.gated_delta_net import Qwen38NextGatedDeltaNet

    assert provider.qwen3_8_next_output_gate_type == "sigmoid"
    assert sum(s.submodules.self_attention.module is Qwen38NextGatedDeltaNet for s in spec.layer_specs) == 36
    assert len(spec.layer_specs) == provider.num_layers
    assert sum(hasattr(s.submodules.self_attention.submodules, "linear_qkv") for s in spec.layer_specs) == 12
    registry = bridge.mapping_registry()
    keys = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    unmapped = []
    for key in keys:
        if key.startswith("mtp.") or ".ple.ple_embedding." in key:
            continue  # Explicitly disabled MTP or frozen table loaded directly.
        if registry.hf_to_megatron_lookup(key) is None:
            unmapped.append(key)
    assert not unmapped, f"Unmapped source tensors: {unmapped[:20]} (total={len(unmapped)})"
