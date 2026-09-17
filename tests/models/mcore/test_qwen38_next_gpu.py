# SPDX-License-Identifier: Apache-2.0
"""Opt-in, memory-bounded GPU component tests; not full-checkpoint acceptance.

RUN_QWEN38_GPU_TESTS=1 torchrun --standalone --nproc-per-node=1 -m pytest -s -q <this file>
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get("RUN_QWEN38_GPU_TESTS") != "1", reason="explicit GPU opt-in required")


@pytest.fixture(scope="module", autouse=True)
def gpu_context():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        pytest.skip("Need 8 GiB free headroom; never evict another job")
    torch.cuda.set_per_process_memory_fraction(min(0.025, 3 * 1024**3 / total))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.distributed.init_process_group("nccl")
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    yield
    torch.cuda.synchronize()
    print(f"QWEN38_GPU_PEAK_ALLOCATED_MIB={torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hc_forward_backward_matches_torch(dtype):
    from verl.models.mcore.qwen3_8_next.ops.kernel.hc_triton import hc_combine_triton, hc_mix_inject_triton

    torch.manual_seed(31)
    n, c, rank, tokens = 4, 128, 16, 17
    shapes = [(tokens, n * c), (n * c,), (rank, n * c), (n * c, rank), (n, n * c), (tokens, c)]
    actual = [(torch.randn(shape, device="cuda", dtype=dtype) * 0.05).requires_grad_() for shape in shapes]
    expected = [value.detach().clone().requires_grad_() for value in actual]
    mixed, injection = hc_mix_inject_triton(*actual[:5], n, 1e-6)
    out = hc_combine_triton(actual[0], actual[5] + mixed, injection, n)

    x, weight, down, up, inject, y = expected
    streams = x.float().reshape(tokens, n, c)
    normed = (streams * torch.rsqrt(streams.square().mean(-1, keepdim=True) + 1e-6)).reshape(tokens, n * c)
    normed = normed * (1 + weight.float())
    gate = F.linear(F.silu(F.linear(normed, down.float()) / n), up.float()).sigmoid()
    mixed_ref = (gate * normed).reshape(tokens, n, c).mean(1).to(dtype)
    injection_ref = 2 * (F.linear(normed, inject.float()) / n).sigmoid()
    # Match the two operation boundaries: HC-mix and HC-combine each return a
    # BF16 input gradient. Sharing the norm's x.float() node with the residual
    # would instead sum both contributions in FP32 before rounding once.
    residual_ref = x.float().reshape(tokens, n, c)
    expected_out = (
        (residual_ref + injection_ref[..., None] * (y + mixed_ref).float()[:, None, :]).reshape_as(x).to(dtype)
    )
    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out, expected_out, rtol=tolerance, atol=tolerance * 0.01)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected_out.backward(grad)
    for result, reference in zip(actual, expected, strict=True):
        assert result.grad is not None and bool(result.grad.isfinite().all())
        torch.testing.assert_close(result.grad, reference.grad, rtol=tolerance, atol=tolerance * 0.05)


def make_tiny_provider(tmp_path):
    from safetensors.torch import save_file
    from vllm.models.qwen4_exp.config import Qwen4ExpConfig
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding

    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge

    # Same architectural components, deliberately small random weights.
    config = Qwen4ExpConfig(
        text_config=dict(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=128,
            intermediate_size=256,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            hc_count=2,
            hc_lowrank=16,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            full_attention_interval=4,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            ple_layer_ids=[2],
            ple_embed_dim=64,
            ngram_size=3,
            heads_per_ngram=2,
            ngram_vocab_size_base=32,
            split_ngram_parts=4,
            ple_conv_kernel_size=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.5,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
            eos_token_id=2,
            bos_token_id=1,
            dtype="bfloat16",
            max_position_embeddings=256,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            out_hidden_size=128,
            num_position_embeddings=64,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            deepstack_visual_indexes=[],
        ),
    )
    config.architectures = ["Qwen4ExpForConditionalGeneration"]
    config.save_pretrained(tmp_path)
    prefix = "model.language_model.layers.1.ple.ple_embedding"
    sizes, offsets, total_rows = Qwen4ExpNGramEmbedding._make_vocab_layout(
        ngram_vocab_size_base=32, ngram_heads=4, ple_dense_layer_id=0
    )
    divisor = config.text_config.make_ngram_vocab_size_divisible_by
    rows_per_shard = ((total_rows + divisor - 1) // divisor * divisor) // 4
    tensors = {
        f"{prefix}.layer_multipliers": torch.tensor([3, 5, 7]),
        f"{prefix}.ngram_heads_vocab_sizes": torch.tensor(sizes),
        f"{prefix}.ngram_heads_offsets": torch.tensor(offsets),
    }
    for i in range(4):
        tensors[f"{prefix}.ngram_embedding.shard_{i}.weight"] = torch.randn(rows_per_shard, 16).to(torch.bfloat16)
    save_file(tensors, str(tmp_path / "tiny-ple.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "tiny-ple.safetensors")})
    )
    provider = Qwen38NextBridge().provider_bridge(SimpleNamespace(config=config, model_name_or_path=str(tmp_path)))
    provider.moe_router_load_balancing_type = "none"
    provider.moe_permute_fusion = False
    provider.sequence_parallel = False
    provider.language_max_sequence_length = 256
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    # Standalone autograd test, without Megatron DDP's main_grad buffers.
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    from megatron.core.process_groups_config import ProcessGroupCollection

    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return provider


def test_tiny_vlm_lora_update_export_and_recompute(tmp_path):
    from megatron.bridge.peft.lora import LoRA
    from megatron.core.packed_seq_params import PackedSeqParams

    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge

    provider = make_tiny_provider(tmp_path)
    # As in the real recipe, configure recompute BEFORE applying PEFT so Bridge
    # installs its frozen-embedding input-grad hook. Enabling it only after
    # PEFT setup bypasses that required initialization.
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    model = provider.provide(pre_process=True, post_process=True).cuda()
    registry = Qwen38NextBridge().mapping_registry()
    missing = [name for name, _ in model.named_parameters() if registry.megatron_to_hf_lookup(name) is None]
    assert not missing, f"Target parameters lacking mapping: {missing}"
    ids = torch.randint(3, 200, (1, 16), device="cuda")
    positions = torch.arange(16, device="cuda").reshape(1, -1).repeat(3, 1, 1)
    cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=16, max_seqlen_kv=16)
    inputs = dict(input_ids=ids, position_ids=positions, attention_mask=None, packed_seq_params=packed)
    model.eval()
    with torch.no_grad():
        base = model(**inputs)
    fixture_dir = os.environ.get("QWEN38_TINY_EXPORT_DIR")
    if fixture_dir:
        from safetensors.torch import load_file, save_file
        from transformers import AutoConfig

        fixture_dir = Path(fixture_dir)
        model_path = fixture_dir / "model"
        model_path.mkdir(parents=True, exist_ok=False)
        hf_config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
        hf_config.save_pretrained(model_path)
        exported_base = {
            item.param_name: item.weight.detach().contiguous().clone()
            for item in Qwen38NextBridge().stream_weights_megatron_to_hf(
                [model], hf_config, cpu=True, show_progress=False
            )
        }
        exported_base.update(load_file(str(tmp_path / "tiny-ple.safetensors")))
        save_file(exported_base, str(model_path / "model.safetensors"))
    peft = LoRA(
        dim=4,
        alpha=8,
        target_modules=[
            "language_model.decoder.layers.*.self_attention.in_proj",
            "language_model.decoder.layers.*.self_attention.out_proj",
            "language_model.decoder.layers.*.self_attention.linear_qkv",
            "language_model.decoder.layers.*.self_attention.linear_proj",
            "language_model.decoder.layers.*.mlp.*.linear_fc1",
            "language_model.decoder.layers.*.mlp.*.linear_fc2",
        ],
    )
    model = peft([model])[0]
    peft.set_params_to_save([model])
    model.eval()
    with torch.no_grad():
        zero_lora = model(**inputs)
    torch.testing.assert_close(zero_lora, base, atol=0, rtol=0)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("adapter" in name for name, _ in trainable)
    for family in (".self_attention.in_proj.", ".self_attention.linear_qkv.", ".mlp.experts.", ".mlp.shared_experts."):
        assert any(family in name for name, _ in trainable), family
    frozen = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not p.requires_grad}
    before = {name: p.detach().clone() for name, p in trainable}
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-2)
    model.train()
    logits = model(**inputs)
    loss = logits.float().square().mean()
    assert bool(loss.isfinite())
    loss.backward()
    assert any(p.grad is not None and bool((p.grad != 0).any()) for _, p in trainable)
    assert all(p.grad is None or bool(p.grad.isfinite().all()) for _, p in trainable)
    optimizer.step()
    assert any(not torch.equal(p, before[name]) for name, p in trainable)
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p.cpu(), frozen[name], rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        updated = model(**inputs)
    assert bool(updated.isfinite().all()) and not torch.equal(updated, zero_lora)
    with peft.disable_adapter([model]), torch.no_grad():
        torch.testing.assert_close(model(**inputs), base, rtol=0, atol=0)

    exports = list(Qwen38NextBridge().stream_adapter_weights_megatron_to_hf([model], cpu=True, show_progress=False))
    weights = {entry.param_name: entry.weight for entry in exports}
    assert weights and len(weights) == len(exports), "Missing or duplicate adapter export keys"
    assert all("lora_" in name and bool(weight.isfinite().all()) for name, weight in weights.items())
    for target in (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "experts",
        "shared_expert",
    ):
        assert any(target in name for name in weights), target
    print(f"QWEN38_TINY_ADAPTER_EXPORT_TENSORS={len(exports)}")
    if fixture_dir:
        from megatron.bridge.models.conversion.peft_bridge import (
            build_adapter_config_dict,
            convert_adapter_weights_to_peft_state,
            infer_rank_pattern_from_adapter_weights,
            infer_target_modules_from_adapter_weights,
        )

        adapter_state, adapter_keys, target_parameters = convert_adapter_weights_to_peft_state(exports)
        adapter_config = build_adapter_config_dict(
            peft,
            target_modules=infer_target_modules_from_adapter_weights(adapter_keys),
            target_parameters=target_parameters,
            base_model_name_or_path=str(model_path),
            rank_pattern=infer_rank_pattern_from_adapter_weights(exports, default_rank=peft.dim),
        )
        adapter_dir = fixture_dir / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
        save_file(
            {name: value.contiguous().clone() for name, value in adapter_state.items()},
            str(adapter_dir / "adapter_model.safetensors"),
        )
        save_file(
            {"input_ids": ids.cpu(), "base_logits": base.cpu(), "adapter_logits": updated.cpu()},
            str(fixture_dir / "reference.safetensors"),
        )

    # Native distributed adapter checkpoint round trip (not a trainer/optimizer
    # resume test): corrupt the live adapters, load, then compare every tensor.
    from megatron.bridge.peft.utils import _model_state_dict, load_peft_adapter_checkpoint
    from megatron.bridge.training.checkpointing import apply_peft_adapter_filter_to_state_dict
    from megatron.core import dist_checkpointing

    adapter_path = tmp_path / "adapter_dist_ckpt"
    adapter_path.mkdir()
    state = apply_peft_adapter_filter_to_state_dict(
        _model_state_dict([model], pg_collection=provider._pg_collection), peft
    )
    dist_checkpointing.save(state, str(adapter_path))
    saved = {name: p.detach().clone() for name, p in trainable}
    with torch.no_grad():
        for _, p in trainable:
            p.zero_()
    load_peft_adapter_checkpoint([model], adapter_path, peft, pg_collection=provider._pg_collection)
    for name, p in trainable:
        torch.testing.assert_close(p, saved[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs), updated, rtol=0, atol=0)

    # Compare two distinct microbatches, each containing two packed documents.
    # Frozen PLE context must survive backward replay but not leak to the next
    # forward. The same weights/inputs are used with and without recompute.
    from verl.models.mcore.qwen3_8_next.ops.ple import current_ple_batch

    model.train()
    decoder_config = model.language_model.decoder.config
    for offset in (0, 7):
        batch = dict(inputs)
        batch["input_ids"] = ids + offset
        boundaries = torch.tensor([0, 8, 16], dtype=torch.int32, device="cuda")
        batch["packed_seq_params"] = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=boundaries,
            cu_seqlens_kv=boundaries,
            max_seqlen_q=8,
            max_seqlen_kv=8,
        )
        batch["position_ids"] = torch.arange(8, device="cuda").repeat(2).reshape(1, 1, 16).repeat(3, 1, 1)
        reference_logits, reference_grads = None, None
        for chunk in (None, 1, 2):
            decoder_config.recompute_granularity = "full" if chunk else None
            decoder_config.recompute_method = "uniform" if chunk else None
            decoder_config.recompute_num_layers = chunk
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            result.float().square().mean().backward()
            grads = {name: p.grad.clone() for name, p in trainable if p.grad is not None}
            assert grads and all(bool(value.isfinite().all()) for value in grads.values())
            with pytest.raises(RuntimeError, match="no n-gram ids published"):
                current_ple_batch()
            assert all(not getattr(module, "_ple_recompute_fifo", []) for module in model.modules())
            if chunk is None:
                reference_logits, reference_grads = result.detach(), grads
            else:
                torch.testing.assert_close(result, reference_logits, rtol=0, atol=0)
                assert grads.keys() == reference_grads.keys()
                for name, grad in grads.items():
                    torch.testing.assert_close(grad, reference_grads[name], rtol=0.02, atol=2e-5, msg=name)
