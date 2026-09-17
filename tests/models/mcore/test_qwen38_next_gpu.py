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
    # The production engine replaces Core's checkpoint backward. Exercising
    # only the unpatched Core path missed its checkpoint-state regression.
    from megatron.core.tensor_parallel import random as tensor_random

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    original_backward = tensor_random.CheckpointFunction.backward
    apply_patch_megatron_recomputation_backward()
    yield
    tensor_random.CheckpointFunction.backward = staticmethod(original_backward)
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

    # The default preserves the original fixture; capacity eight covers the
    # target TP8/EP8 topology without substituting for a full-model run.
    capacity = int(os.environ.get("QWEN38_TINY_TP_CAPACITY", "2"))
    if capacity not in (2, 8):
        raise ValueError("Tiny TP capacity must be 2 or 8")
    experts = max(4, capacity)
    shards = max(4, capacity)
    # Same architectural components, deliberately small random weights.
    config = Qwen4ExpConfig(
        # Keep multimodal special tokens inside this fixture's tiny vocabulary.
        image_token_id=252,
        video_token_id=253,
        vision_start_token_id=250,
        vision_end_token_id=251,
        text_config=dict(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=max(4, capacity),
            num_key_value_heads=1,
            head_dim=128,
            intermediate_size=256,
            num_experts=experts,
            num_experts_per_tok=2,
            moe_intermediate_size=96 if capacity == 8 else 64,
            shared_expert_intermediate_size=80 if capacity == 8 else 64,
            hc_count=2,
            hc_lowrank=16,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            full_attention_interval=4,
            linear_num_key_heads=capacity,
            linear_num_value_heads=2 * capacity,
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
            split_ngram_parts=shards,
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
            num_heads=max(4, capacity),
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
    rows_per_shard = ((total_rows + divisor - 1) // divisor * divisor) // shards
    tensors = {
        f"{prefix}.layer_multipliers": torch.tensor([3, 5, 7]),
        f"{prefix}.ngram_heads_vocab_sizes": torch.tensor(sizes),
        f"{prefix}.ngram_heads_offsets": torch.tensor(offsets),
    }
    for i in range(shards):
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
    from megatron.core.transformer.module import Float16Module
    from PIL import Image

    from tests.models.mcore.qwen38_vision_fixture import make_tiny_rgb, make_tiny_vision_processor
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge

    provider = make_tiny_provider(tmp_path)
    # As in the real recipe, configure recompute BEFORE applying PEFT so Bridge
    # installs its frozen-embedding input-grad hook. Enabling it only after
    # PEFT setup bypasses that required initialization.
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    model = provider.provide(pre_process=True, post_process=True).cuda()
    # Match verl's real mixed-precision path: raw provider construction leaves
    # some vision parameters FP32, unlike the BF16 language embeddings. The
    # wrapper applies the configured weight dtype and normal forward precision
    # behavior; do not patch the VL model to work around a bare-provider test.
    mixed_model = Float16Module(provider, model)
    registry = Qwen38NextBridge().mapping_registry()
    missing = [name for name, _ in model.named_parameters() if registry.megatron_to_hf_lookup(name) is None]
    assert not missing, f"Target parameters lacking mapping: {missing}"
    ids = torch.randint(3, 200, (1, 16), device="cuda")
    positions = torch.arange(16, device="cuda").reshape(1, -1).repeat(3, 1, 1)
    cu = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=16, max_seqlen_kv=16)
    inputs = dict(input_ids=ids, position_ids=positions, attention_mask=None, packed_seq_params=packed)
    traces = {}
    handles = []

    def record(name, index=None):
        def hook(_module, _args, output):
            value = output[index] if index is not None else output
            traces[name] = value.detach().reshape(-1, value.shape[-1]).cpu().contiguous().clone()

        return hook

    for i, layer in enumerate(model.language_model.decoder.layers):
        for site in ("self_attention", "mlp"):
            hc = getattr(layer, f"{site}_hyper_connection")
            handles.append(hc.register_forward_hook(record(f"trace.{i}.{site}.input", 0)))
            handles.append(hc.register_forward_hook(record(f"trace.{i}.{site}.residual", 3)))
            handles.append(getattr(layer, site).register_forward_hook(record(f"trace.{i}.{site}.output", 0)))
        ple = getattr(layer.self_attention_hyper_connection, "ple", None)
        if ple is not None:
            handles.append(ple.register_forward_hook(record(f"trace.{i}.ple")))
    handles.append(model.language_model.decoder.final_layernorm.register_forward_hook(record("trace.contraction")))
    model.eval()
    with torch.no_grad():
        base = mixed_model(**inputs)
    for handle in handles:
        handle.remove()
    # Exercise the actual vision tower and VL wrapper, not just processor
    # metadata. Sixteen input patches merge into four image token embeddings.
    image_ids = ids.clone()
    image_ids[0, 3:9] = torch.tensor([250, 252, 252, 252, 252, 251], device="cuda")
    processor = make_tiny_vision_processor()
    image_rgb = make_tiny_rgb()
    processed = processor.image_processor(images=[Image.fromarray(image_rgb)], return_tensors="pt")
    changed_processed = processor.image_processor(images=[Image.fromarray(255 - image_rgb)], return_tensors="pt")
    image_pixels = processed["pixel_values"].cuda()
    image_grid = processed["image_grid_thw"].cuda()
    changed_pixels = changed_processed["pixel_values"].cuda()
    assert image_pixels.shape == (16, 1536) and image_grid.tolist() == [[1, 4, 4]]
    image_inputs = dict(inputs, input_ids=image_ids, pixel_values=image_pixels, image_grid_thw=image_grid)
    vision_outputs = []
    vision_positions = []
    vision_hook = model.vision_model.register_forward_hook(
        lambda _module, _args, output: vision_outputs.append(output[0].detach().cpu().clone())
    )
    position_hook = model.language_model.register_forward_pre_hook(
        lambda _module, _args, kwargs: vision_positions.append(kwargs["position_ids"].detach().cpu().clone()),
        with_kwargs=True,
    )
    with torch.no_grad():
        image_base = mixed_model(**image_inputs)
        changed_image = mixed_model(**dict(image_inputs, pixel_values=changed_pixels))
        torch.testing.assert_close(mixed_model(**image_inputs), image_base, rtol=0, atol=0)
    vision_hook.remove()
    position_hook.remove()
    assert len(vision_outputs) == 3 and vision_outputs[0].shape == (4, 128)
    assert bool(image_base.isfinite().all()) and bool(changed_image.isfinite().all())
    assert not torch.equal(vision_outputs[0], vision_outputs[1]), "Vision encoder ignored changed pixels"
    assert not torch.equal(image_base, changed_image), "Language output ignored changed vision embeddings"
    print("QWEN38_TINY_VISION_FORWARD_PASSED image_tokens=4")
    fixture_dir = os.environ.get("QWEN38_TINY_EXPORT_DIR")
    if fixture_dir:
        from safetensors.torch import load_file, save_file
        from transformers import AutoConfig

        fixture_dir = Path(fixture_dir)
        model_path = fixture_dir / "model"
        model_path.mkdir(parents=True, exist_ok=False)
        hf_config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
        hf_config.save_pretrained(model_path)
        processor.save_pretrained(model_path)
        exported_base = {
            item.param_name: item.weight.detach().contiguous().clone()
            for item in Qwen38NextBridge().stream_weights_megatron_to_hf(
                [model], hf_config, cpu=True, show_progress=False
            )
        }
        exported_base.update(load_file(str(tmp_path / "tiny-ple.safetensors")))
        save_file(exported_base, str(model_path / "model.safetensors"))
    lora_rank = max(4, provider.tensor_model_parallel_size, int(os.environ.get("QWEN38_TINY_TP_CAPACITY", "2")))
    lora_rank = int(os.environ.get("QWEN38_TINY_LORA_RANK", str(lora_rank)))
    assert lora_rank > 0 and lora_rank % provider.tensor_model_parallel_size == 0
    peft = LoRA(
        dim=lora_rank,
        alpha=2 * lora_rank,
        target_modules=[
            "language_model.decoder.layers.*.self_attention.in_proj",
            "language_model.decoder.layers.*.self_attention.out_proj",
            "language_model.decoder.layers.*.self_attention.linear_qkv",
            "language_model.decoder.layers.*.self_attention.linear_proj",
            "language_model.decoder.layers.*.mlp.*.linear_fc1",
            "language_model.decoder.layers.*.mlp.*.linear_fc2",
        ],
    )
    original_parameters = {id(p): p.detach().cpu().clone() for p in model.parameters()}
    original_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    model = peft([model])[0]
    peft.set_params_to_save([model])
    # PEFT modifies this model in place. Do not wrap/cast a second time after
    # warmup: dynamically created FP32 vision RoPE caches would be downcast.
    assert mixed_model.module is model
    assert original_parameters.keys() <= {id(p) for p in model.parameters()}
    for name, param in model.named_parameters():
        if id(param) in original_parameters:
            torch.testing.assert_close(param.cpu(), original_parameters[id(param)], rtol=0, atol=0, msg=name)
    for name, value in model.named_buffers():
        if name in original_buffers:
            torch.testing.assert_close(value.cpu(), original_buffers[name], rtol=0, atol=0, msg=name)
    model.eval()
    with torch.no_grad():
        zero_lora = mixed_model(**inputs)
        print(f"QWEN38_ZERO_LORA_TEXT_MAX_DIFF={(zero_lora - base).abs().max().item():.8f}")
        torch.testing.assert_close(mixed_model(**image_inputs), image_base, atol=0, rtol=0)
    torch.testing.assert_close(zero_lora, base, atol=0, rtol=0)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("adapter" in name for name, _ in trainable)
    for family in (".self_attention.in_proj.", ".self_attention.linear_qkv.", ".mlp.experts.", ".mlp.shared_experts."):
        assert any(family in name for name, _ in trainable), family
    frozen = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not p.requires_grad}
    before = {name: p.detach().clone() for name, p in trainable}
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-2)
    model.train()
    logits = mixed_model(**inputs)
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
        updated = mixed_model(**inputs)
    assert bool(updated.isfinite().all()) and not torch.equal(updated, zero_lora)
    with peft.disable_adapter([model]), torch.no_grad():
        torch.testing.assert_close(mixed_model(**inputs), base, rtol=0, atol=0)
        torch.testing.assert_close(mixed_model(**image_inputs), image_base, rtol=0, atol=0)
    with torch.no_grad():
        image_updated = mixed_model(**image_inputs)
        changed_image_updated = mixed_model(**dict(image_inputs, pixel_values=changed_pixels))
    assert bool(image_updated.isfinite().all()) and not torch.equal(image_updated, image_base)

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
            {name: value.detach().contiguous().clone() for name, value in weights.items()},
            str(fixture_dir / "raw_adapter.safetensors"),
        )
        save_file(
            {
                "input_ids": ids.cpu(),
                "base_logits": base.cpu(),
                "adapter_logits": updated.cpu(),
                "lora_rank": torch.tensor(peft.dim),
                "lora_alpha": torch.tensor(peft.alpha),
                "vision_input_ids": image_ids.cpu(),
                "vision_pixel_values": image_pixels.cpu(),
                "vision_rgb": torch.from_numpy(image_rgb.copy()),
                "vision_changed_pixel_values": changed_pixels.cpu(),
                "vision_image_grid_thw": image_grid.cpu(),
                "vision_position_ids": vision_positions[0],
                "vision_embeddings": vision_outputs[0],
                "vision_base_logits": image_base.cpu(),
                "vision_adapter_logits": image_updated.cpu(),
                "vision_changed_base_logits": changed_image.cpu(),
                "vision_changed_adapter_logits": changed_image_updated.cpu(),
                **traces,
            },
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
        torch.testing.assert_close(mixed_model(**inputs), updated, rtol=0, atol=0)

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
            result = mixed_model(**batch)
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

    # The vision tower is frozen by PEFT, but image-conditioned language LoRA
    # must still receive gradients. Verify full recompute against no recompute.
    image_reference, image_grads = None, None
    for chunk in (None, 1):
        decoder_config.recompute_granularity = "full" if chunk else None
        decoder_config.recompute_method = "uniform" if chunk else None
        decoder_config.recompute_num_layers = chunk
        optimizer.zero_grad(set_to_none=True)
        result = mixed_model(**image_inputs)
        result.float().square().mean().backward()
        grads = {name: p.grad.detach().clone() for name, p in trainable if p.grad is not None}
        assert grads and all(bool(value.isfinite().all()) for value in grads.values())
        for family in (".self_attention.", ".mlp.experts.", ".mlp.shared_experts."):
            assert any(family in name and bool((grad != 0).any()) for name, grad in grads.items()), family
        assert all(not p.requires_grad and p.grad is None for p in model.vision_model.parameters())
        with pytest.raises(RuntimeError, match="no n-gram ids published"):
            current_ple_batch()
        assert all(not getattr(module, "_ple_recompute_fifo", []) for module in model.modules())
        if chunk is None:
            image_reference, image_grads = result.detach(), grads
        else:
            torch.testing.assert_close(result, image_reference, rtol=0, atol=0)
            assert grads.keys() == image_grads.keys()
            for name, grad in grads.items():
                torch.testing.assert_close(grad, image_grads[name], rtol=0.02, atol=2e-5, msg=name)
    print("QWEN38_TINY_VISION_LORA_RECOMPUTE_PASSED")
