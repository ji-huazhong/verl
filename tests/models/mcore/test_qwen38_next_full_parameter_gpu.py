# Copyright 2026 Individual Contributor
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
    torch.cuda.set_per_process_memory_fraction(min(0.10, 8 * 1024**3 / total))
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
@pytest.mark.parametrize("input_scale", [1.0, 0.001])
def test_gdn_gated_norm_keeps_fp32_intermediates_and_gradients(dtype, input_scale):
    """Check the single-rounding forward and all three gradients against FP64."""
    from verl.models.mcore.qwen3_8_next.ops.gated_delta_net import Qwen38NextGatedDeltaNet

    torch.manual_seed(73)
    x = (torch.randn(7, 6, 128, device="cuda", dtype=dtype) * input_scale).requires_grad_()
    gate = torch.randn_like(x, requires_grad=True)
    gamma = torch.randn(128, device="cuda", dtype=dtype, requires_grad=True)
    block = SimpleNamespace(
        config=SimpleNamespace(qwen3_8_next_output_gate_type="sigmoid", layernorm_epsilon=1e-6),
        out_norm=SimpleNamespace(weight=gamma),
    )
    actual = Qwen38NextGatedDeltaNet._apply_gated_norm(block, x, gate)
    ref_x, ref_gate, ref_gamma = [value.detach().double().requires_grad_() for value in (x, gate, gamma)]
    expected = F.rms_norm(ref_x, (128,), weight=ref_gamma, eps=1e-6) * ref_gate.sigmoid()
    rounded = expected.reshape_as(actual).to(dtype)
    forward_error = (actual.double() - rounded.double()).norm() / rounded.double().norm()
    assert forward_error < 1e-4
    assert actual.dtype == gamma.dtype == dtype
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.reshape_as(expected).double())
    for value, oracle in zip((x, gate, gamma), (ref_x, ref_gate, ref_gamma), strict=True):
        assert value.grad is not None and value.grad.isfinite().all() and value.grad.abs().sum() > 0
        rounded_grad = oracle.grad.to(dtype).double()
        error = (value.grad.double() - rounded_grad).norm() / rounded_grad.norm()
        assert error < 1e-3


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
    from transformers import Qwen4ExpConfig

    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge

    # The default preserves the original fixture. Capacity eight also covers
    # production head geometry: 24Q/2KV, head dim 256, 3:1 GDN value/key
    # heads, four HC streams and quarter-head RoPE, with a tiny hidden size.
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
            num_attention_heads=24 if capacity == 8 else 4,
            num_key_value_heads=2 if capacity == 8 else 1,
            head_dim=256 if capacity == 8 else 128,
            intermediate_size=256,
            num_experts=experts,
            num_experts_per_tok=2,
            moe_intermediate_size=96 if capacity == 8 else 64,
            shared_expert_intermediate_size=80 if capacity == 8 else 64,
            hc_count=4 if capacity == 8 else 2,
            hc_lowrank=16,
            output_gate_type="sigmoid",
            layer_types=["linear_attention"] * 3 + ["qwen_sparse_attention"],
            full_attention_interval=4,
            linear_num_key_heads=capacity,
            linear_num_value_heads=(3 if capacity == 8 else 2) * capacity,
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
            make_ngram_vocab_size_divisible_by=128,
            ple_conv_kernel_size=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25 if capacity == 8 else 0.5,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
            eos_token_id=2,
            bos_token_id=1,
            dtype="bfloat16",
            max_position_embeddings=4096,
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
    sizes, offsets, total_rows = [37, 41, 43, 47], [0, 37, 78, 121], 168
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
    provider.qwen3_8_next_gdn_backend = os.environ.get("QWEN38_TEST_GDN_BACKEND", "fla")
    # Standalone autograd test, without Megatron DDP's main_grad buffers.
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    from megatron.core.process_groups_config import ProcessGroupCollection

    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return provider


def test_frozen_ple_table_keeps_reading_layers_trainable(tmp_path):
    from megatron.core import parallel_state

    from verl.models.mcore.qwen3_8_next.ops.ple import Qwen38NextPLE, build_ngram_contexts

    provider = make_tiny_provider(tmp_path)
    provider.qwen3_8_next_train_ple = False
    ple = Qwen38NextPLE(provider, 2, tp_group=parallel_state.get_tensor_model_parallel_group()).cuda()
    embedding = ple.ple_embedding
    embedding.load_from_hf(str(tmp_path))
    table_before = embedding.table.clone()
    assert embedding.table.is_pinned() and not embedding.table.requires_grad
    assert not any("ngram_embedding" in name for name, _ in ple.named_parameters())
    parameters = dict(ple.named_parameters())
    before = {name: value.detach().clone() for name, value in parameters.items()}
    ids = torch.arange(17, device="cuda") % 8
    ngrams = embedding.compute_ngram_ids(build_ngram_contexts(ids, embedding.ngram_size, embedding.eos_token_id))
    state = torch.randn(
        17,
        provider.num_residual_streams * provider.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    optimizer = torch.optim.SGD(ple.parameters(), lr=0.1)
    ple(state, ngrams).float().square().mean().backward()
    assert state.grad is not None and state.grad.isfinite().all() and state.grad.abs().sum() > 0
    for name in ("key_proj.weight", "value_proj.weight", "conv1d_weight"):
        grad = parameters[name].grad
        assert grad is not None and grad.isfinite().all() and grad.abs().sum() > 0, name
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in parameters.items())
    assert torch.equal(embedding.table, table_before)
    assert embedding.table.device.type == "cpu" and embedding.table.is_pinned()


@pytest.mark.parametrize("trainable", [False, True])
def test_ple_meta_inspection_does_not_allocate_table(tmp_path, trainable):
    from megatron.core import parallel_state

    from verl.models.mcore.qwen3_8_next.ops.ple import Qwen38NextNGramEmbedding

    provider = make_tiny_provider(tmp_path)
    provider.qwen3_8_next_train_ple = trainable
    provider.init_model_with_meta_device = True
    provider.use_cpu_initialization = False
    provider.perform_initialization = True
    allocated = torch.cuda.memory_allocated()
    with torch.device("meta"):
        embedding = Qwen38NextNGramEmbedding(provider, 2, tp_group=parallel_state.get_tensor_model_parallel_group())
    table = embedding.ngram_embedding.weight if trainable else embedding.table
    assert table.is_meta
    assert all(buffer.is_meta for buffer in embedding.buffers())
    assert torch.cuda.memory_allocated() == allocated
    assert provider.use_cpu_initialization is False
    assert provider.perform_initialization is True
    if trainable:
        assert table.tensor_model_parallel
        assert table.partition_dim == 0


@pytest.mark.parametrize("lengths,budget,head_dim", [((77, 61), 16, 128), ((2057, 19), 2048, 256)])
def test_qsa_sparse_packed_forward_backward(tmp_path, lengths, budget, head_dim):
    """Exercise real pruning and unaligned packed boundaries against dense math."""
    from verl.models.mcore.qwen3_8_next.ops.attention import Qwen38NextAttention
    from verl.models.mcore.qwen3_8_next.ops.kernel.qsa_block_sparse_attn import qsa_block_sparse_attention_triton
    from verl.models.mcore.qwen3_8_next.ops.qsa_indexer import Qwen38NextQSAIndexer

    provider = make_tiny_provider(tmp_path)
    provider.qwen3_8_next_indexer_budget = budget
    indexer = Qwen38NextQSAIndexer(provider, layer_number=4).cuda()
    total = sum(lengths)
    cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    positions = torch.cat([torch.arange(length, device="cuda") for length in lengths])
    starts = torch.arange(total, device="cuda") - positions
    hidden = torch.randn(total, provider.hidden_size, device="cuda", dtype=torch.bfloat16)
    # Composed angles are deliberately different across documents; compressed
    # keys must use the first token of their own document-local block.
    angles = torch.randn(total, 1, 1, 64, device="cuda") * 0.1
    with torch.no_grad():
        selection = indexer(hidden, positions, cu_seqlens=cu, rotary_pos_emb=angles)
        tail_positions = (positions[:, None] + 1) // 4 * 4 + torch.arange(4, device="cuda")
        tail = (starts[:, None] + tail_positions).masked_fill(tail_positions > positions[:, None], -1)
        selection = torch.cat((selection, tail.to(torch.int32)), dim=1)
    valid = selection >= 0
    assert bool((selection[valid] >= starts[:, None].expand_as(selection)[valid]).all())
    assert bool((selection[valid] <= torch.arange(total, device="cuda")[:, None].expand_as(selection)[valid]).all())
    # Every long document must actually drop causal keys, rather than merely
    # executing a sparse kernel with an effectively dense selection.
    assert int(valid[lengths[0] - 1].sum()) < lengths[0]
    mask = torch.zeros(total, total, device="cuda", dtype=torch.bool)
    rows = torch.arange(total, device="cuda")[:, None].expand_as(selection)
    mask[rows[valid], selection[valid].long()] = True
    assert bool(mask.any(dim=1).all())
    # The incomplete current block is always included. At a complete block's
    # final token, that block competes in top-k and may legitimately be dropped.
    assert bool(mask.diagonal()[(positions + 1) % 4 != 0].all())
    assert not bool(mask[: lengths[0], lengths[0] :].any())
    assert not bool(mask[lengths[0] :, : lengths[0]].any())

    owner = SimpleNamespace(compress_ratio=4, _qsa_cu_seqlens=cu)
    bitmap, lo, hi, block_base, token_base, ratio = Qwen38NextAttention._publish_block_form(
        owner, selection, positions, starts, total
    )
    actual = [
        (torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16) * 0.3).requires_grad_()
        for heads in (4, 1, 1)
    ]
    expected = [tensor.detach().float().requires_grad_() for tensor in actual]
    result = qsa_block_sparse_attention_triton(*actual, bitmap, lo, hi, block_base, token_base, head_dim**-0.5, ratio)
    q, k, v = (tensor.transpose(0, 1) for tensor in expected)
    scores = q @ k.transpose(-1, -2) * head_dim**-0.5
    probability = scores.masked_fill(~mask[None], -torch.inf).softmax(-1)
    reference = (probability @ v).transpose(0, 1)
    torch.testing.assert_close(result.float(), reference, rtol=0.02, atol=0.003)
    gradient = torch.randn_like(result)
    result.backward(gradient)
    reference.backward(gradient.float())
    for name, tensor, oracle in zip(("q", "k", "v"), actual, expected, strict=True):
        assert tensor.grad is not None and bool(tensor.grad.isfinite().all())
        relative_rms = (tensor.grad.float() - oracle.grad).square().mean().sqrt() / oracle.grad.square().mean().sqrt()
        print(f"QWEN38_QSA_SPARSE_{budget}_{head_dim}_{name}_GRAD_REL_RMS={relative_rms.item():.6f}")
        assert relative_rms < 0.02


def test_full_parameter_update_recompute_and_hf_roundtrip(tmp_path):
    from megatron.bridge import AutoBridge
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.module import Float16Module
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig

    from verl.models.mcore.qwen3_8_next.ops.ple import current_ple_batch

    provider = make_tiny_provider(tmp_path)
    model = provider.provide(pre_process=True, post_process=True).cuda()
    if provider.qwen3_8_next_gdn_backend == "flashqla":
        from flash_qla import chunk_gated_delta_rule

        from verl.models.mcore.qwen3_8_next.ops.gated_delta_net import Qwen38NextGatedDeltaNet

        gdn_modules = [m for m in model.modules() if isinstance(m, Qwen38NextGatedDeltaNet)]
        assert len(gdn_modules) == 3 and all(m.gated_delta_rule is chunk_gated_delta_rule for m in gdn_modules)
    mixed = Float16Module(provider, model)
    # There is no PEFT transformation: native text/vision/PLE parameters remain
    # trainable. Text-only examples legitimately leave vision gradients absent.
    assert all(parameter.requires_grad for parameter in model.parameters())
    parameters = dict(model.named_parameters())
    ple_name = next(name for name in parameters if name.endswith("ple_embedding.ngram_embedding.weight"))
    ids = torch.randint(3, 200, (1, 16), device="cuda")
    bounds = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
    inputs = dict(
        input_ids=ids,
        position_ids=torch.arange(8, device="cuda").repeat(2).reshape(1, 1, 16).repeat(3, 1, 1),
        attention_mask=None,
        packed_seq_params=PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=bounds, cu_seqlens_kv=bounds, max_seqlen_q=8, max_seqlen_kv=8
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = model.language_model.decoder.config
    model.train()
    reference_logits = reference_grads = None
    for chunk in (None, 1, 2):
        config.recompute_granularity = "full" if chunk else None
        config.recompute_method = "uniform" if chunk else None
        config.recompute_num_layers = chunk
        optimizer.zero_grad(set_to_none=True)
        logits = mixed(**inputs)
        loss = logits.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in parameters.items()
            if parameter.grad is not None
        }
        assert grads and all(bool(value.isfinite().all()) for value in grads.values())
        assert ple_name in grads and bool(grads[ple_name].abs().sum() > 0)
        for family in ("self_attention.in_proj", "self_attention.linear_qkv", "hyper_connection", "mlp.experts"):
            assert any(family in name and bool(grad.abs().sum() > 0) for name, grad in grads.items()), family
        with pytest.raises(RuntimeError, match="no n-gram ids published"):
            current_ple_batch()
        assert all(not getattr(module, "_ple_recompute_fifo", []) for module in model.modules())
        if chunk is None:
            reference_logits, reference_grads = logits.detach().clone(), grads
        else:
            torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
            assert grads.keys() == reference_grads.keys()
            for name, grad in grads.items():
                torch.testing.assert_close(grad, reference_grads[name], rtol=0.02, atol=2e-5, msg=name)
    before = parameters[ple_name].detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameters[ple_name])
    model.eval()
    with torch.no_grad():
        updated = mixed(**inputs).detach().clone()
    assert not torch.equal(updated, reference_logits)

    # Export every parameter through the production Bridge stream, add only
    # immutable hash metadata, then load through a fresh registered AutoBridge.
    hf_config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    bridge = AutoBridge.from_hf_config(hf_config)
    exported = {
        item.param_name: item.weight.detach().cpu().contiguous().clone()
        for item in bridge.export_hf_weights([model], cpu=True, show_progress=False)
    }
    metadata = load_file(str(tmp_path / "tiny-ple.safetensors"))
    exported.update({name: value for name, value in metadata.items() if "ngram_embedding.shard_" not in name})
    ple_shards = [name for name in exported if "ngram_embedding.shard_" in name]
    assert len(ple_shards) == provider.qwen3_8_next_split_ngram_parts
    checkpoint = Path(os.environ.get("QWEN38_TINY_EXPORT_DIR", str(tmp_path / "export")))
    checkpoint.mkdir(parents=True, exist_ok=False)
    hf_config.save_pretrained(checkpoint)
    save_file(exported, str(checkpoint / "model.safetensors"))
    save_file(
        {"input_ids": ids.cpu(), "logits": updated.cpu()},
        str(checkpoint.parent / (checkpoint.name + "-reference.safetensors")),
    )
    reloader = AutoBridge.from_hf_pretrained(checkpoint, trust_remote_code=False)
    saved = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    with torch.no_grad():
        for parameter in parameters.values():
            parameter.zero_()
    reloader.load_hf_weights([model])
    for name, parameter in parameters.items():
        torch.testing.assert_close(parameter, saved[name], rtol=0, atol=0, msg=name)
    with torch.no_grad():
        torch.testing.assert_close(mixed(**inputs), updated, rtol=0, atol=0)
        from verl.models.mcore.model_forward import gptmodel_forward_model_engine

        documents = torch.nested.nested_tensor([ids[0, :8], ids[0, 8:]], layout=torch.jagged)
        packed_output = gptmodel_forward_model_engine(
            mixed, documents, multi_modal_inputs={}, vision_model=True, pad_token_id=0
        )
        for start, actual in zip((0, 8), packed_output.unbind(), strict=True):
            torch.testing.assert_close(actual, updated[0, start : start + 8], rtol=0, atol=0)
        # An independent target for rollout refit: change only the trainable
        # PLE table. The vLLM check must reload these rows into pinned CPU
        # storage after sleep, then match this Megatron forward.
        parameters[ple_name].zero_()
        ple_zero_logits = mixed(**inputs).detach().cpu()
        assert not torch.equal(ple_zero_logits, updated.cpu())
        save_file(
            {"input_ids": ids.cpu(), "logits": ple_zero_logits},
            str(checkpoint.parent / (checkpoint.name + "-ple-zero-reference.safetensors")),
        )
    print("QWEN38_TINY_FULL_PARAMETER_RECOMPUTE_AND_HF_ROUNDTRIP_PASS", flush=True)
