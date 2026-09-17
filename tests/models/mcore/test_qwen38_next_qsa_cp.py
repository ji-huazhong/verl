# SPDX-License-Identifier: Apache-2.0
"""Opt-in two-rank QSA CP gate, not full Flash-Next CP acceptance.

RUN_QWEN38_QSA_CP_TESTS=1, QWEN38_TINY_EXPORT_DIR=<original tiny fixture>,
torchrun --standalone --nproc-per-node=2 -m pytest -s -q <this file>.
Provider-level CP remains guarded until PLE and end-to-end tests pass.
"""

import copy
import os
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_QSA_CP_TESTS") != "1", reason="explicit two-GPU QSA CP opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def cp_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tensor_random
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    assert int(os.environ.get("WORLD_SIZE", "0")) == 2
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.025, 3 * 1024**3 / total))
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
    original_backward = tensor_random.CheckpointFunction.backward
    try:
        headroom = torch.tensor(int(free >= 8 * 1024**3), device=device)
        torch.distributed.all_reduce(headroom, op=torch.distributed.ReduceOp.MIN)
        if not headroom.item():
            pytest.skip("Every GPU needs 8 GiB headroom; never evict another job")
        parallel_state.initialize_model_parallel(context_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        apply_patch_megatron_recomputation_backward()
        yield
    finally:
        tensor_random.CheckpointFunction.backward = staticmethod(original_backward)
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_cp_gather_sums_remote_key_gradients():
    from megatron.core import parallel_state

    from verl.models.mcore.qwen3_8_next.ops.context_parallel import PackedContextParallelLayout

    group = parallel_state.get_context_parallel_group()
    layout = PackedContextParallelLayout(torch.tensor([0, 12, 32], device="cuda"), 2, group.rank())
    full = torch.arange(32 * 3, device="cuda", dtype=torch.float32).reshape(32, 3)
    local = layout.local(full).requires_grad_()
    actual = layout.gather(local, group)
    torch.testing.assert_close(actual, full, rtol=0, atol=0)
    weights = full + 1
    (actual * weights * (group.rank() + 1)).sum().backward()
    torch.testing.assert_close(local.grad, layout.local(weights * 3), rtol=0, atol=0)
    # A deliberate cancellation case catches rounding each rank's partial
    # gradient before reduction: BF16(1000.5) + BF16(-1000) would lose 0.5.
    low_precision = layout.local(full.to(torch.bfloat16)).requires_grad_()
    high_precision_edge = layout.gather(low_precision, group, fp32_output=True)
    assert high_precision_edge.dtype == torch.float32
    coefficient = 1000.5 if group.rank() == 0 else -1000.0
    (high_precision_edge * coefficient).sum().backward()
    torch.testing.assert_close(low_precision.grad, torch.full_like(low_precision, 0.5), rtol=0, atol=0)
    print(f"QWEN38_CP_GATHER_BACKWARD_PASSED RANK={group.rank()}")


def test_packed_block_ids_are_not_physical_key_tiles():
    from verl.models.mcore.qwen3_8_next.ops.kernel.qsa_block_sparse_attn import (
        build_tile_index,
        selection_to_key_tile_bitmap,
    )

    # A document beginning at physical token 13 has packed block base 4.
    # Its selected block 16 covers tokens 61..64, crossing physical tile 0/1.
    sel = torch.zeros(1, 41, dtype=torch.uint8, device="cuda")
    sel[0, 16] = 1
    lo, hi, base = [torch.tensor([value], device="cuda") for value in (13, 64, 4)]
    physical = selection_to_key_tile_bitmap(sel, lo, hi, base, lo, 160, 64, 4)
    assert physical.tolist() == [[1, 1, 0]]
    new_list, new_count = build_tile_index(physical, 64, 1, 1)
    assert new_list[0, : int(new_count[0])].tolist() == [0, 1]
    old_list, old_count = build_tile_index(sel, 64, 64, 4)
    assert old_list[0, : int(old_count[0])].tolist() == [1], "Negative control must expose the omitted tile"


def test_rectangular_qsa_kernel_matches_independent_torch():
    from verl.models.mcore.qwen3_8_next.ops.kernel.qsa_block_sparse_attn import (
        qsa_block_sparse_attention_triton,
        selection_to_key_tile_bitmap,
    )
    from verl.models.mcore.qwen3_8_next.ops.qsa_indexer import PackedBlockLayout
    from verl.models.mcore.qwen3_8_next.ops.sequence import packed_token_segments

    # Non-aligned packed starts, sparse holes and KV beyond the last local
    # query index expose square-kernel bounds and packed-block/tile confusion.
    cu = torch.tensor([0, 13, 88, 160], dtype=torch.int32, device="cuda")
    _, positions = packed_token_segments(cu, 160)
    layout = PackedBlockLayout(cu, positions, 4)
    query_rows = torch.tensor([4, 7, 12, 13, 17, 35, 63, 75, 87, 88, 95, 119, 159], device="cuda")
    lo = layout.token_start[query_rows].int()
    hi = query_rows.int()
    blk_base = layout.token_block_start[query_rows].int()
    tok_base = lo.clone()
    sel = torch.zeros(query_rows.numel(), layout.num_blocks, dtype=torch.uint8, device="cuda")
    dense_mask = torch.zeros(query_rows.numel(), 160, dtype=torch.bool, device="cuda")
    for row, token in enumerate(query_rows.tolist()):
        start, base = int(lo[row]), int(blk_base[row])
        for key in range(start, token + 1):
            block = (key - start) // 4
            if (block + row) % 3 == 0 or block == (token - start) // 4:
                sel[row, base + block] = 1
                dense_mask[row, key] = True
    for tile_size in (32, 64):
        observed = selection_to_key_tile_bitmap(sel, lo, hi, blk_base, tok_base, 160, tile_size, 4)
        expected = torch.zeros_like(observed)
        rows, keys = dense_mask.nonzero(as_tuple=True)
        expected[rows, keys // tile_size] = 1
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.manual_seed(89)
    tensors = [
        (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
        for shape in ((query_rows.numel(), 4, 128), (160, 1, 128), (160, 1, 128))
    ]
    refs = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    actual = qsa_block_sparse_attention_triton(*tensors, sel, lo, hi, blk_base, tok_base, 128**-0.5, 4)
    q, k, v = refs
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float().expand(-1, 4, -1)) * 128**-0.5
    probabilities = scores.masked_fill(~dense_mask.unsqueeze(0), float("-inf")).softmax(-1)
    # Match the declared tensor-core mixed-precision contract: P and dS are
    # rounded to BF16 before their matrix products; reductions remain FP32.
    # A fully FP32 autograd reference is a different rounding computation.
    rounded_probabilities = probabilities.to(q.dtype).float()
    expected_acc = torch.einsum("hqk,khd->qhd", rounded_probabilities, v.float().expand(-1, 4, -1))
    expected = expected_acc.to(q.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=5e-4)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    delta = (grad.float() * expected_acc).sum(-1).T
    dp = torch.einsum("qhd,khd->hqk", grad.float(), v.float().expand(-1, 4, -1))
    ds = (probabilities * (dp - delta.unsqueeze(-1)) * 128**-0.5).to(q.dtype).float()
    q.grad = torch.einsum("hqk,khd->qhd", ds, k.float().expand(-1, 4, -1)).to(q.dtype)
    k.grad = torch.einsum("hqk,qhd->kd", ds, q.float()).unsqueeze(1).to(k.dtype)
    v.grad = torch.einsum("hqk,qhd->kd", rounded_probabilities, grad.float()).unsqueeze(1).to(v.dtype)
    for index, (value, ref) in enumerate(zip(tensors, refs, strict=True)):
        error = (value.grad.float() - ref.grad.float()).norm() / ref.grad.float().norm()
        print(f"QWEN38_RECTANGULAR_GRAD INDEX={index} REL_L2={error.item():.8f}", flush=True)
    for value, ref in zip(tensors, refs, strict=True):
        torch.testing.assert_close(value.grad, ref.grad, rtol=0.03, atol=2e-3)
        assert (value.grad.float() - ref.grad.float()).norm() / ref.grad.float().norm() < 0.02
    print("QWEN38_RECTANGULAR_QSA_TORCH_FORWARD_BACKWARD_PASSED Q=13 KV=160")


@pytest.mark.parametrize("budget", [8, 2048])
def test_qsa_local_queries_global_keys_match_cp1(monkeypatch, budget):
    try:
        _run_qsa_case(monkeypatch, budget)
    except Exception:
        traceback.print_exc()
        raise


def _run_qsa_case(monkeypatch, budget):
    from megatron.bridge.peft.lora import LoRA
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import checkpoint

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.ops import attention
    from verl.models.mcore.qwen3_8_next.ops.context_parallel import PackedContextParallelLayout
    from verl.models.mcore.qwen3_8_next.ops.sequence import packed_token_segments

    # Construct only the attention module, not a model with the CP guard
    # bypassed. PLE is absent from this explicit component-level test.
    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    bridge = AutoBridge.from_hf_pretrained(fixture / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    config = bridge.to_megatron_provider(load_weights=False)
    assert config.num_layers == 4 and config.hidden_size == 128
    config.context_parallel_size = 2
    config.sequence_parallel = False
    config.params_dtype = torch.bfloat16
    config.bf16 = True
    config.gradient_accumulation_fusion = True
    config.apply_rope_fusion = False
    config.attention_dropout = 0.0
    config.qwen3_8_next_indexer_budget = budget
    config.finalize()
    groups = ProcessGroupCollection.use_mpu_process_groups()
    spec = get_transformer_block_with_experimental_attention_variant_spec(config)
    submodules = copy.deepcopy(spec.layer_specs[-1].submodules.self_attention.submodules)
    submodules.linear_qkv = TEColumnParallelLinear
    module = attention.Qwen38NextAttention(config, submodules, layer_number=4, pg_collection=groups).cuda()
    ref_config = copy.copy(config)
    ref_config.context_parallel_size = 1
    ref_groups = copy.copy(groups)
    ref_groups.cp = parallel_state.get_tensor_model_parallel_group()  # singleton on each rank
    reference = attention.Qwen38NextAttention(ref_config, submodules, layer_number=4, pg_collection=ref_groups).cuda()
    peft = LoRA(target_modules=["linear_qkv", "linear_proj"], dim=16, alpha=32, dropout=0.0)
    module, reference = peft([module])[0], peft([reference])[0]
    trainable = {name: p for name, p in module.named_parameters() if p.requires_grad}
    assert len(trainable) == 4 and all("adapter" in name for name in trainable)
    with torch.no_grad():
        for name, parameter in trainable.items():
            if "linear_out" in name:
                parameter.normal_(std=0.03)
        for parameter in module.parameters():
            torch.distributed.broadcast(parameter, src=0, group=groups.cp)
    # Bridge's PEFT state_dict flattens ``to_wrap`` names for export; it is
    # not the native load_state_dict schema. Compare identical live modules
    # by copying every named parameter/buffer, not strict=False loading.
    with torch.no_grad():
        for items in ("named_parameters", "named_buffers"):
            source = dict(getattr(module, items)())
            target = dict(getattr(reference, items)())
            assert source.keys() == target.keys()
            for name, value in source.items():
                target[name].copy_(value)
    # Production LoRA accumulates GEMM weight gradients into FP32 main_grad.
    # Comparing BF16 partial weight gradients would add an extra rounding
    # boundary before CP reduction that the production path does not have.
    for component in (module, reference):
        for parameter in component.parameters():
            if parameter.requires_grad:
                parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
    module.train()
    reference.train()
    cu = torch.tensor([0, 12, 32, 76], dtype=torch.int32, device="cuda")
    layout = PackedContextParallelLayout(cu, 2, groups.cp.rank())
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=44, max_seqlen_kv=44)
    torch.manual_seed(87)
    states = torch.randn(76, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.1
    _, positions = packed_token_segments(cu, 76)
    angles = (positions[:, None].float() * torch.linspace(0.001, 0.1, 32, device="cuda")).repeat(1, 2)[:, None, None]
    loss_weights = torch.randn_like(states)
    local_shapes = []
    original_sparse = attention.qsa_block_sparse_attention_triton

    def record_shapes(q, k, v, *args):
        if q.shape[0] != k.shape[0]:
            local_shapes.append((q.shape[0], k.shape[0], v.shape[0]))
        return original_sparse(q, k, v, *args)

    monkeypatch.setattr(attention, "qsa_block_sparse_attention_triton", record_shapes)
    reference_input = states.detach().clone().requires_grad_()
    expected, _ = reference(reference_input, attention_mask=None, rotary_pos_emb=angles, packed_seq_params=packed)
    (expected.float() * loss_weights.float()).sum().backward()
    reference_grads = {
        name: p.main_grad.detach().clone() for name, p in reference.named_parameters() if p.requires_grad
    }
    for recompute in (False, True):
        module.zero_grad(set_to_none=True)
        for parameter in trainable.values():
            parameter.main_grad.zero_()
        local_input = layout.local(states).detach().requires_grad_()

        def forward(value):
            return module(value, attention_mask=None, rotary_pos_emb=layout.local(angles), packed_seq_params=packed)[0]

        actual = checkpoint(forward, False, local_input) if recompute else forward(local_input)
        gap = (actual.float() - layout.local(expected.detach()).float()).abs()
        print(
            f"QWEN38_QSA_CP2 BUDGET={budget} RECOMPUTE={recompute} "
            f"GAP_MEAN={gap.mean().item():.8f} GAP_MAX={gap.max().item():.8f}"
        )
        torch.testing.assert_close(actual, layout.local(expected), rtol=0.02, atol=5e-4)
        (actual.float() * layout.local(loss_weights).float()).sum().backward()
        torch.testing.assert_close(local_input.grad, layout.local(reference_input.grad), rtol=0.03, atol=2e-3)
        input_expected = layout.local(reference_input.grad).float()
        assert (local_input.grad.float() - input_expected).norm() / input_expected.norm() < 0.02
        for name, parameter in trainable.items():
            gradient = parameter.main_grad
            assert bool(gradient.isfinite().all()) and bool((gradient != 0).any())
            # These LoRA parameters are CP-replicated. Sum local-query losses
            # exactly once, as DDP would, before comparing to full-sequence loss.
            torch.distributed.all_reduce(gradient, group=groups.cp)
            relative_error = (gradient - reference_grads[name]).norm()
            relative_error = relative_error / reference_grads[name].float().norm()
            print(f"QWEN38_CP_GRAD NAME={name} REL_L2={relative_error.item():.8f}", flush=True)
            assert relative_error < 0.02, (name, relative_error)
        for name, parameter in trainable.items():
            torch.testing.assert_close(parameter.main_grad, reference_grads[name], rtol=0.03, atol=2e-3)
        print(f"QWEN38_QSA_CP2_BACKWARD_PASSED RANK={groups.cp.rank()} RECOMPUTE={recompute}", flush=True)
        assert module._qsa_selection is None and module._qsa_cp_layout is None
    assert local_shapes and all(shape == (38, 76, 76) for shape in local_shapes)
    print(f"QWEN38_QSA_CP2_LORA_GRADS_PASSED RANK={groups.cp.rank()} BUDGET={budget} TENSORS={len(trainable)}")
