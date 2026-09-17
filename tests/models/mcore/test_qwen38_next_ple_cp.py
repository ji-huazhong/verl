# SPDX-License-Identifier: Apache-2.0
"""Opt-in real packed PLE CP communication and gradient gates.

RUN_QWEN38_PLE_CP_TESTS=1 torchrun --standalone --nproc-per-node=2
-m pytest -s -q <this file>. No full-model CP support is claimed by this gate.
"""

import copy
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_PLE_CP_TESTS") != "1", reason="explicit multi-GPU PLE CP opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def cp_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tensor_random
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    size = int(os.environ.get("WORLD_SIZE", "0"))
    assert size in (2, 4)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.025, 3 * 1024**3 / total))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
    original_backward = tensor_random.CheckpointFunction.backward
    try:
        headroom = torch.tensor(int(free >= 8 * 1024**3), device=device)
        torch.distributed.all_reduce(headroom, op=torch.distributed.ReduceOp.MIN)
        if not headroom.item():
            pytest.skip("Every GPU needs 8 GiB headroom; never evict another job")
        parallel_state.initialize_model_parallel(context_parallel_size=size)
        model_parallel_cuda_manual_seed(123)
        apply_patch_megatron_recomputation_backward()
        yield
    finally:
        tensor_random.CheckpointFunction.backward = staticmethod(original_backward)
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _layout():
    from megatron.core import parallel_state

    from verl.models.mcore.qwen3_8_next.ops.context_parallel import PackedContextParallelLayout

    group = parallel_state.get_context_parallel_group()
    # Empty documents, a chunk shorter than the halo, and nonuniform lengths.
    bounds = [0, 0, 4, 16, 60, 60] if group.size() == 2 else [0, 0, 8, 32, 96, 96]
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    return PackedContextParallelLayout(cu, group.size(), group.rank()), group


@pytest.mark.parametrize("depth", [0, 3, 9, 32])
def test_halo_real_alltoall_matches_global_rows_and_scatter_gradient(depth):
    from verl.models.mcore.qwen3_8_next.ops.ple_context_parallel import PackedCPHalo

    layout, group = _layout()
    plan = PackedCPHalo(layout, depth)
    global_values = torch.arange(layout.total * 5, device="cuda", dtype=torch.float32).reshape(-1, 5)
    local = layout.local(global_values).detach().requires_grad_()
    actual = plan.apply(local, group)
    torch.testing.assert_close(actual, global_values[plan.global_indices], rtol=0, atol=0)
    weights = torch.arange(actual.numel(), device="cuda", dtype=torch.float32).reshape_as(actual) % 11
    (actual * weights).sum().backward()
    global_grad = torch.zeros_like(global_values)
    global_grad.index_add_(0, plan.global_indices, weights)
    torch.distributed.all_reduce(global_grad, group=group)
    torch.testing.assert_close(local.grad, layout.local(global_grad), rtol=0, atol=0)
    assert sum(plan.recv_splits) <= 2 * (layout.cu_seqlens.numel() - 1) * depth
    print(f"QWEN38_PLE_HALO_EXACT CP={group.size()} RANK={group.rank()} DEPTH={depth}", flush=True)


def test_ngram_contexts_and_hashes_match_document_and_eos_reference():
    from verl.models.mcore.qwen3_8_next.ops.ple import (
        build_ngram_contexts_cp,
        build_ngram_contexts_packed,
        ngram_hash_ids,
    )

    layout, group = _layout()
    tokens = torch.arange(layout.total, device="cuda") % 91 + 3
    tokens[torch.arange(2, layout.total, 7, device="cuda")] = 99
    actual = build_ngram_contexts_cp(layout.local(tokens), layout.cu_seqlens, 3, 99, group)
    full_contexts = build_ngram_contexts_packed(tokens, layout.cu_seqlens, 3, 99)
    torch.testing.assert_close(actual, layout.local(full_contexts), rtol=0, atol=0)
    multipliers = torch.tensor([3, 5, 7], device="cuda")
    sizes = torch.tensor([31, 37, 41, 43], device="cuda")
    offsets = torch.tensor([0, 31, 68, 109], device="cuda")
    hashes = ngram_hash_ids(actual, multipliers, sizes, offsets, 3, 2, 99)
    expected = []
    for lo, hi in zip(layout.cu_seqlens.tolist(), layout.cu_seqlens.tolist()[1:], strict=False):
        context = [99, 99]
        for token in tokens[lo:hi].tolist():
            bigram = token * 3 ^ context[-1] * 5
            trigram = bigram ^ context[-2] * 7
            expected.append([bigram % 31, bigram % 37 + 31, trigram % 41 + 68, trigram % 43 + 109])
            context = [99, 99] if token == 99 else [context[-1], token]
    reference = torch.tensor(expected, device="cuda")
    torch.testing.assert_close(hashes, layout.local(reference), rtol=0, atol=0)
    naive = build_ngram_contexts_packed(layout.local(tokens), None, 3, 99)
    wrong = (ngram_hash_ids(naive, multipliers, sizes, offsets, 3, 2, 99) != hashes).sum()
    torch.distributed.all_reduce(wrong, group=group)
    assert wrong.item() > 0, "Negative control must catch treating zigzag chunks as adjacent tokens"


def test_ple_halo_rejects_short_or_mismatched_metadata():
    from verl.models.mcore.qwen3_8_next.ops.kernel.ple_triton import ple_gate_conv_triton
    from verl.models.mcore.qwen3_8_next.ops.ple_context_parallel import PackedCPHalo

    layout, group = _layout()
    t, n, c = layout.local_indices.numel(), 2, 128
    shapes = [(t, n * c), (t, n * c), (t, c), (n * c,), (n * c,), (n * c,), (n * c, 1, 4)]
    values = [torch.zeros(shape, device="cuda") for shape in shapes]
    with pytest.raises(ValueError, match="shorter"):
        ple_gate_conv_triton(*values, n, 1e-6, 3, layout.cu_seqlens, halo=PackedCPHalo(layout, 8), cp_group=group)
    with pytest.raises(ValueError, match="boundaries"):
        ple_gate_conv_triton(*values, n, 1e-6, 3, layout.cu_seqlens + 1, halo=PackedCPHalo(layout, 9), cp_group=group)
    with pytest.raises(ValueError, match="local token"):
        ple_gate_conv_triton(
            values[0][1:], *values[1:], n, 1e-6, 3, layout.cu_seqlens, halo=PackedCPHalo(layout, 9), cp_group=group
        )


def _torch_ple(inputs, n, eps, dilation, bounds):
    query, key, value, wk, wq, wc, conv_w = inputs
    t, wide = query.shape
    c = wide // n

    def norm(x, weight):
        x = x.float().reshape(t, n, c)
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1 + weight.float().reshape(n, c))

    score = (norm(key, wk) * norm(query, wq)).sum(-1) / c**0.5
    gate = (torch.where(score >= 0, 1.0, -1.0) * score.abs().clamp_min(1e-6).sqrt()).sigmoid()
    gated = (gate[..., None] * value.float()[:, None, :]).reshape(t, wide)
    normed = norm(gated, wc).reshape(t, wide)
    outputs = []
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        rows = normed[lo:hi]
        conv = torch.zeros_like(rows)
        for j in range(conv_w.shape[-1]):
            shift = (conv_w.shape[-1] - 1 - j) * dilation
            if shift < hi - lo:
                contribution = rows[: hi - lo - shift] * conv_w[:, 0, j].float()
                conv = conv + F.pad(contribution, (0, 0, shift, 0))
        outputs.append(gated[lo:hi] + F.silu(conv))
    return torch.cat(outputs).to(query.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dilation", [1, 3])
def test_ple_kernel_local_gates_halos_and_recompute_match_cp1_and_torch(dtype, dilation):
    from megatron.core.tensor_parallel.random import checkpoint

    from verl.models.mcore.qwen3_8_next.ops.kernel.ple_triton import ple_gate_conv_triton
    from verl.models.mcore.qwen3_8_next.ops.ple_context_parallel import PackedCPHalo

    layout, group = _layout()
    plan = PackedCPHalo(layout, 3 * dilation)
    torch.manual_seed(37)
    n, c, t = 2, 128, layout.total
    shapes = [(t, n * c), (t, n * c), (t, c), (n * c,), (n * c,), (n * c,), (n * c, 1, 4)]
    # Replicated norm/conv parameters use FP32 in this derivative oracle to
    # isolate communication from rounding partial BF16 weight gradients.
    values = [
        torch.randn(shape, device="cuda", dtype=dtype if i < 3 else torch.float32) * 0.1
        for i, shape in enumerate(shapes)
    ]
    ref = [v.detach().clone().requires_grad_() for v in values]
    independent = [v.detach().clone().requires_grad_() for v in values]
    expected = ple_gate_conv_triton(*ref, n, 1e-6, dilation, layout.cu_seqlens)
    torch_expected = _torch_ple(independent, n, 1e-6, dilation, layout.cu_seqlens.tolist())
    tolerance = dict(rtol=0.03, atol=2e-3) if dtype == torch.bfloat16 else dict(rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(expected, torch_expected, **tolerance)
    weights = torch.randn_like(expected) * 0.1
    (expected.float() * weights.float()).sum().backward()
    (torch_expected.float() * weights.float()).sum().backward()
    for index, (actual, oracle) in enumerate(zip(ref, independent, strict=True)):
        torch.testing.assert_close(
            actual.grad, oracle.grad, **tolerance, msg=lambda s, i=index: f"torch oracle {i}: {s}"
        )
    for recompute in (False, True):
        local = [(layout.local(v) if i < 3 else v).detach().clone().requires_grad_() for i, v in enumerate(values)]

        def forward(*inputs):
            return ple_gate_conv_triton(*inputs, n, 1e-6, dilation, layout.cu_seqlens, halo=plan, cp_group=group)

        output = checkpoint(forward, False, *local) if recompute else forward(*local)
        torch.testing.assert_close(output, layout.local(expected), **tolerance)
        (output.float() * layout.local(weights).float()).sum().backward()
        errors = []
        for index, (actual, oracle) in enumerate(zip(local, ref, strict=True)):
            reference = layout.local(oracle.grad) if index < 3 else oracle.grad
            if index >= 3:
                torch.distributed.all_reduce(actual.grad, group=group)
            torch.testing.assert_close(
                actual.grad, reference, **tolerance, msg=lambda s, i=index: f"CP gradient {i}: {s}"
            )
            error = (actual.grad.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
            errors.append(error.item())
            assert error < 0.02
        gap = (output.float() - layout.local(expected).float()).abs().max().item()
        print(
            f"QWEN38_PLE_CP_KERNEL CP={group.size()} RANK={group.rank()} DTYPE={dtype} "
            f"DILATION={dilation} RECOMPUTE={recompute} GAP_MAX={gap:.9f} MAX_GRAD_REL_L2={max(errors):.9f}",
            flush=True,
        )


def test_frozen_ple_hc_context_hook_and_two_microbatch_recompute_fifo():
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import checkpoint

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.hyper_connection import Qwen38NextPLEHyperConnection
    from verl.models.mcore.qwen3_8_next.ops.context_parallel import PackedContextParallelLayout
    from verl.models.mcore.qwen3_8_next.ops.ple import clear_ple_batch, current_ple_batch, publish_ple_batch
    from verl.models.mcore.qwen3_8_next.provider import install_ple_context_hooks

    group = parallel_state.get_context_parallel_group()
    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    bridge = AutoBridge.from_hf_pretrained(fixture / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    config = bridge.to_megatron_provider(load_weights=False)
    assert config.num_layers == 4 and config.hidden_size == 128
    config.context_parallel_size = group.size()
    config.sequence_parallel = False
    config.params_dtype = torch.bfloat16
    config.bf16 = True
    config.gradient_accumulation_fusion = False
    config.finalize()
    ref_config = copy.copy(config)
    ref_config.context_parallel_size = 1

    class Component(torch.nn.Module):
        def __init__(self, provider):
            super().__init__()
            self.block = Qwen38NextPLEHyperConnection(provider, provider.qwen3_8_next_ple_layer_ids[0] + 1)
            self.recompute = False

        def forward(self, input_ids, hidden_states, packed_seq_params):
            def block_forward(value):
                return self.block(value)[0]

            return checkpoint(block_forward, False, hidden_states) if self.recompute else block_forward(hidden_states)

    # Explicit HC/PLE components, not construction of a model with its CP
    # guard bypassed. The real recipe freezes PLE/HC; upstream LoRA requires
    # their input gradients, not gradients of the immutable embedding table.
    module, reference = Component(config).cuda(), Component(ref_config).cuda()
    torch.manual_seed(79)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.05)
            torch.distributed.broadcast(parameter, src=0, group=group)
        for method in ("named_parameters", "named_buffers"):
            actual, target = dict(getattr(module, method)()), dict(getattr(reference, method)())
            assert actual.keys() == target.keys()
            for name, value in actual.items():
                target[name].copy_(value)
    module.requires_grad_(False)
    reference.requires_grad_(False)
    install_ple_context_hooks(module)
    install_ple_context_hooks(reference)
    layouts = [_layout()[0]]
    bounds = [0, 8, 36] if group.size() == 2 else [0, 16, 40]
    layouts.append(PackedContextParallelLayout(torch.tensor(bounds, device="cuda"), group.size(), group.rank()))
    batches = []
    for index, layout in enumerate(layouts):
        tokens = (torch.arange(layout.total, device="cuda") + 11 * index) % 101 + 3
        tokens[2::7] = config.qwen3_8_next_eos_token_id
        states = torch.randn(layout.total, 1, 2 * 128, dtype=torch.bfloat16, device="cuda") * 0.1
        weights = torch.randn(layout.total, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.1
        cu = layout.cu_seqlens
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
        full_input = states.detach().requires_grad_()
        expected = reference(input_ids=tokens[None], hidden_states=full_input, packed_seq_params=packed)
        (expected.float() * weights.float()).sum().backward()
        batches.append((layout, tokens, states, weights, packed, expected.detach(), full_input.grad.clone()))
    for recompute in (False, True):
        module.recompute = recompute
        pending = []
        for layout, tokens, states, weights, packed, expected, gradient in batches:
            local_input = layout.local(states).detach().requires_grad_()
            output = module(input_ids=layout.local(tokens)[None], hidden_states=local_input, packed_seq_params=packed)
            torch.testing.assert_close(output, layout.local(expected), rtol=0.02, atol=5e-4)
            with pytest.raises(RuntimeError, match="no n-gram ids published"):
                current_ple_batch()
            pending.append((layout, local_input, output, weights, gradient))
        assert len(module.block._ple_recompute_fifo) == (2 if recompute else 0)
        # Deliberately publish unusable later-batch metadata. Recompute MUST
        # consume this layer/chunk's queued ids instead of this side channel.
        publish_ple_batch(torch.zeros(1, 1, dtype=torch.long, device="cuda"), None)
        try:
            for index, (layout, local_input, output, weights, gradient) in enumerate(pending):
                (output.float() * layout.local(weights).float()).sum().backward()
                expected_grad = layout.local(gradient)
                torch.testing.assert_close(local_input.grad, expected_grad, rtol=0.03, atol=2e-3)
                error = (local_input.grad.float() - expected_grad.float()).norm() / expected_grad.float().norm()
                assert error < 0.02 and bool((local_input.grad != 0).any())
                assert len(module.block._ple_recompute_fifo) == (1 - index if recompute else 0)
                print(
                    f"QWEN38_PLE_HC_FIFO CP={group.size()} RANK={group.rank()} RECOMPUTE={recompute} "
                    f"MICROBATCH={index} INPUT_GRAD_REL_L2={error.item():.9f}",
                    flush=True,
                )
        finally:
            clear_ple_batch()
    assert not module.block._ple_recompute_fifo
    with pytest.raises(ValueError, match="packed physical"):
        module(input_ids=layout.local(tokens)[None], hidden_states=local_input, packed_seq_params=None)
    bad_packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=layout.cu_seqlens,
        cu_seqlens_q_padded=layout.cu_seqlens + 1,
        cu_seqlens_kv=layout.cu_seqlens,
    )
    with pytest.raises(NotImplementedError, match="padding gaps"):
        module(input_ids=layout.local(tokens)[None], hidden_states=local_input, packed_seq_params=bad_packed)
    with pytest.raises(RuntimeError, match="no n-gram ids published"):
        current_ple_batch()
