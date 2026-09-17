# SPDX-License-Identifier: Apache-2.0
"""PLE sequence/context-parallel input-gradient and recompute gate.

Set RUN_QWEN38_PLE_SP_CP_TESTS=1 and QWEN38_TINY_EXPORT_DIR, then launch
two (TP2/CP1) or four (TP2/CP2) torchrun workers. This is a component gate,
not a bypass of whole-model mixed-parallel validation.
"""

import copy
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_PLE_SP_CP_TESTS") != "1", reason="explicit PLE TP/CP GPU opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def parallel_context():
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
        enough = torch.tensor(int(free >= 8 * 1024**3), device=device)
        torch.distributed.all_reduce(enough, op=torch.distributed.ReduceOp.MIN)
        if not enough.item():
            pytest.skip("Every GPU needs 8 GiB free; never evict another job")
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=size // 2)
        model_parallel_cuda_manual_seed(123)
        apply_patch_megatron_recomputation_backward()
        yield
    finally:
        tensor_random.CheckpointFunction.backward = staticmethod(original_backward)
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_ple_sp_cross_boundary_consumers_sum_input_gradients_and_recompute():
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import checkpoint

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.hyper_connection import Qwen38NextPLEHyperConnection
    from verl.models.mcore.qwen3_8_next.ops.context_parallel import PackedContextParallelLayout
    from verl.models.mcore.qwen3_8_next.ops.ple import clear_ple_batch, current_ple_batch, publish_ple_batch
    from verl.models.mcore.qwen3_8_next.provider import install_ple_context_hooks

    cp = parallel_state.get_context_parallel_group()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    bridge = AutoBridge.from_hf_pretrained(Path(os.environ["QWEN38_TINY_EXPORT_DIR"]) / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    config = bridge.to_megatron_provider(load_weights=False)
    assert config.num_layers == 4 and config.hidden_size == 128
    config.tensor_model_parallel_size = 2
    config.context_parallel_size = cp.size()
    config.sequence_parallel = True
    config.params_dtype = torch.bfloat16
    config.bf16 = True
    config.gradient_accumulation_fusion = False
    config.finalize()
    ref_config = copy.copy(config)
    ref_config.context_parallel_size = 1
    ref_config.sequence_parallel = False

    class Component(torch.nn.Module):
        def __init__(self, provider):
            super().__init__()
            self.block = Qwen38NextPLEHyperConnection(provider, provider.qwen3_8_next_ple_layer_ids[0] + 1)
            self.recompute = False

        def forward(self, input_ids, hidden_states, packed_seq_params):
            def forward_block(value):
                # Real HC's residual is post-PLE. An objective only on the
                # right SP shard isolates its convolution's left-shard gradient.
                return self.block(value)[3]

            return checkpoint(forward_block, False, hidden_states) if self.recompute else forward_block(hidden_states)

    module, reference = Component(config).cuda(), Component(ref_config).cuda()
    torch.manual_seed(79)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.1)
            torch.distributed.broadcast(parameter, src=0)
        for method in ("named_parameters", "named_buffers"):
            actual, target = dict(getattr(module, method)()), dict(getattr(reference, method)())
            assert actual.keys() == target.keys()
            for name, value in actual.items():
                target[name].copy_(value)
    module.requires_grad_(False)
    reference.requires_grad_(False)
    install_ple_context_hooks(module)
    install_ple_context_hooks(reference)
    batches = []
    wide = config.num_residual_streams * config.hidden_size
    for index, bounds in enumerate(([0, 16, 64], [0, 24, 80])):
        cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
        layout = PackedContextParallelLayout(cu, cp.size(), cp.rank())
        tokens = (torch.arange(layout.total, device="cuda") + 11 * index) % 101 + 3
        tokens[2::7] = config.qwen3_8_next_eos_token_id
        states = torch.randn(layout.total, 1, wide, dtype=torch.bfloat16, device="cuda") * 0.1
        torch.distributed.broadcast(states, src=0)
        weights = torch.zeros_like(states)
        local_size = layout.local_indices.numel() // 2
        # Every nonzero output belongs to SP rank 1. SP rank 0 must still
        # receive a nonzero convolution-halo gradient from that other consumer.
        for owner in range(cp.size()):
            other = PackedContextParallelLayout(cu, cp.size(), owner)
            row = other.local_indices[local_size]
            weights[row] = 0.1
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
        full_input = states.detach().requires_grad_()
        expected = reference(input_ids=tokens[None], hidden_states=full_input, packed_seq_params=packed)
        (expected.float() * weights.float()).sum().backward()
        batches.append((layout, tokens, states, weights, packed, expected.detach(), full_input.grad.clone()))

    def shard(layout, value):
        return layout.local(value).chunk(2, dim=0)[tp_rank].contiguous()

    for recompute in (False, True):
        module.recompute = recompute
        pending = []
        for layout, tokens, states, weights, packed, expected, gradient in batches:
            local_input = shard(layout, states).detach().requires_grad_()
            output = module(input_ids=layout.local(tokens)[None], hidden_states=local_input, packed_seq_params=packed)
            torch.testing.assert_close(output, shard(layout, expected), rtol=0.02, atol=5e-4)
            with pytest.raises(RuntimeError, match="no n-gram ids published"):
                current_ple_batch()
            pending.append((layout, local_input, output, weights, gradient))
        assert len(module.block._ple_recompute_fifo) == (2 if recompute else 0)
        publish_ple_batch(torch.zeros(1, 1, dtype=torch.long, device="cuda"), None)
        try:
            for index, (layout, local_input, output, weights, gradient) in enumerate(pending):
                local_weights = shard(layout, weights)
                if tp_rank == 0:
                    assert not bool(local_weights.any())
                (output.float() * local_weights.float()).sum().backward()
                expected_grad = shard(layout, gradient)
                assert expected_grad.float().norm() > 1e-8
                relative = (local_input.grad.float() - expected_grad.float()).norm() / expected_grad.float().norm()
                print(
                    f"QWEN38_PLE_SP_CP CP={cp.size()} TP_RANK={tp_rank} CP_RANK={cp.rank()} "
                    f"RECOMPUTE={recompute} BATCH={index} GRAD_REL_L2={relative.item():.9f}",
                    flush=True,
                )
                close = torch.allclose(local_input.grad, expected_grad, rtol=0.03, atol=2e-3)
                passed = torch.tensor(int(close and relative < 0.02), device="cuda")
                # Fail all ranks together so a numerical negative control does
                # not leave peers in the next microbatch's collective.
                torch.distributed.all_reduce(passed, op=torch.distributed.ReduceOp.MIN)
                assert passed.item(), f"PLE SP input gradient mismatch; local relative L2={relative.item()}"
                assert len(module.block._ple_recompute_fifo) == (1 - index if recompute else 0)
        finally:
            clear_ple_batch()
