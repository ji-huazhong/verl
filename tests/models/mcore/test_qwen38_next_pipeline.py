# SPDX-License-Identifier: Apache-2.0
"""Opt-in PP2 numerical gate with real dynamic P2P, not a construction-only check.

Run under two torchrun workers with RUN_QWEN38_PIPELINE_TESTS=1 and
QWEN38_TINY_EXPORT_DIR pointing to the original four-layer GPU export.
The full trainer smoke separately covers optimizer/recompute/adapter reload.
"""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_PIPELINE_TESTS") != "1", reason="explicit two-GPU pipeline opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def pipeline_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    assert int(os.environ.get("WORLD_SIZE", "0")) == 2, "PP2 gate requires two ranks"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.025, 3 * 1024**3 / total))
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
    try:
        headroom = torch.tensor(int(free >= 8 * 1024**3), device=device)
        torch.distributed.all_reduce(headroom, op=torch.distributed.ReduceOp.MIN)
        if not headroom.item():
            pytest.skip("Every GPU needs 8 GiB headroom; never evict another job")
        parallel_state.initialize_model_parallel(pipeline_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        yield
    finally:
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_dynamic_hc_pipeline_matches_unpartitioned_logits(monkeypatch):
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.module import Float16Module
    from safetensors.torch import load_file

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge

    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    bridge = AutoBridge.from_hf_pretrained(fixture / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert provider.num_layers == 4 and provider.hidden_size == 128, "Never run this gate on the full model"
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 2
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.variable_seq_lengths = True
    provider.batch_p2p_comm = False
    provider.overlap_p2p_comm = False
    provider.moe_router_load_balancing_type = "none"
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_permute_fusion = False
    provider.language_max_sequence_length = 256
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    rank = parallel_state.get_pipeline_model_parallel_rank()
    model = provider.provide(pre_process=rank == 0, post_process=rank == 1).cuda()
    bridge.load_hf_weights([model])
    model = Float16Module(provider, model).eval()

    shapes = []
    original = P2PCommunicator._communicate_shapes

    def record_shapes(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        shapes.extend(tuple(shape) for shape in result if any(shape))
        return result

    monkeypatch.setattr(P2PCommunicator, "_communicate_shapes", record_shapes)
    reference = load_file(str(fixture / "reference.safetensors"))
    lengths = [16, 13, 7]
    assert reference["input_ids"].shape[-1] >= max(lengths)

    def forward_step(iterator, module):
        length = next(iterator)
        ids = reference["input_ids"][:, :length].cuda()
        positions = torch.arange(length, device="cuda").reshape(1, 1, -1).repeat(3, 1, 1)
        cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=length, max_seqlen_kv=length
        )
        output = module(input_ids=ids, position_ids=positions, attention_mask=None, packed_seq_params=packed)

        def collect(value, non_loss_data=False):
            assert non_loss_data
            return value.detach().float().cpu()

        return output, collect

    with torch.no_grad():
        outputs = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter(lengths),
            model=[model],
            num_microbatches=len(lengths),
            seq_length=max(lengths),
            micro_batch_size=1,
            forward_only=True,
            collect_non_loss_data=True,
        )
    if rank == 1:
        assert shapes == [(length, 1, 256) for length in lengths], shapes
        assert len(outputs) == len(lengths)
        for length, output in zip(lengths, outputs, strict=True):
            expected = reference["base_logits"][:, :length].float()
            assert output.shape == expected.shape
            gap = (output.log_softmax(-1) - expected.log_softmax(-1)).abs()
            assert bool(gap.isfinite().all()) and gap.mean() < 0.005 and gap.max() < 0.05
            print(f"QWEN38_PP2_LENGTH={length} GAP_MEAN={gap.mean().item():.8f} GAP_MAX={gap.max().item():.8f}")
