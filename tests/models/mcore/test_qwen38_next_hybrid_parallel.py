# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP2/PP2/EP2/CP2/VPP2 topology and provider-construction gates.

Export the four-layer tiny HF fixture first, then run this file with eight
torchrun workers, RUN_QWEN38_HYBRID_TESTS=1 and QWEN38_TINY_EXPORT_DIR set.
ETP=1; VPP=2 means two model chunks per physical pipeline stage, not two
layers per chunk. Expert groups reuse ranks rather than multiplying world size.

The topology check is not a model numerical/schedule test. The construction
gates deliberately fail while the provider rejects PP/CP/VPP: no xfail, skipped
unsupported topology, or bypass of production guards may imply model support.
Even if construction passes later, interleaved forward/backward, CP parity,
LoRA updates and cross-engine reload still need their own acceptance tests.
"""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_HYBRID_TESTS") != "1", reason="explicit eight-GPU hybrid-parallel opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def hybrid_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        pytest.fail("TP2/PP2/CP2 with dense DP1 requires exactly eight torchrun workers")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.04, 5 * 1024**3 / total))
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
    try:
        # Make the resource decision collectively; a local-only skip would
        # leave peers hanging in initialize_model_parallel or a collective.
        headroom = torch.tensor(int(free >= 8 * 1024**3), device=device)
        torch.distributed.all_reduce(headroom, op=torch.distributed.ReduceOp.MIN)
        if not headroom.item():
            pytest.skip("Every GPU needs 8 GiB free headroom; never evict another job")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            context_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            distributed_timeout_minutes=2,
        )
        model_parallel_cuda_manual_seed(123)
        yield
    finally:
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_hybrid_groups_and_collectives():
    from megatron.core import parallel_state as ps

    groups = {
        "TP": (ps.get_tensor_model_parallel_group(), 2),
        "PP": (ps.get_pipeline_model_parallel_group(), 2),
        "CP": (ps.get_context_parallel_group(), 2),
        "EP": (ps.get_expert_model_parallel_group(), 2),
        "ETP": (ps.get_expert_tensor_parallel_group(), 1),
        "DP": (ps.get_data_parallel_group(), 1),
        "DP_CP": (ps.get_data_parallel_group(with_context_parallel=True), 2),
        "EDP": (ps.get_expert_data_parallel_group(), 2),
    }
    rank = torch.distributed.get_rank()
    for name, (group, size) in groups.items():
        ranks = torch.distributed.get_process_group_ranks(group)
        assert len(ranks) == size and rank in ranks, name
        value = torch.tensor(rank + 1, device="cuda", dtype=torch.int64)
        torch.distributed.all_reduce(value, group=group)
        assert value.item() == sum(member + 1 for member in ranks), name
    assert ps.get_virtual_pipeline_model_parallel_world_size() == 2
    if rank == 0:
        print("QWEN38_HYBRID_GROUPS_PASSED TP=2 PP=2 EP=2 CP=2 VPP=2 ETP=1 DP=1 EDP=2")


@pytest.fixture(scope="module")
def hybrid_provider():
    from megatron.core.process_groups_config import ProcessGroupCollection

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge

    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    bridge = AutoBridge.from_hf_pretrained(fixture / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 2
    provider.pipeline_model_parallel_size = 2
    provider.context_parallel_size = 2
    provider.virtual_pipeline_model_parallel_size = 2
    provider.expert_model_parallel_size = 2
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = True
    provider.moe_router_load_balancing_type = "none"
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_permute_fusion = False
    provider.language_max_sequence_length = 256
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return provider


def test_hybrid_virtual_layer_partition(hybrid_provider):
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    provider = hybrid_provider
    assert provider.num_layers == 4, "Use the four-layer exported fixture, not the full checkpoint"
    partition = []
    for vp_stage in range(2):
        for pp_rank in range(2):
            count = get_num_layers_to_build(provider, vp_stage=vp_stage, pp_rank=pp_rank)
            offset = get_transformer_layer_offset(provider, vp_stage=vp_stage, pp_rank=pp_rank)
            assert count == 1
            partition.extend(range(offset, offset + count))
    assert partition == list(range(provider.num_layers))
    if torch.distributed.get_rank() == 0:
        print("QWEN38_HYBRID_LAYER_PARTITION_PASSED layers=4 chunks_per_stage=2 layers_per_chunk=1")


@pytest.mark.parametrize("vp_stage", [0, 1])
def test_hybrid_provider_build(hybrid_provider, vp_stage):
    from megatron.core import parallel_state

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    # Keep the production fail-closed validation. This is intentionally a red
    # acceptance gate today, not an assertion that unsupported execution works.
    model = hybrid_provider.provide(
        pre_process=pp_rank == 0 and vp_stage == 0,
        post_process=pp_rank == 1 and vp_stage == 1,
        vp_stage=vp_stage,
    )
    assert model is not None
