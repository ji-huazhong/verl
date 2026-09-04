# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CUDA gate: real Core TransformerBlock, full recompute, TE, NCCL, and DDP.

CUDA_DEVICE_MAX_CONNECTIONS=8 torchrun --standalone --nproc_per_node=4 -m pytest -q \
    tests/special_distributed/test_chunked_ep_overlap.py

With four ranks EP=2 and expert DP=2, so expert gradient reduction is exercised.
Two ranks are also accepted for a smaller smoke test. This is not a PPO benchmark.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.getenv("WORLD_SIZE", "1")) < 2,
    reason="requires torchrun with at least two CUDA GPUs and Megatron-Core/TE",
)


@pytest.fixture(scope="module", autouse=True)
def parallel_groups():
    if not torch.cuda.is_available() or int(os.getenv("WORLD_SIZE", "1")) < 2:
        yield
        return
    from megatron.core import parallel_state

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    parallel_state.initialize_model_parallel(expert_model_parallel_size=2)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("method,layers", [("uniform", 1), ("uniform", 2), ("block", 1)])
@pytest.mark.parametrize("distributed_optimizer", [False, True])
def test_core_full_recompute_and_ddp(fused, method, layers, distributed_optimizer):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_config import TransformerConfig

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward
    from verl.utils.megatron.chunked_ep_overlap import install_chunked_ep_overlap
    from verl.workers.config import ChunkedEPOverlapConfig, McoreEngineConfig

    apply_patch_megatron_recomputation_backward()
    engine = McoreEngineConfig(
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        chunked_ep_overlap=ChunkedEPOverlapConfig(enabled=True, min_tokens_per_chunk=0),
    )
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_permute_fusion=fused,
        moe_shared_expert_intermediate_size=128,
        moe_shared_expert_overlap=False,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0,
        attention_dropout=0,
        recompute_granularity="full",
        recompute_method=method,
        recompute_num_layers=layers,
        # Exercise the normal TE main_grad path through Core DDP.
        gradient_accumulation_fusion=True,
    )
    model_parallel_cuda_manual_seed(71)
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
    baseline = TransformerBlock(config, spec).cuda().train()
    chunked = TransformerBlock(config, spec).cuda().train()
    chunked.load_state_dict(baseline.state_dict())
    before_keys = set(chunked.state_dict())
    before_params = {name: id(p) for name, p in chunked.named_parameters()}
    install_chunked_ep_overlap(chunked, engine)
    install_chunked_ep_overlap(chunked, engine)  # idempotent
    assert set(chunked.state_dict()) == before_keys
    assert {name: id(p) for name, p in chunked.named_parameters()} == before_params

    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False, use_distributed_optimizer=distributed_optimizer, overlap_param_gather=False
    )
    baseline = DistributedDataParallel(config, ddp_config, baseline)
    chunked = DistributedDataParallel(config, ddp_config, chunked)
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-3,
        min_lr=1e-3,
        weight_decay=0,
        bf16=True,
        clip_grad=0,
        use_distributed_optimizer=distributed_optimizer,
    )
    ref_optimizer = get_megatron_optimizer(optimizer_config, [baseline])
    new_optimizer = get_megatron_optimizer(optimizer_config, [chunked])
    for step in range(3):
        baseline.zero_grad_buffer()
        chunked.zero_grad_buffer()
        ref_optimizer.zero_grad()
        new_optimizer.zero_grad()
        for microbatch in range(2):
            torch.manual_seed(120 + dist.get_rank() + microbatch + step * 3)
            source = torch.randn(32 + dist.get_rank() * 16, 1, 64, device="cuda", dtype=torch.bfloat16)
            x_ref = source.clone().requires_grad_()
            x_new = source.clone().requires_grad_()
            target = torch.randn_like(source)
            # The verl checkpoint patch releases input storage; never reuse the
            # checkpoint's detached input across the paired forwards.
            out_ref = baseline(hidden_states=x_ref * 1, attention_mask=None)
            (out_ref.float() * target.float()).mean().backward()
            out_new = chunked(hidden_states=x_new * 1, attention_mask=None)
            (out_new.float() * target.float()).mean().backward()
            torch.testing.assert_close(out_new, out_ref, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(x_new.grad, x_ref.grad, atol=2e-3, rtol=5e-2)
        # Compare complete local contributions before a distributed optimizer
        # reduce-scatters them; the other shards are then no longer full gradients.
        for (name_ref, p_ref), (name_new, p_new) in zip(
            baseline.named_parameters(), chunked.named_parameters(), strict=True
        ):
            assert name_ref == name_new
            torch.testing.assert_close(p_new.main_grad, p_ref.main_grad, atol=3e-3, rtol=5e-2, msg=name_ref)
        baseline.finish_grad_sync()
        chunked.finish_grad_sync()
        assert ref_optimizer.step()[0]
        assert new_optimizer.step()[0]
        for (name_ref, p_ref), (_, p_new) in zip(baseline.named_parameters(), chunked.named_parameters(), strict=True):
            torch.testing.assert_close(p_new, p_ref, atol=2e-3, rtol=5e-2, msg=name_ref)
    torch.cuda.synchronize()
    del baseline, chunked, ref_optimizer, new_optimizer
