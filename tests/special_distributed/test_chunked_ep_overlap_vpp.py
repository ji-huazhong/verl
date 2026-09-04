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
"""CUDA gate for ordinary interleaved 1F1B + chunked EP + full recompute.

CUDA_DEVICE_MAX_CONNECTIONS=8 torchrun --standalone --nproc_per_node=4 \
    -m pytest tests/special_distributed/test_chunked_ep_overlap_vpp.py -q

PP=2, VPP=2, EP=2. Eight GPUs additionally exercise expert DP=2.
This runs the actual Core scheduler and P2P communication, not a simulated order.
"""

import gc
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.getenv("WORLD_SIZE", "1")) < 4,
    reason="requires torchrun with at least four CUDA GPUs and Megatron-Core/TE",
)


@pytest.fixture(scope="module", autouse=True)
def parallel_groups():
    if not torch.cuda.is_available() or int(os.getenv("WORLD_SIZE", "1")) < 4:
        yield
        return
    from megatron.core import parallel_state

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    parallel_state.initialize_model_parallel(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
    )
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _build_models(method, layers, overlap_p2p, enabled):
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.transformer_config import TransformerConfig

    from verl.utils.megatron.chunked_ep_overlap import install_chunked_ep_overlap
    from verl.workers.config import ChunkedEPOverlapConfig, McoreEngineConfig

    config = TransformerConfig(
        num_layers=8,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        # PP rank 0 / VP stage 0 contains only dense layers. Other chunks have MoE.
        moe_layer_freq=[0, 0, 1, 1, 1, 1, 1, 1],
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_shared_expert_intermediate_size=128,
        moe_shared_expert_overlap=False,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        microbatch_group_size_per_vp_stage=2,
        overlap_moe_expert_parallel_comm=False,
        overlap_p2p_comm=overlap_p2p,
        batch_p2p_comm=False,
        deallocate_pipeline_outputs=True,
        pipeline_dtype=torch.bfloat16,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0,
        attention_dropout=0,
        recompute_granularity="full",
        recompute_method=method,
        recompute_num_layers=layers,
        gradient_accumulation_fusion=True,
    )
    engine = McoreEngineConfig(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        chunked_ep_overlap=ChunkedEPOverlapConfig(enabled=enabled, min_tokens_per_chunk=0),
    )
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False, use_distributed_optimizer=True, overlap_param_gather=False
    )
    model_parallel_cuda_manual_seed(71)
    models = []
    param_ids = []
    for vp_stage in range(2):
        model = (
            GPTModel(
                config,
                get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage),
                vocab_size=128,
                max_sequence_length=32,
                pre_process=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
                post_process=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
                share_embeddings_and_output_weights=False,
                vp_stage=vp_stage,
            )
            .cuda()
            .train()
        )
        models.append(model)
        param_ids.append({name: id(p) for name, p in model.named_parameters()})
    # Bridge hooks may supply all virtual chunks together, before DDP wrapping.
    assert install_chunked_ep_overlap(models, engine) is models
    for vp_stage, (model, before_params) in enumerate(zip(models, param_ids, strict=True)):
        assert {name: id(p) for name, p in model.named_parameters()} == before_params
        if enabled:
            moe_layers = [m for m in model.modules() if isinstance(m, MoELayer)]
            if parallel_state.get_pipeline_model_parallel_rank() == 0 and vp_stage == 0:
                assert not moe_layers
            else:
                assert moe_layers
                assert all(hasattr(m, "_verl_chunked_ep_runtime") for m in moe_layers)
    models = [DistributedDataParallel(config, ddp_config, model) for model in models]
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-3,
            min_lr=1e-3,
            weight_decay=0,
            bf16=True,
            clip_grad=0,
            use_distributed_optimizer=True,
        ),
        models,
    )
    config.no_sync_func = [model.no_sync for model in models]
    config.grad_scale_func = optimizer.scale_loss
    return models, optimizer


def _forward_step(data_iterator, model):
    tokens, labels = next(data_iterator)
    positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
    output = model(tokens, positions, attention_mask=None, labels=labels)

    def loss_func(output):
        loss = output.float().mean()
        return loss, {"loss": loss.detach().clone(), "output": output.detach().clone()}

    return output, loss_func


def _assert_same_results(actual, expected):
    assert len(actual) == len(expected)
    for new, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(new["output"], ref["output"], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(new["loss"], ref["loss"], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("method,layers", [("uniform", 1), ("uniform", 2), ("block", 1)])
@pytest.mark.parametrize("overlap_p2p", [False, True])
def test_vpp_full_recompute(method, layers, overlap_p2p, monkeypatch):
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import schedules

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    apply_patch_megatron_recomputation_backward()

    def unexpected_combined(*args, **kwargs):
        pytest.fail("chunked EP must use ordinary interleaved 1F1B")

    monkeypatch.setattr(schedules, "combined_1f1b_schedule_for_interleaved_pipelining", unexpected_combined)
    schedule = schedules.get_forward_backward_func()
    assert schedule is schedules.forward_backward_pipelining_with_interleaving
    baseline, ref_optimizer = _build_models(method, layers, overlap_p2p, enabled=False)
    chunked, new_optimizer = _build_models(method, layers, overlap_p2p, enabled=True)
    for ref, new in zip(baseline, chunked, strict=True):
        new.load_state_dict(ref.state_dict())
    # Optimizers already own FP32 master shards; refresh them after paired loading.
    new_optimizer.reload_model_params()
    try:
        for step in range(3):
            generator = torch.Generator().manual_seed(900 + step + 17 * parallel_state.get_data_parallel_rank())
            batches = [
                tuple(torch.randint(128, (1, 32), generator=generator).cuda() for _ in range(2)) for _ in range(8)
            ]

            def run(models, forward_only=False, batches=batches):
                return schedule(
                    forward_step_func=_forward_step,
                    data_iterator=[iter(batches) for _ in models],
                    model=models,
                    num_microbatches=len(batches),
                    seq_length=32,
                    micro_batch_size=1,
                    forward_only=forward_only,
                )

            for models, optimizer in ((baseline, ref_optimizer), (chunked, new_optimizer)):
                for model in models:
                    model.zero_grad_buffer()
                optimizer.zero_grad()
            ref_results = run(baseline)
            new_results = run(chunked)
            _assert_same_results(new_results, ref_results)
            for ref_model, new_model in zip(baseline, chunked, strict=True):
                for (name, ref), (new_name, new) in zip(
                    ref_model.named_parameters(), new_model.named_parameters(), strict=True
                ):
                    assert name == new_name
                    torch.testing.assert_close(new.main_grad, ref.main_grad, atol=3e-3, rtol=5e-2, msg=name)
            for models, optimizer in ((baseline, ref_optimizer), (chunked, new_optimizer)):
                for model in models:
                    model.finish_grad_sync()
                assert optimizer.step()[0]
            for ref_model, new_model in zip(baseline, chunked, strict=True):
                for ref, new in zip(ref_model.parameters(), new_model.parameters(), strict=True):
                    torch.testing.assert_close(new, ref, atol=2e-3, rtol=5e-2)
            # PPO also uses these model chunks for forward-only log probabilities.
            with torch.no_grad():
                _assert_same_results(run(chunked, forward_only=True), run(baseline, forward_only=True))
    finally:
        torch.cuda.synchronize()
        del baseline, chunked, ref_optimizer, new_optimizer
        gc.collect()
        torch.cuda.empty_cache()
