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
"""Paired single-MoE forward + full recompute + backward benchmark (not PPO).

CUDA_DEVICE_MAX_CONNECTIONS=8 torchrun --standalone --nproc_per_node=2 \
    benchmarks/megatron/benchmark_chunked_ep_overlap.py --tokens 4096 16384
"""

import argparse
import gc
import json
import os
import time

import torch
import torch.distributed as dist


def main():
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_submodules
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward
    from verl.utils.megatron.chunked_ep_overlap import install_chunked_ep_overlap
    from verl.workers.config import ChunkedEPOverlapConfig, McoreEngineConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--ffn-hidden-size", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--fused-permute", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or min(args.tokens + args.chunks) < 1:
        parser.error("steps, tokens and chunks must be positive; warmup must be nonnegative")
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        parser.error("set CUDA_DEVICE_MAX_CONNECTIONS=8 before torchrun to permit concurrent streams")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    ep_size = dist.get_world_size()
    parallel_state.initialize_model_parallel(expert_model_parallel_size=ep_size)
    apply_patch_megatron_recomputation_backward()
    try:
        for tokens in args.tokens:
            for chunks in args.chunks:
                model_parallel_cuda_manual_seed(71)
                config = TransformerConfig(
                    num_layers=1,
                    hidden_size=args.hidden_size,
                    num_attention_heads=8,
                    moe_ffn_hidden_size=args.ffn_hidden_size,
                    num_moe_experts=args.num_experts,
                    moe_router_topk=args.topk,
                    moe_grouped_gemm=True,
                    moe_permute_fusion=args.fused_permute,
                    moe_token_dispatcher_type="alltoall",
                    moe_router_load_balancing_type="none",
                    expert_model_parallel_size=ep_size,
                    expert_tensor_parallel_size=1,
                    add_bias_linear=False,
                    bf16=True,
                    params_dtype=torch.bfloat16,
                    gradient_accumulation_fusion=False,
                    recompute_granularity="full",
                    recompute_method="uniform",
                    recompute_num_layers=1,
                )
                spec = get_submodules(
                    get_gpt_layer_with_transformer_engine_submodules(
                        num_experts=args.num_experts, moe_grouped_gemm=True
                    ).mlp
                )
                layer = MoELayer(config, spec).cuda().train()
                if chunks > 1:
                    install_chunked_ep_overlap(
                        layer,
                        McoreEngineConfig(
                            expert_model_parallel_size=ep_size,
                            expert_tensor_parallel_size=1,
                            chunked_ep_overlap=ChunkedEPOverlapConfig(
                                enabled=True, num_chunks=chunks, min_tokens_per_chunk=0
                            ),
                        ),
                    )
                source = torch.randn(tokens, 1, args.hidden_size, device="cuda", dtype=torch.bfloat16)

                def step(layer=layer, source=source):
                    layer.zero_grad(set_to_none=True)
                    x = source.clone().requires_grad_()
                    output = tensor_parallel.checkpoint(lambda h: layer(h)[0], False, x * 1)
                    output.float().square().mean().backward()

                for _ in range(args.warmup):
                    step()
                layer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for _ in range(args.steps):
                    step()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000 / args.steps
                metrics = torch.tensor(
                    [elapsed, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                    device="cuda",
                    dtype=torch.float64,
                )
                dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
                if dist.get_rank() == 0:
                    print(
                        json.dumps(
                            {
                                **vars(args),
                                "tokens": tokens,
                                "chunks": chunks,
                                "ep_size": ep_size,
                                "gpu": torch.cuda.get_device_name(),
                                "torch": torch.__version__,
                                "step_ms_max_rank": metrics[0].item(),
                                "peak_allocated_bytes_max_rank": metrics[1].item(),
                                "peak_reserved_bytes_max_rank": metrics[2].item(),
                            }
                        ),
                        flush=True,
                    )
                del step, layer, source, metrics
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
