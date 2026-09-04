# Chunked EP overlap with full recompute

This experimental Megatron backend option overlaps each MoE layer's expert
all-to-all communication with computations of adjacent token chunks. It retains
the standard Megatron full recomputation schedule and uses a separate option
from `overlap_moe_expert_parallel_comm`.

The initial implementation has CPU/Gloo numerical coverage. CUDA/TE correctness,
Nsight overlap, and end-to-end actor throughput still require validation on GPUs;
no throughput or peak-memory improvement is claimed yet.

Local validation on 2026-09-04 (Mac, PyTorch 2.11.0, Transformers 5.3.0,
TensorDict 0.10.0): **33 passed, 12 CUDA cases skipped** with:

```bash
python -m pytest -q \
  tests/utils/test_chunked_ep_overlap_on_cpu.py \
  tests/trainer/test_constants_ppo_on_cpu.py \
  tests/workers/config/test_engine_config_on_cpu.py \
  tests/special_distributed/test_chunked_ep_overlap.py
```

The passing cases include engine YAML instantiation and Ray environment setup.
Changed Python files also passed Ruff and syntax checks, and the repository's
device API check passed. Model installation in either real bridge, distributed
optimizer updates, and all CUDA timing/stream behavior still need the GPU gate.

## Configuration

```yaml
actor_rollout_ref:
  actor:
    megatron:
      tensor_model_parallel_size: 1
      expert_tensor_parallel_size: 1
      expert_model_parallel_size: 2
      chunked_ep_overlap:
        enabled: true
        num_chunks: 2
        min_tokens_per_chunk: 4096
      override_ddp_config:
        overlap_grad_reduce: false
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
        moe_token_dispatcher_type: alltoall
        overlap_moe_expert_parallel_comm: false
        overlap_dispatch_backward_with_experts_wgrad: false
        delay_wgrad_compute: false
        moe_shared_expert_overlap: false
```

`enabled` defaults to false. The chunk options belong to the verl engine, not
`override_transformer_config`. `num_chunks=1` uses the original MoE forward.
For short inputs all EP ranks fall back together; one rank cannot skip a
collective independently. The router runs once for the entire input, including
its auxiliary losses and replay, before tokens are partitioned. Shared experts
still execute once on the whole input.

The PPO Ray environment uses `CUDA_DEVICE_MAX_CONNECTIONS=8` for this option
when the launching environment is unset or has the usual Megatron value `1`.
Other explicit values are preserved. For standalone tests, benchmarks or custom
launchers, set this variable before initializing CUDA. Later explicit Ray runtime
environment overrides must also allow multiple connections.
The installer rejects an explicit value of `1` when multiple chunks are requested.

## Initial scope

- CUDA BF16, Megatron-Core 0.18.x, standard `MoELayer` and `TEGroupedMLP`.
- TP=ETP=1, EP>1, native alltoall, dropless routing; fused and unfused permutation.
- Full uniform/block recompute, or no recompute; ordinary DDP with gradient
  reduction overlap disabled. Each chunk accumulates expert gradients through
  the normal parameter hooks, so a bucket must not be reduced after its first
  chunk. Distributed optimizer remains available.
- Installation is shared by Megatron-Bridge and legacy mbridge before DDP
  wrapping. Parameters and state-dict names are retained.

Unsupported combinations are rejected: native MoE/shared-expert/wgrad overlap,
DeepEP/flex, TP/ETP>1, virtual PP, Dynamic CP, MTP, quantization, CUDA graphs,
activation offloading, Megatron-FSDP, capacity dropping/padding, latent/custom
MoE, nested selective expert recompute, LoRA, and layer-wise Muon.
CUDA static CP/PP, router replay and complete PPO training remain GPU integration
validation targets, not measured guarantees. Stages without a standard MoE layer
are currently rejected when the option is enabled.

## Correctness and performance checks

```bash
python -m pytest tests/utils/test_chunked_ep_overlap_on_cpu.py -q

CUDA_DEVICE_MAX_CONNECTIONS=8 torchrun --standalone --nproc_per_node=4 \
  -m pytest tests/special_distributed/test_chunked_ep_overlap.py -q

CUDA_DEVICE_MAX_CONNECTIONS=8 torchrun --standalone --nproc_per_node=2 \
  benchmarks/megatron/benchmark_chunked_ep_overlap.py \
  --tokens 4096 8192 16384 32768 --chunks 1 2 4
```

The CPU test compares real two-rank Gloo communication against a dense MoE
reference, including router/input/expert gradients, imbalanced input lengths,
zero-receive ranks, padding, reentrant checkpointing, and multiple live
microbatches. CPU tests do not prove CUDA stream overlap or TE main-grad behavior.

The CUDA test uses a two-layer Core TransformerBlock, the verl checkpoint
backward patch, TE experts and actual DDP gradient buffers. Four GPUs cover
EP=2 and expert DP=2. It checks uniform(1), uniform(2), block(1), both permutation
implementations, and repeated Adam updates with and without the actual Core
distributed optimizer. These CUDA cases are provided as a gate and have not yet
been executed in the local Mac environment.

The benchmark reports the maximum rank time and peak allocated/reserved memory
for a single MoE forward + recompute + backward. It excludes attention,
optimizer and PPO overhead. Compare the same hardware, shapes and options.
Use Nsight Systems and the `chunked_ep/chunk*/...` NVTX ranges to confirm an
actual A2A/GEMM intersection in both forward and backward. More chunks can
reduce GEMM efficiency and increase launch overhead.

Recompute still evaluates the complete MoE output. Delayed wgrad and eliding
recompute fc2/combine are separate future optimizations; this implementation
does not promise the memory reduction of a fused recompute/backward scheduler.

See [the implementation design](chunked_ep_overlap_design.md) for dependencies
and [the VeOmni investigation](veomni_ep_overlap_investigation.md) for the other backend.
