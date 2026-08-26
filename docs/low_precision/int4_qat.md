# Integer INT4 QAT with Megatron and vLLM

Last updated: 08/26/2026

verl provides an experimental integer INT4 QAT path for `Qwen/Qwen3.5-35B-A3B` on Hopper GPUs. The initial scope is deliberately narrow: Megatron training, vLLM rollout, BF16 activations, and only routed-expert weights quantized to symmetric group-wise INT4.

> [!WARNING]
> This path has CPU reference tests but still requires Hopper GPU integration and end-to-end RL validation. It is not yet the vLLM-Ascend/910C implementation.

## Development snapshot

The first implementation checkpoint lives on branch `hz/feat/int4-qat-rl-vllm`. It includes the configuration contract, Megatron routed-expert fake QAT hook, CPU and Triton quantize/pack paths, Qwen3.5 expert export mapping, vLLM 0.24 layerwise reload integration, a single-node eight-GPU recipe, and CPU reference tests.

Before this checkpoint, the focused CPU suites completed with 26 passed and one CUDA-only test skipped. Python syntax compilation, both launcher syntax checks, JSON parsing, and `git diff --check` were also clean. The implementation has not yet passed a real Transformer Engine, Triton CUDA, Marlin/WNA16 reload, or end-to-end RL run on H20/H200; those validations remain release blockers rather than implied support.

## WNA16 terminology

`WNA16` is vLLM/Compressed Tensors terminology for a weight-only quantization family:

- `W` means weights are quantized.
- `N` is a variable weight bit width. This implementation sets `N=4`.
- `A16` means activations remain FP16 or BF16; this recipe uses BF16.

Consequently, integer W4A16 is one WNA16 instance. WNA16 is not a separate quantization algorithm or numeric format. This implementation specifically uses `type=int`, `num_bits=4`, symmetric per-group absmax scaling, and `pack-quantized` checkpoint tensors. NVFP4 W4A16 uses FP4 E2M1 values and a different scale/checkpoint/kernel contract, so the two paths are not interchangeable.

## End-to-end contract

The actor retains BF16 master weights, gradients, and optimizer states. Each routed-expert `GroupedLinear` forward uses:

```text
amax  = max(abs(weight_group))
scale = BF16(max(amax / 7, 1e-5))
q     = clamp(round_to_nearest_even(weight_group / scale), -7, 7)
dq    = q * scale
```

Backward uses an identity straight-through estimator. The CUDA forward uses a fused Triton QDQ kernel; the CPU implementation is the numerical reference.

During every actor-to-rollout update, the exporter recomputes the same stored scale and emits compact Qwen3.5 fused expert tensors:

```text
experts.gate_up_proj.weight_packed
experts.gate_up_proj.weight_scale
experts.down_proj.weight_packed
experts.down_proj.weight_scale
```

Signed levels `[-7, 7]` are biased by eight and eight nibbles are packed, in order, into one INT32. Quantization therefore happens in the verl trainer/exporter; vLLM does not receive BF16 expert weights and PTQ them.

vLLM starts with `load_format=dummy` and a compressed-tensors WNA16 scheme. vLLM 0.24's native layerwise reload restores checkpoint-facing parameters before bucketed transfer, runs hardware-specific WNA16 repacking afterward, and copies results into the original kernel tensor storage to preserve CUDA Graph addresses. The Qwen3.5 adapter expands fused tensors to per-expert loader calls only inside the rollout worker, keeping the IPC stream compact.

## Configuration

The ready-to-run entry point is:

```bash
bash examples/grpo_trainer/run_qwen3_5_35b_megatron_int4_qat.sh
```

It enables the following actor configuration and injects [`examples/qat/config/int4_w4a16_qwen3_5_moe.json`](../../examples/qat/config/int4_w4a16_qwen3_5_moe.json) into vLLM:

```yaml
actor_rollout_ref:
  actor:
    megatron:
      qat:
        enable: true
        format: int4
        mode: w4a16
        group_size: 32
        scope: routed_experts
        symmetric: true
        scale_dtype: bfloat16
        quantization_config_path: examples/qat/config/int4_w4a16_qwen3_5_moe.json
```

The trainer and JSON configuration are validated together. A mismatch in format, bit width, group size, symmetry, strategy, or activation quantization fails before vLLM launch.

## Recommended experiment

Use one node with 8×H200 141 GB or 8×H20 141 GB. Start from the existing Qwen3.5 topology: actor TP=2, PP=1, EP=8, ETP=1; rollout TP=8. H20 96 GB is suitable for a shortened smoke test, but not a substitute for H200 absolute performance numbers.

Qwen3.5-35B-A3B uses `moe_intermediate_size=512`. With rollout TP=8, each `down_proj` shard has an input dimension of 64, so the MILES-style group size 128 would cross TP shard boundaries and vLLM correctly rejects it. The supplied recipe uses group size 32, which divides both hidden size 2048 and the TP-sharded expert dimension 64.

For convergence comparison, use DAPO-Math-17k for training, AIME-2024 during training, and AIME-2024/AIME-2025/MATH-500 for final evaluation. Run these four matched experiments:

| Experiment | Trainer expert forward | vLLM routed experts |
|---|---|---|
| A | BF16 | BF16 |
| B | BF16 | INT4 PTQ |
| C | INT4 fake QAT | BF16 |
| D | INT4 fake QAT | INT4 WNA16 |

Before on-policy training, force the same prompt/response tokens through trainer and rollout and record mean/P95/P99/max log-prob difference. Then run a 10–20 step DAPO 1K smoke test and at least 100 steps for the final A–D comparison. Also exercise 20, preferably 100, repeated weight reloads while checking tensor pointers, memory, logits, and CUDA Graph reuse.

## Current boundaries

- Supported model/export layout: Qwen3.5 fused or individual routed experts.
- Supported training backend: Megatron with TE `GroupedLinear` experts.
- Supported rollout backend: vLLM 0.24 compressed-tensors WNA16.
- Quantized tensors: routed expert gate/up/down weights only. Router, attention, GDN, shared experts, embeddings, norms, and LM head remain BF16.
- Supported quantizer: symmetric INT4 with vLLM-compatible group size 32, 64, or 128; the supplied Qwen3.5 TP=8 recipe uses 32. There is no zero-point or activation quantization.
- The initial implementation targets full-parameter RL. LoRA, MTP drafter sync, dense INT4, SGLang, NPU, and vLLM-Ascend require separate validation or adapters.

For Blackwell-only maximum performance, also compare against verl's existing [NVFP4 QAT](nvfp4_qat.md). Integer INT4 is the H/B portability path; NVFP4 is the native Blackwell-oriented path.
