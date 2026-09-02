# Integer INT4 QAT with Megatron and vLLM

Last updated: 09/02/2026

verl provides an experimental integer INT4 QAT path for Qwen3 and Qwen3.5 MoE models on Hopper GPUs. The initial scope is deliberately narrow: Megatron training, vLLM rollout, BF16 activations, and only routed-expert weights quantized to symmetric group-wise INT4.

> [!WARNING]
> This is not yet the vLLM-Ascend/910C implementation. H20 validation establishes functional Hopper compatibility, not H200-equivalent throughput.

## Development snapshot

The implementation checkpoint lives on branch `hz/feat/int4-qat-rl-vllm`. It includes the configuration contract, Megatron routed-expert fake QAT hook, CPU and Triton quantize/pack paths, Qwen3/Qwen3.5 expert export mapping, vLLM 0.24 layerwise reload integration, single-node four- and eight-GPU recipes, and focused tests.

On H20, the final focused CPU/Triton/Transformer Engine/vLLM suite completed with 40 tests passing. A full Qwen3-30B-A3B vLLM dummy model selected `CompressedTensorsWNA16MarlinMoEMethod` with the `MARLIN` backend and generated tokens. A four-GPU TP2/EP2 GRPO run completed three optimizer steps, including repeated reloads, nonzero rewards and gradients, finite log-probability/KL metrics, and a clean exit. This is functional validation, not an H200 throughput or long-run quality claim.

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

During every actor-to-rollout update, the exporter recomputes the same stored scale and emits vLLM compressed-tensors expert weights. Qwen3 uses individual experts; Qwen3.5 may use compact fused tensors:

```text
experts.gate_up_proj.weight_packed
experts.gate_up_proj.weight_scale
experts.down_proj.weight_packed
experts.down_proj.weight_scale
```

Signed levels `[-7, 7]` are biased by eight and eight nibbles are packed, in order, into one INT32. Quantization therefore happens in the verl trainer/exporter; vLLM does not receive BF16 expert weights and PTQ them.

vLLM starts with `load_format=dummy` and a compressed-tensors WNA16 scheme. vLLM 0.24's native layerwise reload restores checkpoint-facing parameters before bucketed transfer, runs hardware-specific WNA16 repacking afterward, and copies results into the original kernel tensor storage to preserve CUDA Graph addresses. WNA16 shape, group-index, and sorted-index tensors are derived from static model geometry: group/sort indices are absent from online actor updates, while individual-expert shape tensors can be redundantly streamed into one fused parameter. verl keeps all six resident and excludes them from the per-layer completion count. This allows each `RoutedExperts` layer to finalize after its packed weights and scales arrive instead of retaining temporary buffers for every layer.

Layerwise reload retains input tensors until a parent layer can be processed; attention inputs are deliberately retained until finalization. The colocated transport reuses one IPC bucket after the receiver acknowledges each callback, so retaining a raw bucket view corrupts a layer split across buckets. The INT4 receiver clones incoming tensors before loading so deferred processing owns stable storage. Without this lifetime boundary, Qwen3-30B-A3B showed invalid scales specifically in the two layers split by 4 GiB bucket boundaries and produced NaN rollout log probabilities. The Qwen3.5 adapter expands fused tensors to per-expert loader calls only inside the rollout worker, keeping the wire representation compact.

## Configuration

The ready-to-run entry point is:

```bash
bash examples/grpo_trainer/run_qwen3_30b_a3b_megatron_int4_qat.sh
```

It enables the following actor configuration and injects [`examples/qat/config/int4_w4a16_qwen3_moe.json`](../../examples/qat/config/int4_w4a16_qwen3_moe.json) into vLLM:

```yaml
actor_rollout_ref:
  actor:
    megatron:
      qat:
        enable: true
        format: int4
        mode: w4a16
        group_size: 128
        scope: routed_experts
        symmetric: true
        scale_dtype: bfloat16
        quantization_config_path: examples/qat/config/int4_w4a16_qwen3_moe.json
```

The trainer and JSON configuration are validated together. A mismatch in format, bit width, group size, symmetry, strategy, or activation quantization fails before vLLM launch.

## Recommended experiment

Use one node with 8×H200 141 GB or 8×H20 141 GB. The Qwen3-30B-A3B recipe uses actor/reference TP=4, PP=2, EP=4, ETP=1 and rollout TP=4, MoE TP=1, EP=4. It enables the official Megatron CPU optimizer path because `param_offload`/`optimizer_offload` alone does not prevent FusedAdam from allocating its states at the first optimizer step. When only four GPUs are available, use actor/reference TP=2, PP=1, EP=2 and rollout TP=2, EP=2 for a functional smoke test. H20 is not a substitute for H200 absolute performance numbers.

Qwen3-30B-A3B has hidden size 2048 and MoE intermediate size 768. Its recipe uses MoE TP=1/EP=4, so the MILES group size 128 divides both full expert dimensions and is the default. Qwen3.5-35B-A3B uses `moe_intermediate_size=512`; with rollout MoE TP=8, each `down_proj` shard has an input dimension of 64, so group size 128 would cross TP shard boundaries and vLLM correctly rejects it. Use group size 32 for that topology instead.

For convergence comparison, use DAPO-Math-17k for training, AIME-2024 during training, and AIME-2024/AIME-2025/MATH-500 for final evaluation. Run these four matched experiments:

| Experiment | Trainer expert forward | vLLM routed experts |
|---|---|---|
| A | BF16 | BF16 |
| B | BF16 | INT4 PTQ |
| C | INT4 fake QAT | BF16 |
| D | INT4 fake QAT | INT4 WNA16 |

Before on-policy training, force the same prompt/response tokens through trainer and rollout and record mean/P95/P99/max log-prob difference. Then run a 10–20 step DAPO 1K smoke test and at least 100 steps for the final A–D comparison. Also exercise 20, preferably 100, repeated weight reloads while checking tensor pointers, memory, logits, and CUDA Graph reuse.

## H20 validation snapshot

The four-GPU Qwen3-30B-A3B smoke used GSM8K, GRPO, batch size 8, two samples per prompt, a 1,024-token response limit, actor/reference TP2/EP2, rollout TP2 with MoE TP1/EP2, and the Megatron CPU optimizer path. The three-step Marlin run observed:

- valid exported and reloaded scales at every initial and post-step sync;
- finite rollout/training probability differences with Pearson correlation 0.9865–0.9893 and KL 0.00270–0.00361;
- reward means 0.375, 0.4375, and 0.1875;
- nonzero gradient norms 0.3504 and 0.3267 on steps two and three;
- weight-update times 41.27, 38.10, and 26.43 seconds.

A matched three-step BF16 control also completed with finite metrics. Its weight updates took 5.75–7.27 seconds, so the current INT4 implementation is a correctness baseline rather than a synchronization-performance result. The extra time includes online quantization, ownership copies for deferred reload inputs, and WNA16 repacking. Three GSM8K steps are not evidence of convergence parity.

## Current boundaries

- Supported model/export layout: Qwen3 and Qwen3.5 fused or individual routed experts.
- Supported training backend: Megatron with TE `GroupedLinear` experts.
- Supported rollout backend: vLLM 0.24 compressed-tensors WNA16.
- Quantized tensors: routed expert gate/up/down weights only. Router, attention, GDN, shared experts, embeddings, norms, and LM head remain BF16.
- Supported quantizer: symmetric INT4 with vLLM-compatible group size 32, 64, or 128. The Qwen3-30B recipe uses 128 to match MILES; the Qwen3.5 TP-sharded recipe uses 32. There is no zero-point or activation quantization.
- The initial implementation targets full-parameter RL. LoRA, MTP drafter sync, dense INT4, SGLang, NPU, and vLLM-Ascend require separate validation or adapters.

For Blackwell-only maximum performance, also compare against verl's existing [NVFP4 QAT](nvfp4_qat.md). Integer INT4 is the H/B portability path; NVFP4 is the native Blackwell-oriented path.
