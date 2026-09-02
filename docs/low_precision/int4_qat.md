# Integer INT4 QAT with Megatron and vLLM

Last updated: 09/02/2026

verl provides an experimental integer INT4 QAT path for Qwen3 and Qwen3.5 MoE models on Hopper GPUs. The initial scope is deliberately narrow: Megatron training, vLLM rollout, BF16 activations, and only routed-expert weights quantized to symmetric group-wise INT4.

> [!WARNING]
> This is not yet the vLLM-Ascend/910C implementation. H20 validation establishes functional Hopper compatibility, not H200-equivalent throughput.

## Development snapshot

The implementation checkpoint lives on branch `hz/feat/int4-qat-rl-vllm`. It includes the configuration contract, Megatron routed-expert fake QAT hook, CPU and Triton quantize/pack paths, Qwen3/Qwen3.5 expert export mapping, vLLM 0.24 layerwise reload integration, single-node four- and eight-GPU recipes, and focused tests.

On H20, the final focused CPU/Triton/Transformer Engine/vLLM suite completed with 43 tests passing. A full Qwen3-30B-A3B vLLM dummy model selected `CompressedTensorsWNA16MarlinMoEMethod` with the `MARLIN` backend and generated tokens. A four-GPU TP2/EP2 GRPO run completed three optimizer steps, including repeated reloads, nonzero rewards and gradients, finite log-probability/KL metrics, and a clean exit. This is functional validation, not an H200 throughput or long-run quality claim.

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

vLLM starts with `load_format=dummy` and a compressed-tensors WNA16 scheme. vLLM 0.24's native layerwise reload restores checkpoint-facing parameters before bucketed transfer, runs hardware-specific WNA16 repacking afterward, and copies results into the original kernel tensor storage to preserve CUDA Graph addresses. WNA16 shape, group-index, and sorted-index tensors describe static model geometry and are not read by the current MoE WNA16 post-processing path. verl keeps all six resident, excludes them from the per-layer completion count, and omits per-expert `weight_shape` from online updates. This allows each `RoutedExperts` layer to finalize after its packed weights and scales arrive instead of retaining temporary buffers for every layer or dispatching 18,432 redundant metadata tensors per Qwen3-30B refresh.

Layerwise reload retains input tensors until a parent layer can be processed; attention inputs are deliberately retained until finalization. The colocated transport reuses one IPC bucket after the receiver acknowledges each callback, so retaining a raw bucket view corrupts a layer split across buckets. After each bucket is loaded, the INT4 receiver identifies the views still retained in vLLM's layerwise state and clones only those views before the acknowledgement; completed layers keep the original zero-copy IPC path. Older vLLM versions without that introspection API conservatively clone the full bucket. Without this lifetime boundary, Qwen3-30B-A3B showed invalid scales specifically in the two layers split by 4 GiB bucket boundaries and produced NaN rollout log probabilities. The Qwen3.5 adapter expands fused tensors to per-expert loader calls only inside the rollout worker, keeping the wire representation compact.

### Reload-performance interpretation

The [MILES/SGLang INT4 reference](https://miles.radixark.com/docs/advanced/int4-qat) follows the same fundamental lifecycle: update standard packed INT4 tensors, then restore the checkpoint-facing shapes and invoke backend-specific Marlin repacking before rollout. Its CUDA IPC/P2P paths eliminate avoidable framework-to-framework transfer copies, but cannot eliminate the backend layout conversion. vLLM's WNA16 reload likewise has two independent costs: receiving/loading the streamed tensors and `finalize_layerwise_reload`, which invokes the quantization method's post-load processing and preserves CUDA-graph tensor addresses.

The worker therefore logs end-to-end `reload receive/load`, local vLLM `load_weights`, ownership-copy time, input/retained bytes, and `WNA16 layerwise reload finalized` separately. The sender additionally reports iterator, copy-enqueue, GPU-sync, receiver-ack, cleanup, tensor, byte, and bucket totals. Set `VERL_INT4_QAT_RELOAD_PROFILE=1` for an H20 benchmark: vLLM workers normally suppress INFO messages, so this switch emits only these profiling records at WARNING level. In the controlled Qwen3-30B 4×H20 diagnostic baseline, each rollout rank received 55,731 tensors / 16.79 GiB. The post-step sender spent about 8.35s producing tensors and 7.80s sending metadata/waiting for the receiver, while GPU synchronization was about 0.001s. Omitting resident `weight_shape` metadata reduced the stream to 37,299 tensors with unchanged payload bytes; post-step sender time improved by 11.9%, receiver receive/load by roughly 12%, and end-to-end `update_weights` from 21.018s to 18.834s (-10.4%) in the matched diagnostic run.

`VERL_INT4_QAT_RELOAD_DIAGNOSTICS=1` validates every exported scale but is intentionally not a benchmark setting: the original per-tensor checks raised iterator time from about 3.11s to 8.35s. Diagnostics now aggregate device scalars and synchronize once at the end. With diagnostics disabled, the old metadata stream took 15.775s for one post-step update. The generic shape-elision path completed three post-step updates in 13.613/13.828/13.389s (mean 13.610s, population standard deviation 0.179s), 13.7% below that old single-sample baseline. Its six sender measurements averaged 10.812s: 2.376s iterator, 0.663s copy enqueue, 6.763s receiver acknowledgement, and 0.526s cleanup. Receiver receive/load was 9.717–9.886s, local vLLM loading was 6.172–6.434s, and WNA16 finalization was 0.002s. The next portable target is therefore producer/receiver pipelining or a generic vLLM/compressed-tensors loader index; model-class or parameter-name-specific fast paths are deliberately excluded. An eight-group-per-program pack kernel remains bit-exact for both individual Qwen3 and fused Qwen3.5 expert tensors, but its isolated controlled post-step delta was only -0.7%. A direct trainer-to-Marlin packet format is outside this first vLLM path because it would couple Megatron export to vLLM's private packed layout.

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

Qwen3-30B-A3B has hidden size 2048 and MoE intermediate size 768. Its recipe uses MoE TP=1/EP=4, so group size 128 divides both full expert dimensions and is the default for this verl implementation. MILES' current Qwen3-30B raw-mode recipe uses fake-QAT group size 128 but documents that its direct checkpoint converter defaults to group size 32; the two are independently configured there. Here trainer fake-QAT and online vLLM export intentionally both use 128 to make the simulated and real grouping match. Qwen3.5-35B-A3B uses `moe_intermediate_size=512`; with rollout MoE TP=8, each `down_proj` shard has an input dimension of 64, so group size 128 would cross TP shard boundaries and vLLM correctly rejects it. Use group size 32 for that topology instead.

For convergence comparison, use DAPO-Math-17k for training, AIME-2024 during training, and AIME-2024/AIME-2025/MATH-500 for final evaluation. The reproducible H20 asset bootstrap is [`examples/qat/prepare_qwen3_5_35b_formal_assets.sh`](../../examples/qat/prepare_qwen3_5_35b_formal_assets.sh): it fetches the Qwen3.5-35B-A3B post-trained BF16 checkpoint plus the DAPO/AIME Parquets and verifies the existing 2,101/601 Geo3K split. Its default downloader is ModelScope because the first H20 container could not connect to `huggingface.co:443`; when using Hugging Face instead, set `ASSET_DOWNLOADER=hf MODEL_REVISION=main` after a connectivity check. The H20 preparation has completed and `--verify-only` confirmed the model, Geo3K 2,101/601, DAPO 1,791,700-row and AIME-2024 960-row assets. Keep AIME-2025 and MATH-500 out of the training mixture so that they remain held-out final metrics. Run these four matched experiments:

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

A matched three-step BF16 control also completed with finite metrics. Its weight updates took 5.75–7.27 seconds, so the current INT4 implementation is a correctness baseline rather than a synchronization-performance result. A later fixed-512-token profile attributes the dominant gap to training-side online quantization/export plus IPC arrival: ownership copies and WNA16 repacking are necessary but measured as minor costs in this topology. Three GSM8K steps are not evidence of convergence parity.

## Current boundaries

- Supported model/export layout: Qwen3 and Qwen3.5 fused or individual routed experts.
- Supported training backend: Megatron with TE `GroupedLinear` experts.
- Supported rollout backend: vLLM 0.24 compressed-tensors WNA16.
- Quantized tensors: routed expert gate/up/down weights only. Router, attention, GDN, shared experts, embeddings, norms, and LM head remain BF16.
- Supported quantizer: symmetric INT4 with vLLM-compatible group size 32, 64, or 128. The Qwen3-30B recipe uses 128 to match MILES; the Qwen3.5 TP-sharded recipe uses 32. There is no zero-point or activation quantization.
- The initial implementation targets full-parameter RL. LoRA, MTP drafter sync, dense INT4, SGLang, NPU, and vLLM-Ascend require separate validation or adapters.

For Blackwell-only maximum performance, also compare against verl's existing [NVFP4 QAT](nvfp4_qat.md). Integer INT4 is the H/B portability path; NVFP4 is the native Blackwell-oriented path.
