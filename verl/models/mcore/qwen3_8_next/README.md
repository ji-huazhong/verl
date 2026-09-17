# Qwen3.8-Flash-Next integration (experimental)

This is an opt-in NVIDIA Megatron-Bridge provider for the real `qwen4_exp`
vision-language architecture. It is not an alias for Qwen3.5. Use
`actor_rollout_ref.model.external_lib=verl.models.mcore.qwen3_8_next.bridge`.

Baseline: Megatron-Core 0.18.0, Megatron-Bridge 0.5.2, vLLM 0.29.0,
Transformer Engine 2.16.1. The provider reuses the Qwen3.5 GDN/MoE weight
transformations and vision encoder; HC, QSA, PLE, block norms, and the output
contraction are Flash-Next-specific. No global Core monkey patch is installed.

## Validation status

CPU tests cover config translation, public-checkpoint source-key coverage,
packed boundaries (including empty sequences), partial RoPE, PLE hook cleanup,
and native GDN LoRA B export at TP1/TP2. They also cover the sigmoid GDN output
gate (without changing the SiLU convolution/MLP activation), indexed/single-file
PLE checkpoints, and canonical stacked-expert adapter layouts. Opt-in GPU tests
cover FP32/BF16 HC
forward/backward, a four-layer random VLM's construction and complete target
mapping, zero-adapter equality, effective LoRA updates with frozen base weights,
HF adapter export, native distributed adapter checkpoint round trip, and full
recompute at one/two layers per group across packed microbatches. These do
**not** prove full-checkpoint parity, multimodal execution,
effective GRPO learning, or trainer/optimizer resume.

Run with the baseline packages installed:

```bash
CUDA_VISIBLE_DEVICES='' QWEN38_MODEL_PATH=/path/to/checkpoint \
  python -m pytest -q tests/models/mcore/test_qwen38_next_contract.py

RUN_QWEN38_GPU_TESTS=1 torchrun --standalone --nproc-per-node=1 \
  -m pytest -s -q tests/models/mcore/test_qwen38_next_gpu.py
```

For the separate vLLM text/reload gate, set `QWEN38_TINY_EXPORT_DIR` to a new
directory during the GPU test, then run `test_qwen38_next_vllm.py` in a separate
process with that directory, `RUN_QWEN38_VLLM_TESTS=1`, and
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. After fixing the GDN output gate and converting
canonical raw expert adapters through public packed-module mappings, the tiny
TP1 base/adapter mean logprob gaps are 0.00044113/0.00056800. The existing mean
< 0.005 and max < 0.05 gates pass without relaxed tolerances. This exercises the
actual verl `TensorLoRARequest` loader, activation, disabling and remove/reload,
not the PEFT ParamWrapper disk format or Ray/IPC transport. It checks short-text
prefill, not multi-token decode, long-context QSA selection or vision inputs.

`test_qwen38_next_parallel.py` consumes the same exported fixture through the
real AutoBridge HF import path. A 128-tensor exact round trip and TP1/TP2-EP2
base forward parity have passed (TP2 mean/max logprob gap 0.00033432/0.00201797).
Use `RUN_QWEN38_PARALLEL_TESTS=1` under torchrun; `QWEN38_TEST_TP/EP` default to
the process count, with ETP/PP/CP fixed to one. DDP LoRA updates with full
recompute and a 78-tensor adapter export have also passed at TP2/EP2. Loading
that adapter in independent TP1 vLLM passes (base/adapter mean logprob gaps
0.00043296/0.00057397), including disabling and remove/reload. These are tiny
component tests, not the complete trainer or IPC path. Set the optional
`QWEN38_PARALLEL_EXPORT_DIR` to a new directory to save tiny artifacts for the
independent vLLM gate. No artifacts are published automatically.

The GPU suites enforce memory headroom and
per-process allocation caps. Never evict another job to run them.

The smoke recipe is `examples/tuning/lora/run_qwen38_flash_next_megatron.sh`.
Its LoRA targets include language attention/GDN and routed/shared expert
linears, not the HC architectural low-rank matrices, frozen QSA indexer, or
vision encoder. Adapter-only reload is requested; no silent full-weight merge.
The canonical stacked-expert conversion requires the active vLLM loader to
advertise matching per-expert 2D targets. Native 3D/shared-stack-only layouts
are not silently treated as equivalent; unsupported mappings fail explicitly.

## Important current boundaries

- PP1/CP1 only; TP/EP are retained for the real-model validation target. HC
  residual width is not the ordinary hidden width used by Core's PP scheduler.
- Full-layer recompute only; no selective attention recompute, CUDA graphs,
  or activation offloading in this first implementation.
- One packed stream per microbatch; independently keyed multiple PLE layers,
  padding gaps, and interleaved PP are not implemented.
- MTP is explicitly disabled for the GRPO policy.
- The frozen PLE host table and hash metadata load directly from the original
  checkpoint and are excluded from ordinary actor weight export. **Rollout
  must load the original base checkpoint (`load_format=auto` or `safetensors`),
  not dummy weights.** The configured external module exposes a validation hook
  called by the vLLM server before engine allocation. Unsupported formats and
  conflicting engine kwargs fail explicitly. Full-table ownership across base
  load and adapter reload still requires integration validation.
- Host PLE loading/ownership, full-model memory, numerical parity, all adapter
  targets on the complete model, vision inputs, and trainer checkpoint resume
  still need integration validation. Do not use the current component tests as
  a production-readiness claim.
