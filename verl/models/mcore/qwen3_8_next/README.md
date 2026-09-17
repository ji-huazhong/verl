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
and native GDN LoRA B export at TP1/TP2. They do **not** prove actual model
construction, GPU kernel parity, adapter reload, effective GRPO learning, or
checkpoint resume. Those remain required integration gates.

Run with the baseline packages installed:

```bash
CUDA_VISIBLE_DEVICES='' QWEN38_MODEL_PATH=/path/to/checkpoint \
  python -m pytest -q tests/models/mcore/test_qwen38_next_contract.py
```

The smoke recipe is `examples/tuning/lora/run_qwen38_flash_next_megatron.sh`.
Its LoRA targets include language attention/GDN and routed/shared expert
linears, not the HC architectural low-rank matrices, frozen QSA indexer, or
vision encoder. Adapter-only reload is requested; no silent full-weight merge.

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
  must load the original base checkpoint (`load_format=auto`), not dummy
  weights.** Enforcing this at the complete actor/rollout configuration boundary
  is still a pending integration task; do not override this recipe setting.
- Host PLE loading/ownership, full-model memory, numerical parity, all adapter
  targets, vision inputs, and adapter checkpoint save/restore still need actual
  GPU validation. Do not use the current tests as a production-readiness claim.
