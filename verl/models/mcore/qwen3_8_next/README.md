# Qwen3.8-Flash-Next integration (experimental)

This is an opt-in NVIDIA Megatron-Bridge provider for the real `qwen4_exp`
vision-language architecture. It is not an alias for Qwen3.5. Use
`actor_rollout_ref.model.external_lib=verl.models.mcore.qwen3_8_next.bridge`.

Baseline: Megatron-Core 0.18.0, Megatron-Bridge 0.5.2, vLLM 0.29.0,
Transformer Engine 2.16.1. The provider reuses the Qwen3.5 GDN/MoE weight
transformations and vision encoder; HC, QSA, PLE, block norms, and the output
contraction are Flash-Next-specific. No global Core monkey patch is installed.

The inspected Bridge 0.5.2 VL text constructor temporarily holds two complete
decoders. An isolated dependency patch is provided at
`examples/tuning/lora/patches/megatron_bridge_qwen_vl_decoder_peak.patch`, with
application/validation instructions alongside it. It releases the temporary
decoder while preserving its module registration slot. Do not silently patch
shared site-packages; use a private dependency copy and verify its import path.
The opt-in `test_qwen38_next_bridge_allocation.py` checks object lifetime and
parameter traversal order. This does not itself validate GPU memory or GRPO.
An additional real-shape one-layer TP4/EP4 probe reduced peak allocated memory
from 3.424 to 2.282 GiB per rank, with identical final parameter SHA-256 on all
four ranks. This is not a full-model memory measurement.

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
recompute at one/two layers per group across packed microbatches. The latest
tiny test also runs actual image patches through the vision tower, checks that
changed pixels affect logits, and verifies image-conditioned language LoRA
gradients/recompute with the vision tower frozen. Use the real Float16Module
precision contract exactly once: recasting after warmup changes lazy FP32
vision RoPE buffers. The final image/text GPU suite passed 3 tests; the new
export also passes the vLLM text/reload/decode gate (base/adapter mean logprob
gaps 0.00060168/0.00066835). These do
**not** prove full-checkpoint parity, cross-engine multimodal execution,
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
not the PEFT ParamWrapper disk format or Ray/IPC transport. The latest test also
compares six-token cached decode against teacher-forced prefill within vLLM,
for both base and adapter. This is not cross-engine parity on the generated
continuation, long-context QSA selection or vision execution.

`test_qwen38_next_ipc.py` separately covers the production `ServerAdapter`,
bucketed CUDA IPC transport, and colocate worker extension through a real Ray
actor wrapping vLLM. Export a fresh capacity-eight fixture with
`QWEN38_TINY_LORA_RANK=16`; its 78 BF16 adapter tensors total 1,610,752 bytes and
must span multiple 1 MiB buckets. Run with exactly one visible GPU,
`RUN_QWEN38_IPC_TESTS=1`, the fixture directory, and
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. Use a short `TMPDIR` for Ray sockets.
The test corrupts the live LM head, restores the base over IPC, and verifies
exact outputs and live base-parameter SHA-256. It then streams the trained
adapter, a zero-B adapter, and the trained adapter again, checking activation,
disable, stale-state replacement, exact repeatability, cache-reset/version
callbacks, and unchanged base-parameter hashes. The latest run passed in
42.93 seconds; base/adapter mean logprob gaps were 0.00060168/0.00067403.
The test explicitly tears down its own engine/process groups and Ray cluster.
This is TP1 text-only transport/loader integration, not the HTTP server, a live
Megatron actor/optimizer, sleep/wake, distributed rollout, or complete GRPO.
The hashes cover named base parameters, not every buffer or host PLE table.

The actual trainer has a separate **random-model** smoke fixture:

```bash
python tests/models/mcore/prepare_qwen38_next_trainer_fixture.py \
  --fixture /path/to/tiny-export --assets /path/to/real-checkpoint \
  --output /path/to/new-trainer-fixture

CUDA_VISIBLE_DEVICES=0 QWEN38_TRAINER_FIXTURE=/path/to/new-trainer-fixture \
  QWEN38_SMOKE_OUTPUT=/path/to/new-run QWEN38_RAY_TEMP=/tmp/q38-smoke \
  bash tests/models/mcore/run_qwen38_next_trainer_fixture.sh
```

This uses four random layers, real tokenizer/processor assets, a 248320-entry
vocabulary, and a deterministic synthetic reward (not math accuracy). It runs
the production Megatron/GRPO trainer, HTTP vLLM, TransferQueue, full recompute,
sleep/wake, optimizer, and multi-bucket adapter updates at TP1. Two bugs surfaced
only in this path: the production backward patch omitted Core's checkpoint
marker, and nested mRoPE tensors lost their semantic ragged axis in transport.
Both have generic fixes and negative/positive regressions; the GPU fixture now
also exercises the production backward patch rather than only native Core.

The first successful step saved model/optimizer/extra. With
`QWEN38_RESUME_FROM=/path/to/previous-run/checkpoints/global_step_1`, a fresh
trainer loaded adapter, optimizer and RNG, completed step 2, saved another
complete checkpoint, and exited 0. Its grad norm was 0.27648; this proves tiny
trainer execution, not yet complete state restoration: auditing that first run
found the default configuration skipped LR scheduler loading. The recipe now
explicitly enables `use_checkpoint_opt_param_scheduler`. A separate run from
scratch completed both steps; another resume loaded all four states and saved
optimizer step 2 and scheduler step 2, matching the uninterrupted counters.
These runs do not establish bitwise trajectory equivalence, effective learning,
or real-checkpoint acceptance. The combined CPU suite passed 132 tests with five
opt-in skips; the production-backward GPU fixture passed three tests.

The trainer fixture also accepts `QWEN38_SMOKE_GPUS=2` (or 8) with exactly that
many visible devices. It validates fixture dimensions and at least 10 GiB free
on every device before launch. Ray CPU slots scale as `max(16, 3 * GPUs + 8)`:
the actual resource pool reserves three CPU slots per GPU. TP2/EP2 actor/reference with TP2 vLLM completed
two real trainer steps, including multi-bucket adapter sync and full checkpoint
saves. PP2/TP1 actor/reference with TP2 vLLM also completed two steps using the
same wrapper with explicit actor/ref TP/EP=1 and PP=2 overrides. Both runs saved
optimizer/scheduler step 2. A separate PP2 resume restored adapter, optimizer,
scheduler and RNG from step 1, completed step 2 and saved both optimizer shards
at step 2 with scheduler step 2. These are random-model integration results,
not full-model learning, trajectory equivalence or performance benchmarks.
The eight-GPU trainer remains unvalidated: its initial attempt was stopped at
placement-group scheduling because the earlier fixed 16-CPU quota was too
small; the corrected 32-CPU logical placement test passed without GPU compute,
but a subsequent GPU headroom check prevented a trainer rerun. Do not count
that scheduling probe as eight-GPU training acceptance.

Non-interleaved PP now requires `variable_seq_lengths=True`: Core's existing
P2P protocol exchanges all three tensor dimensions, including HC's wider
residual dimension. Fixed-shape PP remains rejected; no shared Core patch is
needed. The opt-in `test_qwen38_next_pipeline.py` runs the actual PP2 scheduler
on three microbatches (16/13/7 tokens). It checks receive shapes of
`[tokens, 1, 256]` and compares logprobs with the unpartitioned fixture reference;
all three mean/max gaps were zero. Launch it under two torchrun workers with
`RUN_QWEN38_PIPELINE_TESTS=1` and the original `QWEN38_TINY_EXPORT_DIR`.
The tested PLE layer belongs to the first stage. This does not validate later
PLE placement, PP>2, multimodal PP, CP, VPP or their combined topology.
The final combined CPU regression passed 133 tests with six opt-in skips.

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

For TP8/EP8, export a fresh fixture with `QWEN38_TINY_TP_CAPACITY=8` before
launching eight workers. The fixture has eight experts and TP-divisible vision
and language heads, with different routed/shared FFN widths. All eight ranks
passed the 128-tensor round trip, actual verl unequal-length packing, DDP LoRA
update and 78-tensor export. The TP8-to-TP1 base mean/max logprob gap was
0.00045866/0.00207090; independent TP1 vLLM base/adapter mean gaps were
0.00063046/0.00075272. PP and CP remained one for these passing model tests.

The requested TP2/PP2/EP2/CP2/VPP2 topology has a separate opt-in gate:

```bash
RUN_QWEN38_HYBRID_TESTS=1 QWEN38_TINY_EXPORT_DIR=/path/to/tiny-fixture \
  torchrun --standalone --nproc-per-node=8 -m pytest -s -q \
  tests/models/mcore/test_qwen38_next_hybrid_parallel.py
```

ETP=1, dense DP=1, expert DP=2; VPP=2 means two chunks per physical stage.
Actual group collectives and virtual layer partition checks passed. Both model
construction cases previously failed at the provider's PP1 guard. Dynamic-shape
non-interleaved PP2 is now separately tested, but CP and VPP remain explicitly
unsupported and the requested hybrid model gate is still red. It is not marked xfail and does not bypass
the guards. No hybrid forward/backward, LoRA update or reload has run. Passing
these construction tests in future will still not prove numerical or schedule
correctness; HC P2P widths, global CP contexts and chunk/microbatch-keyed PLE
recompute state need their own integration validation.

The GPU suites enforce memory headroom and
per-process allocation caps. Never evict another job to run them.

`test_qwen38_next_ple_loading.py` exercises short reads, truncation, wrong
dtype/byte ranges and failed-reload state using synthetic files on CPU. The
direct reader validates BF16 table/I64 metadata and fills the existing pinned
buffer without another table-sized allocation. A separate opt-in real-table
test is available:

```bash
RUN_QWEN38_REAL_PLE_TESTS=1 QWEN38_MODEL_PATH=/path/to/checkpoint \
  torchrun --standalone --nproc-per-node=4 -m pytest -s -q \
  tests/models/mcore/test_qwen38_next_ple_real.py
```

All four ranks passed: all 128 real table shards (102,400,491,520 bytes) were
loaded into TP-sharded pinned host memory; 496 lookup rows per rank matched an
independent safetensors reader exactly after GPU gather/all-reduce. The rows
cover every shard's start/middle/end and n-gram-derived lookups. This is not a
full-table checksum, an all-element finiteness check, or full-model inference.
The test has a 1 GiB per-process allocator cap and checks host/GPU headroom.

The smoke recipe is `examples/tuning/lora/run_qwen38_flash_next_megatron.sh`.
Its LoRA targets include language attention/GDN and routed/shared expert
linears, not the HC architectural low-rank matrices, frozen QSA indexer, or
vision encoder. Adapter-only reload is requested; no silent full-weight merge.
The canonical stacked-expert conversion requires the active vLLM loader to
advertise matching per-expert 2D targets. Native 3D/shared-stack-only layouts
are not silently treated as equivalent; unsupported mappings fail explicitly.

## Important current boundaries

- CP1 and no VPP; non-interleaved PP requires dynamic P2P shapes, with PP2 tested
  on the random fixture. TP/EP and complete-model PP still require full-model
  validation; HC residual width is not the ordinary hidden width.
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
