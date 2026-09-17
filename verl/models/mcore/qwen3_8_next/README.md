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

The trainer fixture also accepts `QWEN38_SMOKE_GPUS=2`, 4 or 8 with exactly that
many visible devices. It validates fixture dimensions and at least 10 GiB free
on every device before launch. Ray CPU slots scale as `3 * GPUs + 16`:
the actual resource pool reserves three CPU slots per GPU, and the tested
trainer/queue services already occupy nine slots before its placement group.
TP2/EP2 actor/reference with TP2 vLLM completed
two real trainer steps, including multi-bucket adapter sync and full checkpoint
saves. PP2/TP1 actor/reference with TP2 vLLM also completed two steps using the
same wrapper with explicit actor/ref TP/EP=1 and PP=2 overrides. Both runs saved
optimizer/scheduler step 2. A separate PP2 resume restored adapter, optimizer,
scheduler and RNG from step 1, completed step 2 and saved both optimizer shards
at step 2 with scheduler step 2. These are random-model integration results,
not full-model learning, trajectory equivalence or performance benchmarks.
The eight-GPU trainer subsequently completed two steps with TP8/EP8 actor/ref
and TP8/EP1 vLLM, using the 40-CPU reservation. Its 184 TensorBoard scalars were
finite; each saved checkpoint has world_size=8 and 16 optimizer fragments,
all with the matching step (1 or 2), alongside scheduler step 1/2. A 32-CPU
placement-only probe had passed but omitted the nine occupied service slots;
the complete trainer still waited with only 23 of the required 24 slots free.
Eight-GPU actual checkpoint restoration remains untested. These are random
four-layer results, not full-model execution or effective learning.

Non-interleaved PP now requires `variable_seq_lengths=True`: Core's existing
P2P protocol exchanges all three tensor dimensions, including HC's wider
residual dimension. Fixed-shape PP remains rejected; no shared Core patch is
needed. The opt-in `test_qwen38_next_pipeline.py` runs the actual PP2 scheduler
on three microbatches (16/13/7 tokens). It checks receive shapes of
`[tokens, 1, 256]` and compares logprobs with the unpartitioned fixture reference;
all three mean/max gaps were zero. Launch it under two torchrun workers with
`RUN_QWEN38_PIPELINE_TESTS=1` and the original `QWEN38_TINY_EXPORT_DIR`.
The tested PLE layer belongs to the first stage. This initial result does not
validate PP>2, multimodal PP, CP or their combined topology.
The final combined CPU regression passed 133 tests with six opt-in skips.

The pipeline gate now also accepts `QWEN38_PIPELINE_VPP=2` and defaults to
overlap P2P, matching the production engine. It constructs two real chunks per
rank, checks global layer identities and HC receive shapes, and compares four
variable-length microbatches (16/13/7/11 tokens) with the unpartitioned reference.
The initial overlap run had zero mean/max logprob gaps and passed no-recompute
versus full-recompute LoRA gradient checks (24 gradient tensors per rank).
Here the PLE layer is on physical rank 1, chunk 0, not the embedding stage.
The same production trainer completed two PP2/VPP2 steps and independently
resumed step 1 to step 2, including adapter/optimizer/scheduler/RNG loading and
checkpoint saving. Saved optimizer shards and scheduler counters were all 2.
These remain random four-layer, text-only results, not full-checkpoint acceptance.

Synchronous VPP with dynamic P2P did not complete backward in bounded tests,
with either unbatched or batched communication. Removing synchronous host
copies from the test did not resolve the wait. Its root cause is still open;
VPP therefore requires `overlap_p2p_comm=True` and rejects the unvalidated path
before model allocation. No shared Core implementation is patched.
The final overlap gate additionally checks exact receive-shape multiplicities;
both ranks passed (35.27/35.44 seconds), with the same zero reference gaps and
24 gradient tensors each. The final CPU suite passed 134 tests with six skips.

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
PP2/VPP2 is now separately tested with CP1 and overlap P2P. Standalone CP2 has
the separate evidence below, as does TP2/EP2/CP2; CP with PP/VPP remains guarded
and the requested hybrid model gate is still red. It is not marked xfail and does not bypass
the guards. No hybrid forward/backward, LoRA update or reload has run. Passing
these construction tests in future will still not prove numerical or schedule
correctness; the full combined schedule still needs its own integration
validation. The per-layer PLE FIFO is not shared across chunks.

### QSA context-parallel component gate

```bash
RUN_QWEN38_QSA_CP_TESTS=1 QWEN38_TINY_EXPORT_DIR=/path/to/tiny-export \
  torchrun --standalone --nproc-per-node=2 -m pytest -s -q \
  tests/models/mcore/test_qwen38_next_qsa_cp.py
```

The QSA component uses real packed zigzag CP2: local queries and output
projections, global projected KV, and global document-aware indexer keys.
It does not replicate the decoder's full hidden-state computation. Tensor-core
kernels now support unequal Q/KV lengths and map document-relative blocks to
physical KV tiles, including tile-boundary straddles.

Forward KV communication stays BF16. An FP32 autograd edge retains partial KV
gradients until reduce-scatter, casting only after the sum. This adds temporary
FP32 KV/gradient storage and FP32 backward communication; it is not a
memory-free optimization. The component test uses fused FP32 LoRA weight
gradient accumulation, matching the tested Bridge trainer configuration.

Two ranks each passed five tests: exact gather/backward including a cancellation
case, a packed-tile negative control, rectangular Q=13/KV=160 forward/backward
against an independent mixed-precision Torch reference, and CP2/CP1 QSA with
sparse/full-coverage budgets and full recompute. Forward gaps were zero; the
largest relative LoRA gradient L2 error was 8.59e-6. Existing elementwise
tolerances were not relaxed. This is not a full-model, DDP optimizer, hybrid
TP/PP/EP/CP/VPP, or rollout CP test. The standalone full-model CP2 gate below
provides separate evidence; combined pipeline/context schedules remain guarded.

### PLE context-parallel component gate

```bash
RUN_QWEN38_PLE_CP_TESTS=1 QWEN38_TINY_EXPORT_DIR=/path/to/tiny-export \
  torchrun --standalone --nproc-per-node=2 -m pytest -s -q \
  tests/models/mcore/test_qwen38_next_ple_cp.py
```

The same gate supports four processes. Only integer token metadata is globally
gathered for document/EOS-aware n-gram hashing. Gates and grouped normalization
remain local; normalized FP32 rows in the left convolution halo are exchanged
with variable-size all-to-all. Each peer receives a requested remote row once;
locally owned rows are copied without network traffic. Each expanded chunk is
clipped at its document boundary. Backward reverses the exchange and sums all
halo consumers into the owner in FP32 before the local norm/gate backward.
This introduces halo buffers and communication; it is not a measured speedup.

CP2 and CP4 each passed 11 tests per rank, including zero/asymmetric exchanges,
halos longer than a chunk, empty documents, independent EOS/hash calculations,
FP32/BF16 kernel forward/backward against an independent Torch formula, and
dilation 1/3 with full recompute. CP/CP1 kernel forward gaps were zero; maximum
gradient relative L2 errors were 6.14e-5 (CP2) and 5.22e-5 (CP4), with unchanged
elementwise gates. The derivative oracle uses FP32 replicated norm/conv weights
to avoid conflating communication with BF16 partial-weight-gradient rounding.
The real frozen BF16 PLE/HC component separately tests context hooks and two
different queued microbatches under recompute, deliberately replacing the live
side channel with unusable metadata. Its input-gradient relative L2 gaps were
zero in these fixtures, and queues were fully consumed. The recipe freezes
PLE/HC; this does not validate training their BF16 weights or table.

The original CP1 GPU model suite still passes three tests, including vision,
LoRA export and recompute. CP2 whole-model/optimizer/rollout and mixed TP/EP/PP/
VPP are not covered by the PLE component gate. See the separate model gate below.

### Native GDN and complete random-model CP gates

`test_qwen38_next_gdn_cp.py` uses Core's native packed context/head all-to-all,
not a replacement GDN implementation. With `RUN_QWEN38_GDN_CP_TESTS=1` and the
original tiny fixture, two/four torchrun ranks each passed two tests. Unequal
documents, input gradients, four LoRA gradients and full recompute match CP1;
forward gaps are zero and maximum gradient relative L2 errors are 1.40e-6/1.41e-6.
Initial compilation is included in elapsed test time, not measured throughput.

`test_qwen38_next_model_cp.py` compares independent CP1 and CP2 processes:

```bash
RUN_QWEN38_MODEL_CP_TESTS=1 QWEN38_TINY_EXPORT_DIR=/path/to/tiny-export \
  QWEN38_MODEL_CP_OUTPUT=/path/to/new-cp1-output \
  torchrun --standalone --nproc-per-node=1 -m pytest -s -q \
  tests/models/mcore/test_qwen38_next_model_cp.py

RUN_QWEN38_MODEL_CP_TESTS=1 QWEN38_TINY_EXPORT_DIR=/path/to/tiny-export \
  QWEN38_MODEL_CP_REFERENCE=/path/to/new-cp1-output \
  QWEN38_MODEL_CP_OUTPUT=/path/to/new-cp2-output \
  torchrun --standalone --nproc-per-node=2 -m pytest -s -q \
  tests/models/mcore/test_qwen38_next_model_cp.py
```

The gate uses the production provider and jagged packing path, actual Core DDP,
two unequal-length document batches, all 48 LoRA gradient tensors, an AdamW
update, frozen-base checks, adapter disabling and 78-tensor HF export. It does
not bypass the provider guard: the initial CP2 attempt failed that guard before
the candidate was enabled. Base and initial-adapter logprobs match CP1 exactly.
Normal/recompute gradient maximum relative L2 is 0.00059369; after the update,
the two batches' mean logprob gaps are 0.00051237/0.00051043 (maximum 0.00371314).
Independent TP1 vLLM accepts the CP2 export, including disable/remove/reload and
six-token decode/prefill checks: base/adapter mean logprob gaps are
0.00059145/0.00069111. No numerical thresholds were relaxed.

Standalone CP2 is enabled only with TP/PP/EP=1, `calculate_per_token_loss=True`
and CP-divisible GDN key/value head counts. CP4 is a component result only;
CP>2 and CP combined with PP/VPP remain rejected. The TP2/EP2 extension is
described below. The DDP gate uses an explicitly
globally normalized synthetic objective and SUM reduction; it is not the
production trainer's token-count normalization or effective RL learning.
These are four-layer random text-model results, not full-checkpoint or visual CP
validation. The current Flash-Next CPU collection passes 68 tests with 34
opt-in skips; skipped tests are not passing execution evidence.

Separately, the real GRPO fixture completed two steps and full checkpoint
saves with actor/reference TP1/EP1/PP1/CP2 and vLLM TP2/EP1. Use the two-GPU
wrapper above, overriding actor and ref TP/EP to 1 and CP to 2, with
`actor_rollout_ref.actor.loss_agg_mode=token-mean`. This executes the production
token-count normalization, HTTP rollout, full recompute, sleep/wake and adapter
sync, not the synthetic squared-logits objective in the numerical gate.
All 184 TensorBoard scalar samples were finite; grad norms were 0.28595847 and
0.28534076. The saved checkpoints contain model/optimizer/extra at steps 1/2.
This is still a random-model smoke with synthetic rewards, not evidence of
learning quality, full-checkpoint correctness or bitwise resume equivalence.

The first independent CP2 resume exposed a generic checkpoint-loading issue:
expert factories create no-grad views, while distributed restore can mutate
their live storage before merging them with autograd enabled. The loader now
keeps Core's build/load/merge in one scoped `torch.no_grad()` region, without
disabling subsequent training or modifying shared dependencies. CPU regression
tests reproduce the old view error, verify exact restoration and subsequent
backward, and check caller grad-mode restoration on success and failure.
The new tests plus existing checkpoint-manager tests pass 43 cases; the full
combined Flash-Next/checkpoint CPU collection passes 111 with 34 opt-in skips.
After that fix, a fresh standalone CP2 trainer restored adapter, optimizer,
scheduler and RNG, completed step 2 and saved optimizer/scheduler step 2. This
does not establish bitwise trajectory equivalence with the uninterrupted run.

### PLE SP gradient ownership and TP2/EP2/CP2

PLE's sequence-parallel all-gather must reduce-scatter SUM its input gradient:
each TP rank consumes different output rows, whose causal convolutions can
read another rank's input rows. The original split-only backward silently lost
those contributions despite correct forward outputs. This is a correctness
fix, not a claimed throughput optimization; backward now includes a collective.
The frozen QSA indexer's no-grad gather is a different contract and is unchanged.

`test_qwen38_next_ple_sp_cp.py` sets loss only on the right SP output shard,
requiring the left input shard to receive its nonzero convolution gradient.
With `RUN_QWEN38_PLE_SP_CP_TESTS=1` and the original tiny fixture, run two
workers for TP2/CP1 or four for TP2/CP2. The old implementation fails both
topologies with relative input-gradient L2 error 1.0 on the left shard; the
fixed implementation matches the reference exactly for two packed batches,
ordinary/full recompute and queued-context cleanup. Failures are coordinated
across ranks so a numerical negative control cannot strand peers in the next
collective. Earlier finite-gradient/forward tests did not cover this invariant.

The full model gate above also accepts `QWEN38_MODEL_TP=2 QWEN38_MODEL_EP=2`:
use two workers for its independent CP1 reference, then four for CP2. It saves
and compares each TP shard separately, gathering vocab-parallel logits before
the globally normalized objective. Both TP/EP sizes remain fixed between these
runs; this comparison does not claim TP1-to-TP2 gradient equivalence.
Each rank passed the 48-LoRA-gradient ordinary/recompute, AdamW, frozen-base,
adapter-disable and 78-tensor export checks. Maximum CP2/CP1 gradient relative
L2 was 7.30e-8, with no relaxed gates. Base and initialized-adapter logprob gaps
were zero; updated batch means were 0 and 0.000044384 (maximum 0.001970768).
Independent TP1 vLLM passed both exports: base/adapter mean gaps were
0.00061147/0.00066502, including disable/remove/reload and six-token decode.

TP2/EP2/CP2 additionally requires sequence parallel, PP1, per-token loss and
GDN key/value head counts divisible by TP*CP, not merely CP. Other TP/EP pairs,
CP>2 and CP with pipeline stages remain rejected. These are random four-layer
text-model gates, not the requested eight-rank TP2/PP2/EP2/CP2/VPP2 schedule,
full-checkpoint execution, visual CP or effective RL learning. The current
combined CPU regression has 111 passing tests and 35 opt-in skips.

The separate four-GPU GRPO fixture also completed two steps and full saves with
actor/reference TP2/EP2/CP2/PP1 and vLLM TP4/EP1. Set `QWEN38_SMOKE_GPUS=4`
in the wrapper, override actor/ref TP=2, EP=2, CP=2, and use `token-mean` loss.
This covers real HTTP rollout, token-count normalization, full recompute,
sleep/wake and adapter sync across different training/rollout layouts.
All 184 TensorBoard scalars were finite. Both checkpoints have world_size=4
and four optimizer fragments, each at the matching step 1/2, with scheduler
step 1/2. Saving and counter checks are not proof of actual restoration.

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

- CP1, or CP2 with TP1/EP1 or TP2/EP2 and the restrictions above; CP with
  PP/VPP remains guarded. PP requires dynamic P2P shapes, and VPP additionally requires overlap
  P2P. PP2/VPP2 is tested on the random fixture. TP/EP and complete-model PP still require full-model
  validation; HC residual width is not the ordinary hidden width.
- Full-layer recompute only; no selective attention recompute, CUDA graphs,
  or activation offloading in this first implementation.
- One packed stream per microbatch; independently keyed multiple PLE layers
  and padding gaps are not implemented. Synchronous VPP remains guarded.
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
