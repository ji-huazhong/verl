# Qwen3.8-Flash-Next: Megatron + vLLM

This recipe supports the native `qwen4_exp` / `Qwen4ExpForConditionalGeneration`
checkpoint. The model adapter lives in
[`verl/models/mcore/qwen3_8_next`](../../../verl/models/mcore/qwen3_8_next).
It trains the policy with **only the large PLE n-gram lookup table frozen** in
pinned host memory. PLE projection, gate, normalization and convolution layers
remain trainable. LoRA and MTP are disabled. Hard QSA top-k selection has no
auxiliary indexer objective; text-only training supplies no vision loss.

Read the [adaptation report](report.html), [change inventory](CHANGES.md) and
[external dependency patch guide](UPSTREAM_PATCHES.md). Original third-party
attribution is in [THIRD_PARTY.md](../../../verl/models/mcore/qwen3_8_next/THIRD_PARTY.md).
Raw experiment logs, activations, model paths and internal service credentials
are not repository artifacts.

## Registration and weight lifecycle

Set `actor_rollout_ref.model.external_lib=verl.models.mcore.qwen3_8_next.bridge`.
Importing this module registers the model through
`MegatronModelBridge.register_bridge(source="Qwen4ExpForConditionalGeneration",
model_type="qwen4_exp", target=Qwen3VLModel, provider=Qwen38NextModelProvider)`.
No installed Bridge registry or Transformers registry needs editing.

The provider installs HC / GDN / QSA / PLE layers and packed-sequence hooks.
Mappings validate source and local target coverage across TP/PP/EP, preserve
ordinary GDN output-norm gamma, and stream PLE HF shards. The complete original
HF checkpoint remains required for immutable PLE metadata and the frozen table.
Trainable-table support is retained for independent regression tests.

Training starts from HF when `dist_checkpointing_path` is unset and saves
model/optimizer shards for resume. Streaming actor export is for refitting an
already initialized rollout; it does not create a self-contained HF checkpoint.
The Core 0.19.2 CPU optimizer resume compatibility fix is in the verl plugin;
see the external patch guide for its exact process-wide scope.

## Runtime and dependency build

Use a separate Python 3.13 / Torch 2.10.0 / CUDA 13.1 environment, with matching
Transformer Engine and FlashAttention binaries. Do not install this profile
into an unrelated environment or combine it with verl's normal dependency
resolution. This profile intentionally uses Transformers 5.16.1.

The validated vLLM wheel is based on main commit
`6e517b15c1833cf72a7f557ee32524d98682e617`, version
`0.30.1.dev0+g6e517b15.torch210.cu131.hcgdnfp32.plestate`.
Bridge is `0.6.2+nebula1`, built from upstream v0.6.2 with Python metadata changes.
**An unmodified stock vLLM wheel does not contain the required PLE state fix.**
All custom dependency patches and preparation/build tools are checked into this
directory. [UPSTREAM_PATCHES.md](UPSTREAM_PATCHES.md) records their order,
base revisions, external effects and upstream review candidates.

With the ABI-matched Torch/TorchVision/TorchAudio/Triton already supplied by the
isolated GPU image, install the validated wheels from your wheelhouse:

```bash
PROFILE=examples/grpo_trainer/qwen3_8_next
uv pip install --python "$PYTHON_BIN" --no-deps "$WHEELHOUSE/megatron_bridge-0.6.2+nebula1-py3-none-any.whl"
uv pip install --python "$PYTHON_BIN" --find-links "$WHEELHOUSE" \
  --excludes "$PROFILE/excludes.txt" --override "$PROFILE/ple-state-overrides.txt" \
  -r "$PROFILE/requirements.txt"
uv pip install --python "$PYTHON_BIN" --no-deps -e .
```

Install ABI-matched TE separately; the validated image supplies it. The excludes
file protects preinstalled ABI-sensitive dependencies. The underlying native
vLLM pin and older `precision-overrides.txt` remain available for reproducing
historical diagnostics; normal training uses `ple-state-overrides.txt`.

## DAPO-17K training

`prepare_dapo17k.py` verifies the pinned official DAPO-Math-17k / AIME-2024
Parquet hashes, removes exact duplicates and conflicting labels, and checks
exact train/validation overlap. It preserves prompt text. The prepared data
contains 17,391 training prompts and 30 AIME-2024 validation prompts; exact
matching is not a semantic contamination audit.

```bash
PROFILE=examples/grpo_trainer/qwen3_8_next
python "$PROFILE/prepare_dapo17k.py" --train-source /data/dapo-official.parquet \
  --val-source /data/aime-official.parquet --output-dir /data/dapo-prepared
export QWEN38_MODEL_PATH=/path/to/complete/hf/checkpoint
export QWEN38_TRAIN_FILE=/data/dapo-prepared/train.parquet
export QWEN38_VAL_FILE=/data/dapo-prepared/aime2024.parquet
export QWEN38_OUTPUT_DIR=/path/to/checkpoints/qwen38-dapo
QWEN38_TRAIN_GPUS=128 bash "$PROFILE/run_qwen3_8_next_dapo17k_megatron.sh"
```

Run through the existing verl/Ray multi-node launcher with the required nodes
available. These scripts do not allocate a cluster. `QWEN38_TRAIN_GPUS=64`
selects the earlier topology; add `QWEN38_QUICK_DEBUG=1` for its small debug
profile. For GSM8K, supply its Parquets and use
`run_qwen3_8_next_gsm8k_megatron.sh`.

| Setting | DAPO 128-GPU profile |
| --- | --- |
| Actor / reference | TP4, PP2, EP8, ETP1, CP1; dense DP16, expert DP8 |
| Rollout | Sixteen node-local TP8 replicas; PP1, DP1, EP1 per replica |
| Batch / mini-batch | 128 prompts / 128; 8 responses per prompt (1,024 responses/update) |
| Prompt / response cap | 2,048 / 8,192 tokens |
| Training | GRPO, LR 1e-6, KL coefficient 0.001, 100 steps |
| Memory | Sequence parallelism, full-layer recompute, parameter/CPU optimizer offload |
| Precision profile | HC/GDN FP32, fixed Q/K normalization, R3, canonical QSA order, PLE state retention |
| Validation / checkpoint | `val_before_train=false`, every 50 steps; validation n=4 |

The launchers reject an incompatible vLLM version before Ray starts and set
node-local compiler caches. Metrics use console, TensorBoard and a unique
per-launch JSONL directory beneath the output root. Checkpoint resume preserves
previous metric files. Optional real-response capture and layer tracing are
disabled unless their diagnostic environment variables are explicitly set.

## Numerical acceptance and final run

The logged `training/train_rollout_logprob_abs_diff` is a masked token mean
within each generated response, then a sample mean before policy update.
`..._token_abs_diff` is token weighted; `..._max_abs_diff` and
`..._nonfinite_tokens` expose tails. These are logprob differences, distinct
from the historical `rollout_probs_diff_*` probability metrics. R3 is the
existing verl route replay mode, enabled in this recipe.

The completed 128-GPU DAPO run has all 100 consecutive step records. Response
mean error ranges from 0.007287 to 0.010918; the last ten steps average 0.008386.
No nonfinite sampled-token logprobs were recorded. Final training score is
0.728516 (last-ten mean 0.784961). These are training rewards, not held-out
accuracy. Maximum individual token gaps remain nonzero (peak 7.31425).
Final validation and step-100 checkpoint save were logged, and the training
head exited 0. **The platform task ended failed** because worker log writing /
cleanup failed after training; the report separates that from model execution.

The user accepted the generated-response metric at the approximate Miles
reference scale. This does not establish bitwise Megatron/vLLM equality.
The stricter fixed-prompt/all-token gate remains unpassed; neither its token
population nor its reduction is interchangeable with the training metric.
Full-checkpoint resume and distributed small-model next-update equivalence
were verified in earlier tests; this final step-100 checkpoint has not been
independently restored in a new 128-GPU task.

## Verification and diagnostics

```bash
pytest -q tests/models/mcore/*on_cpu.py tests/utils/debug/test_metrics.py \
  tests/utils/debug/test_logprob_capture.py tests/utils/test_padding_on_cpu.py
pytest -q tests/trainer/ppo/v1/test_logprob_context_metrics_on_cpu.py

RUN_QWEN38_GPU_TESTS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --standalone --nproc-per-node=1 -m pytest -q \
  tests/models/mcore/test_qwen38_next_full_parameter_gpu.py
RUN_QWEN38_PARALLEL_TESTS=1 QWEN38_TEST_DP=2 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --standalone --nproc-per-node=8 -m pytest -q \
  tests/models/mcore/test_qwen38_next_full_parameter_parallel.py
```

The CPU tests cover mapping, packing, passive/restorable hooks, real-response
capture, comparison controls and PLE lifecycle analysis. GPU tests cover
full small-model gradients, frozen/trainable PLE, distributed conversion,
checkpoint state and next-update equality. See each GPU test's environment
flags before launching; local CPU checks do not replace them.

Diagnostic tools are retained under this directory: `check_logprobs.py` and
`prepare_logprob_prompts.py` compare fixed tokens; `backend_trace.py`,
`real_prompt_trace.py` and `production_trace_analysis.py` compare layer
boundaries on identical real responses; `ple_sleep2_ab.py` and
`validate_vllm_model_state_sleep.py` test native state lifecycle.
Run them with `--help` and fresh output directories. Preserve exact tokens,
weights, routes and scheduling controls when attributing numerical drift.
FlashQLA and router-FP32 experiments remain optional candidates, not defaults.
