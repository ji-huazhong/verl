#!/usr/bin/env bash
# Run from an environment prepared with this directory's dedicated requirements.
set -euo pipefail
: "${QWEN38_MODEL_PATH:?Set the complete HF checkpoint directory}"
: "${QWEN38_TRAIN_FILE:?Set the training parquet}"
: "${QWEN38_VAL_FILE:?Set the validation parquet}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VERL_USE_UV=0
export VLLM_USE_RUST_FRONTEND=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_QWEN4_EXP_HC_FP32=1
export VLLM_QWEN4_EXP_GDN_CONV_FP32=1
export QWEN38_PRECISION_PROFILE=${QWEN38_PRECISION_PROFILE:-hc-gdn-fp32-ple-state}
# An unpatched wheel silently ignores these flags. Reject it before Ray starts.
"${PYTHON_BIN:-python3}" - <<'PY'
from importlib.metadata import version
import os

expected = "0.30.1.dev0+g6e517b15.torch210.cu131.hcgdnfp32"
profile = os.environ["QWEN38_PRECISION_PROFILE"]
overrides = "precision-overrides.txt"
if profile == "hc-gdn-fp32-ple-state":
    expected += ".plestate"
    overrides = "ple-state-overrides.txt"
elif profile != "hc-gdn-fp32":
    raise SystemExit(f"Unknown training precision profile: {profile}")
actual = version("vllm")
if actual != expected:
    raise SystemExit(
        f"Expected vLLM {expected}, found {actual}. "
        f"Install requirements.txt with uv --override {overrides}."
    )
PY
QWEN38_CACHE_DIR=${QWEN38_CACHE_DIR:-/tmp/verl-qwen38-next-cache}
export XDG_CACHE_HOME="${QWEN38_CACHE_DIR}/xdg"
export TRITON_CACHE_DIR="${QWEN38_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${QWEN38_CACHE_DIR}/torchinductor"
export TILELANG_CACHE_DIR="${QWEN38_CACHE_DIR}/tilelang"
export FLASHINFER_WORKSPACE_BASE="${QWEN38_CACHE_DIR}/flashinfer"
export VLLM_CACHE_ROOT="${QWEN38_CACHE_DIR}/vllm"
# Resolve these on the driver: Ray does not run Hydra's resolver setup.
VERL_FILE_LOGGER_ROOT=${VERL_FILE_LOGGER_ROOT:-${QWEN38_OUTPUT_DIR:-outputs/qwen3_8_next_gsm8k_frozen_ple}/metrics/$(date +%Y%m%d_%H%M%S)_$$}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${QWEN38_OUTPUT_DIR:-outputs/qwen3_8_next_gsm8k_frozen_ple}/tensorboard}
exec "${PYTHON_BIN:-python3}" -m verl.trainer.main_ppo --config-name="${QWEN38_TRAIN_CONFIG:-qwen3_8_next_gsm8k}" \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=${VERL_FILE_LOGGER_ROOT}" \
  "ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=${TENSORBOARD_DIR}" "$@"
