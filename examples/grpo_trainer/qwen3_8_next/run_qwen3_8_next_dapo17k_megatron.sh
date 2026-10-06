#!/usr/bin/env bash
# Supply deduplicated official DAPO/AIME files through QWEN38_TRAIN_FILE/VAL_FILE.
set -euo pipefail
case "${QWEN38_TRAIN_GPUS:-64}:${QWEN38_QUICK_DEBUG:-0}" in
  64:0) export QWEN38_TRAIN_CONFIG=qwen3_8_next_dapo17k ;;
  64:1) export QWEN38_TRAIN_CONFIG=qwen3_8_next_dapo17k_debug ;;
  128:0) export QWEN38_TRAIN_CONFIG=qwen3_8_next_dapo17k_128gpu ;;
  *) echo "Use QWEN38_TRAIN_GPUS=64|128; QWEN38_QUICK_DEBUG=1 requires 64 GPUs" >&2; exit 2 ;;
esac
exec bash "$(dirname "$0")/run_qwen3_8_next_gsm8k_megatron.sh" "$@"
