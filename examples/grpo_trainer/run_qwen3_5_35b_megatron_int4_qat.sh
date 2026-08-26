#!/usr/bin/env bash
# Integer INT4 W4A16 QAT for Qwen/Qwen3.5-35B-A3B routed experts.
# Run from the verl repository root on one 8-GPU Hopper node.

set -euo pipefail

export INT4_QAT=True
export VANILLA_MBRIDGE=False
export INT4_QAT_CONFIG=${INT4_QAT_CONFIG:-"examples/qat/config/int4_w4a16_qwen3_5_moe.json"}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/run_qwen3_5_35b_megatron.sh" "$@"
