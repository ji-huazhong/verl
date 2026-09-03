#!/usr/bin/env bash
# BF16 training with dynamically re-quantized INT4 W4A16 vLLM rollout.
# This is the no-fake-QAT/PTQ control for a matched Qwen3-30B-A3B RL study.

set -euo pipefail

export INT4_QAT=True
export INT4_FAKE_QAT=False
export INT4_QAT_CONFIG=${INT4_QAT_CONFIG:-"examples/qat/config/int4_w4a16_qwen3_moe.json"}
export INT4_GROUP_SIZE=${INT4_GROUP_SIZE:-128}
export CPU_OPTIMIZER_OFFLOAD=${CPU_OPTIMIZER_OFFLOAD:-True}

export ACTOR_TP=${ACTOR_TP:-4}
export ACTOR_PP=${ACTOR_PP:-2}
export ACTOR_EP=${ACTOR_EP:-4}
export ACTOR_ETP=${ACTOR_ETP:-1}
export REF_TP=${REF_TP:-${ACTOR_TP}}
export REF_PP=${REF_PP:-${ACTOR_PP}}
export REF_EP=${REF_EP:-${ACTOR_EP}}
export REF_ETP=${REF_ETP:-${ACTOR_ETP}}
export ROLLOUT_TP=${ROLLOUT_TP:-4}
export GEN_MOE_TP=${GEN_MOE_TP:-1}
export GEN_MOE_EP=${GEN_MOE_EP:-${ROLLOUT_TP}}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/run_qwen3_30b_a3b_megatron.sh" "$@"
