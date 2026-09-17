#!/usr/bin/env bash
# RANDOM four-layer integration smoke; never a full-model/accuracy benchmark.
# Run from the repository root after prepare_qwen38_next_trainer_fixture.py.
set -euo pipefail

: "${QWEN38_TRAINER_FIXTURE:?Set the prepared random fixture directory}"
: "${QWEN38_SMOKE_OUTPUT:?Set a fresh output directory}"
: "${QWEN38_RAY_TEMP:?Set a short, unique Ray temp directory}"
: "${CUDA_VISIBLE_DEVICES:?Select the test GPUs with sufficient free memory}"
export QWEN38_SMOKE_GPUS="${QWEN38_SMOKE_GPUS:-1}"
case "$QWEN38_SMOKE_GPUS" in 1|2|4|8) ;; *) exit 2 ;; esac
# The production pool reserves three CPU slots per GPU before any workers
# launch. The queue and trainer already consume nine CPU slots BEFORE the
# placement group: 3*8+8 leaves only 23 for its required 24 and waits forever.
# Reserve 16 additional slots for these and subsequent rollout/reward services.
QWEN38_SMOKE_CPUS=$((3 * QWEN38_SMOKE_GPUS + 16))
[[ ! -e "$QWEN38_SMOKE_OUTPUT" ]] || exit 3
python3 - <<'PY'
import json
import os
from pathlib import Path

import torch

count = int(os.environ["QWEN38_SMOKE_GPUS"])
assert torch.cuda.device_count() == count, "Visible GPUs must match QWEN38_SMOKE_GPUS"
config = json.loads((Path(os.environ["QWEN38_TRAINER_FIXTURE"]) / "model/config.json").read_text())
text = config["text_config"]
assert config["model_type"] == "qwen4_exp" and text["num_hidden_layers"] == 4 and text["hidden_size"] == 128
for key in ("num_experts", "num_attention_heads", "linear_num_key_heads", "linear_num_value_heads", "split_ngram_parts"):
    assert text[key] % count == 0, f"Tiny fixture {key} must be divisible by the requested TP/EP size"
for index in range(count):
    assert torch.cuda.mem_get_info(index)[0] > 10 * 1024**3, "Need 10 GiB free on every GPU; never evict another job"
PY
mkdir -p "$QWEN38_SMOKE_OUTPUT" "$QWEN38_RAY_TEMP"
export MODEL_PATH="$QWEN38_TRAINER_FIXTURE/model"
export TRAIN_FILE="$QWEN38_TRAINER_FIXTURE/train.parquet"
export VAL_FILE="$QWEN38_TRAINER_FIXTURE/val.parquet"
export TENSORBOARD_DIR="$QWEN38_SMOKE_OUTPUT/tensorboard"

resume_args=(trainer.resume_mode=disable)
if [[ -n "${QWEN38_RESUME_FROM:-}" ]]; then
    [[ -f "$QWEN38_RESUME_FROM/actor/ckpt_contents.json" ]] || exit 4
    resume_args=(trainer.resume_mode=resume_path "trainer.resume_from_path=$QWEN38_RESUME_FROM")
fi

bash examples/tuning/lora/run_qwen38_flash_next_megatron.sh \
    "trainer.n_gpus_per_node=$QWEN38_SMOKE_GPUS" \
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$QWEN38_SMOKE_GPUS" \
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size=$QWEN38_SMOKE_GPUS" \
    "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$QWEN38_SMOKE_GPUS" \
    "actor_rollout_ref.ref.megatron.expert_model_parallel_size=$QWEN38_SMOKE_GPUS" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$QWEN38_SMOKE_GPUS" \
    actor_rollout_ref.rollout.expert_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.025 \
    actor_rollout_ref.rollout.max_model_len=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=128 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=134217728 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1 \
    data.train_batch_size=2 \
    data.max_prompt_length=64 \
    data.max_response_length=32 \
    data.filter_overlong_prompts=True \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=256 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=256 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=256 \
    reward.num_workers=1 \
    "reward.custom_reward_function.path=$PWD/tests/models/mcore/prepare_qwen38_next_trainer_fixture.py" \
    "ray_kwargs.ray_init.num_cpus=$QWEN38_SMOKE_CPUS" \
    "+ray_kwargs.ray_init.num_gpus=$QWEN38_SMOKE_GPUS" \
    +ray_kwargs.ray_init.address=local \
    "+ray_kwargs.ray_init._temp_dir=$QWEN38_RAY_TEMP" \
    "trainer.experiment_name=random_trainer_tp$QWEN38_SMOKE_GPUS" \
    "trainer.default_local_dir=$QWEN38_SMOKE_OUTPUT/checkpoints" \
    trainer.save_freq=1 \
    "${resume_args[@]}" \
    "$@"
