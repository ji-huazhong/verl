#!/usr/bin/env bash
# Real 48-layer checkpoint: TP2/PP2/EP2/CP2/VPP2 -> TP8 vLLM, two GRPO steps.
# This is a short integration/resume gate, not an accuracy benchmark.
# Use the validated dependencies/private Bridge overlay documented by the model plugin.
set -euo pipefail

: "${MODEL_PATH:?Set the complete Qwen3.8-Flash-Next checkpoint}"
: "${TRAIN_FILE:?Set the real verl-format training parquet}"
: "${VAL_FILE:?Set the real verl-format validation parquet}"
: "${QWEN38_FULL_OUTPUT:?Set a fresh output directory}"
: "${QWEN38_RAY_TEMP:?Set a short unique Ray temp directory}"
: "${CUDA_VISIBLE_DEVICES:?Select eight GPUs with sufficient free memory}"
[[ ! -e "$QWEN38_FULL_OUTPUT" && ! -e "$QWEN38_RAY_TEMP" ]] || exit 3
[[ -r "$TRAIN_FILE" && -r "$VAL_FILE" ]] || exit 4

python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

import torch

config = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
assert config["model_type"] == "qwen4_exp"
assert config["text_config"]["num_hidden_layers"] == 48, "Never substitute a reduced fixture"
assert config["text_config"]["hidden_size"] >= 1024
index = json.loads((Path(os.environ["MODEL_PATH"]) / "model.safetensors.index.json").read_text())
assert all(
    (Path(os.environ["MODEL_PATH"]) / name).is_file()
    for name in set(index["weight_map"].values())
)
assert torch.cuda.device_count() == 8
for device in range(8):
    assert torch.cuda.mem_get_info(device)[0] >= 100 * 1024**3, "Need 100 GiB free per GPU; never evict jobs"
available_kib = int(next(
    line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
    if line.startswith("MemAvailable:")
))
assert available_kib >= 1024**3, "Need 1 TiB available host memory for this full-model smoke"
assert shutil.disk_usage(Path(os.environ["QWEN38_FULL_OUTPUT"]).parent).free >= 100 * 1024**3
print("QWEN38_REAL_CHECKPOINT_PREFLIGHT layers=48 GPUs=8 passed; resource checks are not peak guarantees")
PY

resume_args=(trainer.resume_mode=disable)
if [[ -n "${QWEN38_RESUME_FROM:-}" ]]; then
    [[ -f "$QWEN38_RESUME_FROM/actor/ckpt_contents.json" ]] || exit 5
    resume_args=(trainer.resume_mode=resume_path "trainer.resume_from_path=$QWEN38_RESUME_FROM")
fi
mkdir -p "$QWEN38_FULL_OUTPUT" "$QWEN38_RAY_TEMP"
export TENSORBOARD_DIR="$QWEN38_FULL_OUTPUT/tensorboard"

hybrid_args=()
for role in actor ref; do
    hybrid_args+=(
        "actor_rollout_ref.$role.megatron.tensor_model_parallel_size=2"
        "actor_rollout_ref.$role.megatron.pipeline_model_parallel_size=2"
        "actor_rollout_ref.$role.megatron.virtual_pipeline_model_parallel_size=2"
        "actor_rollout_ref.$role.megatron.expert_model_parallel_size=2"
        "actor_rollout_ref.$role.megatron.expert_tensor_parallel_size=1"
        "actor_rollout_ref.$role.megatron.context_parallel_size=2"
        "actor_rollout_ref.$role.megatron.sequence_parallel=True"
    )
done

bash examples/tuning/lora/run_qwen38_flash_next_megatron.sh \
    "${hybrid_args[@]}" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.expert_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=536870912 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048 \
    reward.num_workers=1 \
    ray_kwargs.ray_init.num_cpus=40 \
    +ray_kwargs.ray_init.num_gpus=8 \
    +ray_kwargs.ray_init.address=local \
    "+ray_kwargs.ray_init._temp_dir=$QWEN38_RAY_TEMP" \
    trainer.experiment_name=full_checkpoint_hybrid_smoke \
    "trainer.default_local_dir=$QWEN38_FULL_OUTPUT/checkpoints" \
    trainer.save_freq=1 \
    trainer.max_actor_ckpt_to_keep=2 \
    "${resume_args[@]}" \
    "$@"
