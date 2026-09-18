#!/usr/bin/env bash
# Explicit REAL-WEIGHT 12-layer prefix integration test, NOT the full model.
# Retains PLE/width/experts/vocabulary; use level-1 sleep for this gate.
set -euo pipefail
: "${MODEL_PATH:?Set the separately materialized 12-layer checkpoint}"
: "${TRAIN_FILE:?Set the training parquet}"
: "${VAL_FILE:?Set the validation parquet}"
: "${QWEN38_SUBSET_OUTPUT:?Set a fresh output directory}"
: "${QWEN38_RAY_TEMP:?Set a short unique Ray temp directory}"
: "${CUDA_VISIBLE_DEVICES:?Select eight GPUs}"
[[ ! -e "$QWEN38_SUBSET_OUTPUT" && ! -e "$QWEN38_RAY_TEMP" ]] || exit 3
[[ -r "$TRAIN_FILE" && -r "$VAL_FILE" ]] || exit 4
python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

import torch

model = Path(os.environ["MODEL_PATH"])
manifest = json.loads((model / "subset_manifest.json").read_text())
config = json.loads((model / "config.json").read_text())
assert manifest["complete"] and manifest["layers"] == 12 and manifest["source_layers"] == 48
assert manifest["kind"] == "real_checkpoint_decoder_prefix_integration_only"
assert config["model_type"] == "qwen4_exp" and config["text_config"]["num_hidden_layers"] == 12
assert config["text_config"]["layer_types"].count("full_attention") == 3
index = json.loads((model / "model.safetensors.index.json").read_text())
assert all((model / name).is_file() for name in set(index["weight_map"].values()))
assert torch.cuda.device_count() == 8
for device in range(8):
    assert torch.cuda.mem_get_info(device)[0] >= 50 * 1024**3, "Need 50 GiB free per GPU; never evict jobs"
available_kib = int(next(
    line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
    if line.startswith("MemAvailable:")
))
assert available_kib >= 512 * 1024**2, "Need 512 GiB available host memory; frozen PLE is NOT reduced"
assert shutil.disk_usage(Path(os.environ["QWEN38_SUBSET_OUTPUT"]).parent).free >= 400 * 1024**3
print("QWEN38_12_LAYER_PREFLIGHT passed; sleep=1; resource thresholds are not peak guarantees")
PY

resume_args=(trainer.resume_mode=disable)
if [[ -n "${QWEN38_RESUME_FROM:-}" ]]; then
    [[ -f "$QWEN38_RESUME_FROM/actor/ckpt_contents.json" ]] || exit 5
    resume_args=(trainer.resume_mode=resume_path "trainer.resume_from_path=$QWEN38_RESUME_FROM")
fi
mkdir -p "$QWEN38_SUBSET_OUTPUT" "$QWEN38_RAY_TEMP"
export TENSORBOARD_DIR="$QWEN38_SUBSET_OUTPUT/tensorboard"
parallel_args=()
for role in actor ref; do
    parallel_args+=(
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
    "${parallel_args[@]}" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.expert_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024 \
    reward.num_workers=1 \
    ray_kwargs.ray_init.num_cpus=40 \
    +ray_kwargs.ray_init.num_gpus=8 \
    +ray_kwargs.ray_init.address=local \
    "+ray_kwargs.ray_init._temp_dir=$QWEN38_RAY_TEMP" \
    trainer.experiment_name=real_12layer_level1_hybrid_smoke \
    "trainer.default_local_dir=$QWEN38_SUBSET_OUTPUT/checkpoints" \
    trainer.save_freq=1 \
    trainer.max_actor_ckpt_to_keep=2 \
    "${resume_args[@]}" \
    "$@"
