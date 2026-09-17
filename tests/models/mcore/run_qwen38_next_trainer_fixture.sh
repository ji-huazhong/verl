#!/usr/bin/env bash
# RANDOM four-layer integration smoke; never a full-model/accuracy benchmark.
# Run from the repository root after prepare_qwen38_next_trainer_fixture.py.
set -euo pipefail

: "${QWEN38_TRAINER_FIXTURE:?Set the prepared random fixture directory}"
: "${QWEN38_SMOKE_OUTPUT:?Set a fresh output directory}"
: "${QWEN38_RAY_TEMP:?Set a short, unique Ray temp directory}"
: "${CUDA_VISIBLE_DEVICES:?Select exactly one GPU with sufficient free memory}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || exit 2
[[ ! -e "$QWEN38_SMOKE_OUTPUT" ]] || exit 3
python3 -c 'import torch; assert torch.cuda.mem_get_info()[0] > 10 * 1024**3, "Need 10 GiB free; never evict another job"'
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
    trainer.n_gpus_per_node=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
    ray_kwargs.ray_init.num_cpus=16 \
    +ray_kwargs.ray_init.num_gpus=1 \
    +ray_kwargs.ray_init.address=local \
    "+ray_kwargs.ray_init._temp_dir=$QWEN38_RAY_TEMP" \
    trainer.experiment_name=random_trainer_tp1 \
    "trainer.default_local_dir=$QWEN38_SMOKE_OUTPUT/checkpoints" \
    trainer.save_freq=1 \
    "${resume_args[@]}" \
    "$@"
