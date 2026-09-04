#!/usr/bin/env bash
# GRPO | Qwen3-4B | FSDP2 + vLLM | intentional CUDA OOM on one node with 8 GPUs
# Based on examples/grpo_trainer/run_qwen3_4b_fsdp.sh. Run from the repository root.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/grpo_oom_demo}
if [ "${DEVICE:-gpu}" != gpu ]; then
    echo "This example tests the CUDA allocator and requires DEVICE=gpu." >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT=$(cd "${OUTPUT_ROOT}" && pwd)
# A fresh directory prevents artifacts from an earlier run from making the check pass.
RUN_DIR=$(mktemp -d "${OUTPUT_ROOT}/run.XXXXXX")
echo "OOM demo output: ${RUN_DIR}"

# Force the native allocator in the driver AND Ray workers (including existing local Ray).
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export PYTORCH_ALLOC_CONF=backend:native
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi

# 8 prompts * 2 responses / 8 DP ranks = 2 samples/rank. One mini-batch,
# micro-batch and PPO epoch make loss call #3 coincide with GRPO step #3.
# Keep these batch/parallelism/step settings unchanged for this test.
ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "data.train_files='${RUN_DIR}/train.parquet'"
    "data.val_files='${RUN_DIR}/test.parquet'"
    data.train_batch_size=8
    data.max_prompt_length=512
    data.max_response_length=256
    data.dataloader_num_workers=0
    data.truncation=error
    +data.apply_chat_template_kwargs.enable_thinking=False
    "actor_rollout_ref.model.path='${MODEL_PATH}'"
    actor_rollout_ref.model.external_lib=examples.profile.grpo_oom_loss
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.policy_loss.loss_mode=grpo_oom_demo
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.n=2
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25
    actor_rollout_ref.rollout.max_model_len=768
    actor_rollout_ref.rollout.max_num_batched_tokens=1024
    actor_rollout_ref.rollout.max_num_seqs=16
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.profiler.tool=null
    actor_rollout_ref.rollout.profiler.enable=False
    global_profiler.tool=torch_memory
    global_profiler.steps='[1,2,3]'
    global_profiler.profile_continuous_steps=False
    "global_profiler.save_path='${RUN_DIR}/snapshots'"
    global_profiler.global_tool_config.torch_memory.dump_on_oom=True
    global_profiler.global_tool_config.torch_memory.memory_snapshot_num_steps=2
    global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries=100000
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.all_ranks=False
    actor_rollout_ref.actor.profiler.ranks='[0]'
    reward.num_workers=1
    trainer.use_v1=True
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.total_epochs=1
    trainer.total_training_steps=3
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.logger='[console]'
    trainer.project_name=grpo_oom_demo
    trainer.experiment_name=qwen3_4b_fsdp2
    "trainer.default_local_dir='${RUN_DIR}/checkpoints'"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF=backend:native
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_ALLOC_CONF=backend:native
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONUNBUFFERED='1'"
)

# Compose the exact launch configuration without loading models or touching CUDA.
if [ "${CHECK_CONFIG:-0}" = 1 ]; then
    "${LAUNCH[@]}" scripts/print_cfg.py --cfg job --resolve "${ARGS[@]}" "${RAY[@]}" "$@"
    exit 0
fi

"${LAUNCH[@]}" examples/profile/grpo_oom_demo.py prepare --run-dir "${RUN_DIR}"
set +e
"${LAUNCH[@]}" -m verl.trainer.main_ppo "${ARGS[@]}" "${RAY[@]}" "$@" 2>&1 | tee "${RUN_DIR}/train.log"
RUN_STATUS=("${PIPESTATUS[@]}")
set -e
if [ "${RUN_STATUS[1]}" != 0 ]; then
    echo "Failed to write train.log (tee exit ${RUN_STATUS[1]})." >&2
    exit 1
fi
# Training MUST fail, but an arbitrary setup error is not a passing test.
"${LAUNCH[@]}" examples/profile/grpo_oom_demo.py check \
    --run-dir "${RUN_DIR}" --trainer-exit-code "${RUN_STATUS[0]}"
