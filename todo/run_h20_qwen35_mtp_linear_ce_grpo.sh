#!/usr/bin/env bash
# Qwen3.5-35B-A3B MTP + GRPO validation on DAPO-Math-17k.
#
# The first run populates SKIP_DUMP_DIR. Later runs with the same project,
# experiment, batch shape and step load the identical rollout batch, making
# fused/non-fused actor metrics directly comparable.

set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the Qwen3.5-35B-A3B checkpoint}"
: "${DATA_FILE:?set DATA_FILE to dapo-math-17k.parquet}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a fresh run directory}"
: "${SKIP_DUMP_DIR:?set SKIP_DUMP_DIR to the shared rollout-cache directory}"

FUSED_KERNELS=${FUSED_KERNELS:-False}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_mtp_linear_ce_h20_ab}
PROJECT_NAME=${PROJECT_NAME:-verl_mtp_linear_ce_validation}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
# Keep actor/log-prob and rollout budgets consistent with the sequence limit.
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TP=${TP:-2}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-8}
SEED=${SEED:-42}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
if [[ ! ${TOTAL_TRAINING_STEPS} =~ ^[1-9][0-9]*$ ]]; then
    echo "TOTAL_TRAINING_STEPS must be a positive integer" >&2
    exit 1
fi
SKIP_STEPS="[1"
for ((step = 2; step <= TOTAL_TRAINING_STEPS; step++)); do
    SKIP_STEPS+=",${step}"
done
SKIP_STEPS+="]"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# This test consumes a small number of batches. Avoid tokenizing the complete
# source parquet up front; truncation still bounds every selected prompt.
python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${DATA_FILE}" \
    data.val_files="${DATA_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels="${FUSED_KERNELS}" \
    actor_rollout_ref.model.mtp.enable=True \
    actor_rollout_ref.model.mtp.enable_train=True \
    actor_rollout_ref.model.mtp.detach_encoder=True \
    actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.actor.data_loader_seed="${SEED}" \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${CP}" \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${EP}" \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${ETP}" \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    actor_rollout_ref.actor.megatron.seed="${SEED}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.seed="${SEED}" \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=128 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
    skip.rollout_tq.enable=True \
    skip.rollout_tq.dump_dir="${SKIP_DUMP_DIR}" \
    skip.rollout_tq.steps="${SKIP_STEPS}" \
    skip.rollout_tq.action=cache \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    "$@"
