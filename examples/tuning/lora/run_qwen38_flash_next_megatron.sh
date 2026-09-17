#!/usr/bin/env bash
# Experimental Flash-Next GRPO + Megatron-Bridge LoRA + vLLM recipe.
# Full-checkpoint numerical/reload validation is required before a long run.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the complete Qwen3.8-Flash-Next checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to a verl-format training parquet}"
: "${VAL_FILE:?Set VAL_FILE to a verl-format validation parquet}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
# vLLM subprocesses and Ray workers must see the real HF config registration.
export VERL_USE_EXTERNAL_MODULES=verl.models.mcore.qwen3_8_next.bridge

# First target: a single node, TP8/PP1/CP1. Not yet a validated full-model run.
python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.external_lib=verl.models.mcore.qwen3_8_next.bridge \
    actor_rollout_ref.model.lora.rank=16 \
    actor_rollout_ref.model.lora.alpha=32 \
    actor_rollout_ref.model.lora.merge=False \
    'actor_rollout_ref.model.lora.target_modules=[language_model.decoder.layers.*.self_attention.linear_qkv,language_model.decoder.layers.*.self_attention.linear_proj,language_model.decoder.layers.*.self_attention.in_proj,language_model.decoder.layers.*.self_attention.out_proj,language_model.decoder.layers.*.mlp.*.linear_fc1,language_model.decoder.layers.*.mlp.*.linear_fc2]' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=8 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.context_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name=qwen38_flash_next_lora \
    trainer.experiment_name=integration_smoke \
    'trainer.logger=[console,tensorboard]' \
    trainer.total_training_steps=2 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    "$@"
