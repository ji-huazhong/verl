#!/usr/bin/env bash
# Full-model Geo3K integration gate, not a dataset-wide quality benchmark.
set -euo pipefail

: "${GEO3K_DIR:?Set GEO3K_DIR to the existing verl-format Geo3K parquet directory}"
export TRAIN_FILE="$GEO3K_DIR/train.parquet"
export VAL_FILE="$GEO3K_DIR/test.parquet"
[[ -r "$TRAIN_FILE" && -r "$VAL_FILE" ]] || exit 4

# Keep original images and answers. The production dataset loader filters
# overlong prompts without truncation; do not train on the evaluation split.
# Two steps exercise an adapter update and its use in subsequent rollout.
# Level 2 reloads the complete frozen checkpoint before each adapter update;
# bounded runner-state backup preserves constants outside model.named_buffers().
bash examples/tuning/lora/run_qwen38_flash_next_hybrid_smoke.sh \
    trainer.experiment_name=bridge_geo3k_smoke \
    trainer.total_training_steps=2 \
    data.train_max_samples=32 \
    data.val_max_samples=8 \
    data.filter_overlong_prompts_workers=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.lora_sleep_level=2 \
    "$@"
