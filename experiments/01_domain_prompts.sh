#!/usr/bin/env bash
# Stage 1 - learn one DIPT prompt per training domain.
#
# Usage:  VLM=quiltnet K=3 bash experiments/01_domain_prompts.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-5}"
SCORE_WEIGHT="${SCORE_WEIGHT:-0.5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"

banner "Stage 1 - DIPT prompt tuning"

python scripts/train_domain_prompts.py \
  --vlm "$VLM" \
  --dataset "$DATASET" \
  --domain all \
  --num-context-tokens "$K" \
  --num-epochs "$EPOCHS" \
  --lr "$LR" \
  --score-weight "$SCORE_WEIGHT" \
  --eval-interval "$EVAL_INTERVAL" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --gpu-index "$GPU" \
  --seed "$SEED" \
  2>&1 | tee "$(log_file domain_prompts)"
