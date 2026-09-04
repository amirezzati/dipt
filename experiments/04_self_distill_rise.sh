#!/usr/bin/env bash
# Self-distillation with RISE: student = trainable copy of the VLM image encoder.
#
# Usage:  VLM=quiltnet K=3 bash experiments/04_self_distill_rise.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PROMPT_SOURCE="${PROMPT_SOURCE:-dipt}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-8e-4}"
# The self-distillation runs drop the KL term: teacher and student share an
# architecture, so the supervised and distance terms carry the signal.
# Override on the command line to re-enable it.
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.0}"
CLASSIFICATION_WEIGHT="${CLASSIFICATION_WEIGHT:-0.6}"
DISTANCE_WEIGHT="${DISTANCE_WEIGHT:-0.4}"
TEMPERATURE="${TEMPERATURE:-2.0}"

banner "Self-distillation - RISE (${PROMPT_SOURCE} prompts)"

python scripts/train_rise.py \
  --vlm "$VLM" \
  --dataset "$DATASET" \
  --student vlm \
  --prompt-source "$PROMPT_SOURCE" \
  --num-context-tokens "$K" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --distill-weight "$DISTILL_WEIGHT" \
  --classification-weight "$CLASSIFICATION_WEIGHT" \
  --distance-weight "$DISTANCE_WEIGHT" \
  -T "$TEMPERATURE" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --gpu-index "$GPU" \
  --seed "$SEED" \
  2>&1 | tee "$(log_file "self_distill_rise_${PROMPT_SOURCE}")"
