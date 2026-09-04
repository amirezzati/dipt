#!/usr/bin/env bash
# Stage 2 - RISE distillation into a ResNet-50 student.
#
# Usage:  VLM=plip K=4 PROMPT_SOURCE=dipt bash experiments/02_rise.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STUDENT="${STUDENT:-resnet50_bit}"
PROMPT_SOURCE="${PROMPT_SOURCE:-dipt}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-8e-4}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.3}"
CLASSIFICATION_WEIGHT="${CLASSIFICATION_WEIGHT:-0.4}"
DISTANCE_WEIGHT="${DISTANCE_WEIGHT:-0.3}"
TEMPERATURE="${TEMPERATURE:-2.0}"

banner "Stage 2 - RISE (${PROMPT_SOURCE} prompts, student=${STUDENT})"

python scripts/train_rise.py \
  --vlm "$VLM" \
  --dataset "$DATASET" \
  --student "$STUDENT" \
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
  2>&1 | tee "$(log_file "rise_${STUDENT}_${PROMPT_SOURCE}")"
