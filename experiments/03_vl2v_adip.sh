#!/usr/bin/env bash
# Stage 2 - VL2V-ADiP distillation (projection stage + encoder stage).
#
# Usage:  VLM=plip K=4 STUDENT=vit_base bash experiments/03_vl2v_adip.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STUDENT="${STUDENT:-resnet50}"
PROMPT_SOURCE="${PROMPT_SOURCE:-dipt}"
STEPS="${STEPS:-2500}"
LR="${LR:-5e-5}"
LAM="${LAM:-0.5}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-200}"

banner "Stage 2 - VL2V-ADiP (${PROMPT_SOURCE} prompts, student=${STUDENT})"

python scripts/train_vl2v_adip.py \
  --vlm "$VLM" \
  --dataset "$DATASET" \
  --student "$STUDENT" \
  --prompt-source "$PROMPT_SOURCE" \
  --num-context-tokens "$K" \
  --steps "$STEPS" \
  --lr "$LR" \
  --lam "$LAM" \
  --checkpoint-freq "$CHECKPOINT_FREQ" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --gpu-index "$GPU" \
  --seed "$SEED" \
  2>&1 | tee "$(log_file "vl2v_${STUDENT}_${PROMPT_SOURCE}")"
