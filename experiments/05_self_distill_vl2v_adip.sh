#!/usr/bin/env bash
# Self-distillation with VL2V-ADiP: student = trainable copy of the VLM image encoder.
#
# Usage:  VLM=quiltnet K=3 bash experiments/05_self_distill_vl2v_adip.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PROMPT_SOURCE="${PROMPT_SOURCE:-dipt}"
STEPS="${STEPS:-2500}"
LR="${LR:-5e-5}"
LAM="${LAM:-0.5}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-200}"

banner "Self-distillation - VL2V-ADiP (${PROMPT_SOURCE} prompts)"

python scripts/train_vl2v_adip.py \
  --vlm "$VLM" \
  --dataset "$DATASET" \
  --student vlm \
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
  2>&1 | tee "$(log_file "self_distill_vl2v_${PROMPT_SOURCE}")"
