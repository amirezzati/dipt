#!/usr/bin/env bash
# Evaluate finished runs of one (dataset, vlm, k, student) configuration.
#
# Usage:  VLM=plip K=4 STUDENT=resnet50_bit bash experiments/06_evaluate.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STUDENT="${STUDENT:-resnet50_bit}"
PROMPT_SOURCE="${PROMPT_SOURCE:-dipt}"
if [ "$PROMPT_SOURCE" = "dipt" ]; then TAG="dipt_k${K}"; else TAG="$PROMPT_SOURCE"; fi

RISE_DIR="${OUTPUT_ROOT}/rise/${DATASET}/${VLM}/${STUDENT}/${TAG}"
VL2V_DIR="${OUTPUT_ROOT}/vl2v_adip/${DATASET}/${VLM}/${STUDENT}/${TAG}"

banner "Evaluation"

if [ -d "$RISE_DIR" ]; then
  echo "--- RISE: $RISE_DIR"
  python scripts/evaluate_students.py \
    --run-dir "$RISE_DIR" \
    --vlm "$VLM" \
    --dataset "$DATASET" \
    --student "$STUDENT" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --gpu-index "$GPU" \
    2>&1 | tee "$(log_file "eval_rise_${STUDENT}")"
else
  echo "skipping RISE: $RISE_DIR does not exist"
fi

if [ -d "$VL2V_DIR" ]; then
  echo "--- VL2V-ADiP: $VL2V_DIR"
  python scripts/evaluate_vl2v_adip.py \
    --run-dir "$VL2V_DIR" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --gpu-index "$GPU" \
    2>&1 | tee "$(log_file "eval_vl2v_${STUDENT}")"
else
  echo "skipping VL2V-ADiP: $VL2V_DIR does not exist"
fi
