#!/usr/bin/env bash
# Zero-shot inference for every VLM (PLIP and QuiltNet) on every domain.
#
# Usage:  bash experiments/00_zero_shot.sh
#         VLM=plip PROMPT_SOURCES="template handcrafted dipt" K=4 bash experiments/00_zero_shot.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Both default to every registered option; override to narrow the sweep.
VLMS="${VLMS:-all}"
PROMPT_SOURCES="${PROMPT_SOURCES:-template handcrafted}"

banner "Zero-shot inference (vlms=${VLMS}, prompts=${PROMPT_SOURCES})"

python scripts/evaluate_zero_shot.py   --vlm $VLMS   --prompt-source $PROMPT_SOURCES   --dataset "$DATASET"   --num-context-tokens "$K"   --data-root "$DATA_ROOT"   --output-root "$OUTPUT_ROOT"   --batch-size "$BATCH_SIZE"   --num-workers "$NUM_WORKERS"   --gpu-index "$GPU"   2>&1 | tee "$(log_file zero_shot)"
