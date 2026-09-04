#!/usr/bin/env bash
# Shared settings for every experiment script.
#
# Override anything from the command line, e.g.
#   VLM=quiltnet K=3 GPU=1 bash experiments/01_domain_prompts.sh
set -euo pipefail

# Run from the repository root regardless of where the script was invoked.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- experiment knobs -------------------------------------------------------
VLM="${VLM:-plip}"                 # plip | quiltnet
DATASET="${DATASET:-camelyon17}"   # camelyon17 | kather19
K="${K:-4}"                        # number of learnable context tokens
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"

# ---- paths (all relative to the repository root) ----------------------------
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "$LOG_DIR"

timestamp() { date +'%Y-%m-%d_%H-%M-%S'; }

log_file() {  # log_file <name>
  echo "${LOG_DIR}/$1_${VLM}_k${K}_$(timestamp).log"
}

banner() {
  echo "============================================================"
  echo "$*"
  echo "  dataset=${DATASET}  vlm=${VLM}  k=${K}  gpu=${GPU}"
  echo "============================================================"
}
