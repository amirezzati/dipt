#!/bin/bash
# run_experiment.sh
# This script sets up parameters and runs the training main function inside a tmux session

# Default parameters
CLASSNAMES="ADI BACK DEB LYM MUC MUS NORM STR TUM"
DOMAIN_NUM="4"
GPU_INDEX=3
VAL_DOMAIN_NUM="5"
NUM_CONTEXT_TOKENS=4
NUM_EPOCHS=1
LR=5e-5
SCORE_WEIGHT=1.0
EVAL_INTERVAL=50
BATCH_SIZE=128
LOG_DIR="./train_logs/kather"
USE_LEARNABLE_AGG_BOOL="False"
PREFIX="kather"
# TMUX configuration
# Replace spaces in DOMAIN_NUM with underscores for the session name
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")  # Replace colons with hyphens

DOMAIN_NUM_SESSION=$(echo $DOMAIN_NUM | tr ' ' '_')
SESSION_NAME="prompt_session_domain_t${DOMAIN_NUM_SESSION}}"
CONDA_ENV="dg"

# Create log directory if not exists
mkdir -p $LOG_DIR

# Generate timestamp for log file
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

if [ "$USE_LEARNABLE_AGG_BOOL" == "True" ]; then
    USE_LEARNABLE_AGG="--use_learnable_agg"
else
    USE_LEARNABLE_AGG=""
fi

echo "=============================================="
echo "Starting experiment in tmux session: $SESSION_NAME"
echo "Output will be saved to: $LOG_FILE"
echo "To monitor output live, use:"
echo "1. tmux attach -t $SESSION_NAME"
echo "2. tail -f $LOG_FILE"
echo "=============================================="

# Create new tmux session and run the command
tmux new-session -d -s $SESSION_NAME

# Send commands to tmux session
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME "python train_domain_specific_dval.py \
    --classnames $CLASSNAMES \
    --domain_num $DOMAIN_NUM \
    --gpu_index $GPU_INDEX \
    --val_domain_num $VAL_DOMAIN_NUM \
    --num_context_tokens $NUM_CONTEXT_TOKENS \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --score_weight $SCORE_WEIGHT \
    --eval_interval $EVAL_INTERVAL \
    --batch_size $BATCH_SIZE \
    --prefix $PREFIX \
    $USE_LEARNABLE_AGG 2>&1 | tee $LOG_FILE" C-m

echo "Experiment started in tmux session: $SESSION_NAME"
echo "To detach from tmux session: Ctrl-b d"
echo "To reconnect: tmux attach -t $SESSION_NAME"

tmux a -t $SESSION_NAME


