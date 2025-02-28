#!/bin/bash
# run_training.sh
# This script sets up the environment variables for the training script,
# creates a tmux session, activates the conda environment, and runs the training script
# with live logging.

# --- Configuration Variables ---

# TMUX session name
SESSION_NAME="res_lp_training_0"

# Conda environment name (adjust as needed)
CONDA_ENV="dg"

# Path to your Python training script
PYTHON_SCRIPT="./train_rise_res_ad.py"

# --- Parameters for the Python Parser ---
# Define training domains as groups (each group is a comma-separated string)
TRAIN_DOMS=("0,1,2,3,4")

# Define class names
CLASSNAMES=("0" "1" "2" "3" "4" "5" "6" "7" "8")

# Other training parameters
GPU_INDEX=4
EPOCHS=1
BATCH_SIZE=128
LEARNING_RATE=0.0008
DISTILL_WEIGHT=0.3
CLASSIFICATION_WEIGHT=0.4
DISTANCE_WEIGHT=0.3
T=2.0
MODEL_SAVE_PATH="./Outputs/Models/kather"

# --- Log File Configuration ---
LOG_DIR="./training_logs"
mkdir -p "$LOG_DIR"
# Improved timestamp format and include session name in log file name.
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
LOG_FILE="${LOG_DIR}/${SESSION_NAME}_${TIMESTAMP}.log"

# --- Create a New tmux Session ---
tmux new-session -d -s "$SESSION_NAME"

# --- Send Commands to the tmux Session ---

# Load conda (update the path if necessary)
tmux send-keys -t "$SESSION_NAME" "source ~/miniconda3/etc/profile.d/conda.sh" C-m
# Activate the desired conda environment
tmux send-keys -t "$SESSION_NAME" "conda activate $CONDA_ENV" C-m

# Build the command string with all parameters.
# The TRAIN_DOMS and CLASSNAMES arrays expand into separate arguments.
CMD="python $PYTHON_SCRIPT \
--train_doms ${TRAIN_DOMS[@]} \
--classnames ${CLASSNAMES[@]} \
--gpu_index $GPU_INDEX \
--epochs $EPOCHS \
--batch_size $BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--distill_weight $DISTILL_WEIGHT \
--classification_weight $CLASSIFICATION_WEIGHT \
--distance_weight $DISTANCE_WEIGHT \
--T $T \
--model_save_path $MODEL_SAVE_PATH"

# Send the command to tmux, logging both stdout and stderr.
tmux send-keys -t "$SESSION_NAME" "$CMD 2>&1 | tee $LOG_FILE" C-m

# --- Attach to the tmux Session for Live Monitoring ---
tmux attach -t "$SESSION_NAME"
