#!/bin/bash

# Define variables
SESSION_NAME="evaluate_prompt_1"
CONDA_ENV="dg"
SCRIPT_PATH="./evaluate_prompts.py"
CHECKPOINT_PATH="./output_kg/diff_val/2/2025-02-20_01-57-52/last_prompt_learner.pth"
GPU_INDEX=1
NUM_CONTEXT_TOKENS=4



# Create a new tmux session
tmux new-session -d -s $SESSION_NAME

# Send commands to the tmux session
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME "python $SCRIPT_PATH --gpu_index $GPU_INDEX --checkpoint_path $CHECKPOINT_PATH --num_context_tokens $NUM_CONTEXT_TOKENS" C-m

# Attach to the tmux session (optional)
echo "Experiment started in tmux session: $SESSION_NAME"
echo "To attach to the session, run: tmux attach -t $SESSION_NAME"
echo "To detach from the session, press Ctrl-b d"

