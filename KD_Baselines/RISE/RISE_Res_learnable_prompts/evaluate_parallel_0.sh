#!/bin/bash

# Conda environment name
conda_env="dg"

# Directory containing the models
model_dir="./Outputs/Models/kather"

# Output directory for results
output_dir="./Outputs/Results/kather"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# List of models to test
models=(
    "t_0_1_2_3_4v_5_best_model.pth"
)

# GPU index to use
gpu_index=4

# Get a timestamp to append to log filename
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_file="$output_dir/evaluation_log_${timestamp}.txt"

# Tmux session name
session_name="evaluation_learnable_session_0"

# Create a new tmux session
echo "Creating tmux session '$session_name'..."
tmux new-session -d -s "$session_name"

# Prepare the models argument by prefixing each model with the model directory
model_paths=()
for model in "${models[@]}"; do
    model_paths+=("$model_dir/$model")
done

# Send commands to the tmux session
tmux send-keys -t "$session_name" "conda activate $conda_env" C-m
tmux send-keys -t "$session_name" "echo 'Starting evaluation for all models...'" C-m
tmux send-keys -t "$session_name" "python evaluate.py --models ${model_paths[*]} --output_dir $output_dir --gpu_index $gpu_index | tee $log_file" C-m
tmux send-keys -t "$session_name" "echo 'All evaluations completed. Results saved to $output_dir'" C-m

echo "Tmux session '$session_name' created and evaluation started."
echo "To monitor the session, use: tmux attach -t $session_name"
echo "To detach from the session, press Ctrl + b, then d."