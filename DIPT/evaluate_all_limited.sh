#!/bin/bash

# Define variables
SESSION_NAME_PREFIX="evaluate_prompt"
CONDA_ENV="dg"
SCRIPT_PATH="./evaluate_prompts.py"
BASE_CHECKPOINT_DIR="./output_kg"
GPU_INDEX=3
NUM_CONTEXT_TOKENS=4
MAX_CONCURRENT_JOBS=12  # Set this to the number of GPUs you have
JOB_TRACKING_DIR="./job_tracking"

# Create job tracking directory
mkdir -p "$JOB_TRACKING_DIR"

# Function to check if a job is still running
is_job_running() {
    local session_name=$1
    tmux has-session -t "$session_name" 2>/dev/null && pgrep -f "python $SCRIPT_PATH" >/dev/null
}

# Loop through all experiment directories
for experiment_dir in "$BASE_CHECKPOINT_DIR"/*/; do
    # Get experiment name (e.g., 1_2_3)
    exp_name=$(basename "$experiment_dir")
    
    # Loop through all date directories in each experiment directory
    for date_dir in "$experiment_dir"*/; do
        # Get date directory name (e.g., 2025-02-17_21-03-20)
        date_name=$(basename "$date_dir")
        
        # Process both best and last checkpoints
        for checkpoint_type in best last; do
            CHECKPOINT_PATH="$date_dir/${checkpoint_type}_prompt_learner.pth"
            
            if [ -f "$CHECKPOINT_PATH" ]; then
                # Wait until there's a slot available
                while [ "$(ls "$JOB_TRACKING_DIR" | wc -l)" -ge "$MAX_CONCURRENT_JOBS" ]; do
                    sleep 60
                    # Clean up finished jobs
                    for job_file in "$JOB_TRACKING_DIR"/*; do
                        session_name=$(basename "$job_file")
                        if ! is_job_running "$session_name"; then
                            rm -f "$job_file"
                        fi
                    done
                done
                
                # Create unique session name
                SESSION_NAME="${SESSION_NAME_PREFIX}_${exp_name}_${date_name}_${checkpoint_type}"
                
                # Create a new tmux session
                tmux new-session -d -s "$SESSION_NAME"
                
                # Send commands to the tmux session
                tmux send-keys -t "$SESSION_NAME" "conda activate $CONDA_ENV" C-m
                tmux send-keys -t "$SESSION_NAME" "python $SCRIPT_PATH --gpu_index $GPU_INDEX --checkpoint_path $CHECKPOINT_PATH --num_context_tokens $NUM_CONTEXT_TOKENS" C-m

                # Create a job tracking file
                touch "$JOB_TRACKING_DIR/$SESSION_NAME"

                # Print session information
                echo "Experiment started in tmux session: $SESSION_NAME"
                echo "Checkpoint used: $CHECKPOINT_PATH"
                echo "To attach: tmux attach -t $SESSION_NAME"
                echo "To detach: Ctrl-b d"
                echo "-----------------------------------------"
            else
                echo "Checkpoint not found: $CHECKPOINT_PATH"
            fi
        done
    done
done

# Clean up job tracking directory
rm -rf "$JOB_TRACKING_DIR"