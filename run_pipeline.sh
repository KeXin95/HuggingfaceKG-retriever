#!/bin/bash
# run_pipeline.sh

set -e

echo "Starting the full data processing pipeline..."

# --- DEFINE THE UNIQUE RUN DIRECTORY ONCE ---
BASE_DIR="./experiment_runs"
RUN_ID="run_$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$BASE_DIR/$RUN_ID"
# ---------------------------------------------

echo "All outputs will be saved to: $RUN_DIR"

# STAGE 1: Pass the run directory to the script
python -m scripts.build_base_data --run_dir "$RUN_DIR"

# STAGE 2: Pass the SAME run directory to the next script
python -m scripts.generate_features --run_dir "$RUN_DIR"

# STAGE 3: And again, pass the SAME run directory
python -m scripts.build_graph --run_dir "$RUN_DIR" --split_strategy time --remove_isolated --isolated_strategy connected_only

echo "Pipeline finished successfully!"

echo "Start training GCN with CogDL"

nohup python tune_huggingface.py \
    --data_dir "${RUN_DIR}" \
    --save_path "${RUN_DIR}/trained_gcn.pt" \
    --models gcn \
    --devices 0 \
    --seeds 0 1 2 2>&1 | tee -a "${RUN_DIR}/training_log.log" &

echo "Training completed successfully!"
