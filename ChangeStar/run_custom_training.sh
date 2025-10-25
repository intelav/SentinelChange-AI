#!/bin/bash

# Directory for the new fine-tuned model
FINETUNE_MODEL_DIR='./log/finetune-CUSTOM-FINAL/r50_farseg_changestar'

# Ensure the directory exists
mkdir -p "$FINETUNE_MODEL_DIR"

# Run training with the new custom configuration file
# This assumes you have a machine with at least one GPU.
torchrun \
    --nproc_per_node 1 \
    train_sup_change.py \
    --config_path configs/custom/finetune_custom_final.py \
    --model_dir "$FINETUNE_MODEL_DIR"

echo "Custom fine-tuning process started. Monitor the log file in the directory above for progress."

