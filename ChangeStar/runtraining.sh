#!/bin/bash

FINETUNE_MODEL_DIR='./log/finetune-SYSUCD/r50_farseg_changestar'

# The corrected fine-tuning command (removed --resume_from)
torchrun \
    --nproc_per_node 1 \
    train_sup_change.py \
    --config_path configs/sysucd/r50_farseg_changestar_finetune.py \
    --model_dir "$FINETUNE_MODEL_DIR"