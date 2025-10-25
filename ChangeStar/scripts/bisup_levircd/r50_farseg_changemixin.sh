#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

config_path='configs//levircd/r50_farseg_changestar_finetune.py'
model_dir='./log/bisup-LEVIRCD/r50_farseg_changestar'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9558 scripts/trainxView2/../../train_sup_change.py \
  --config_path=${config_path} \
  --model_dir=${model_dir}