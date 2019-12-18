#!/usr/bin/env bash

MODEL=$1
NUM_GPUS=$2

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=/ppln:/ppln/examples \
python -m torch.distributed.launch --nproc_per_node="${NUM_GPUS}" "${MODEL}"/train.py "${MODEL}"/config.py
