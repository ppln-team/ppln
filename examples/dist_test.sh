#!/usr/bin/env bash

MODEL=$1
NUM_GPUS=$2

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/ppln:/ppln/examples \
python -m torch.distributed.launch --nproc_per_node="${NUM_GPUS}" "${MODEL}"/test.py "${MODEL}"/config.py \
    --checkpoint=/data/demo/best.pth \
    --out=/data/demo/predictions.pkl
