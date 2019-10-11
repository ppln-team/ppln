#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/ppln:/ppln/examples \
python -m torch.distributed.launch --nproc_per_node=$2 train.py $1 --launcher pytorch ${@:3}
