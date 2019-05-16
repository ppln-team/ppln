#!/usr/bin/env bash

PYTHONPATH=/ppln CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=$2 /ppln/imet/train.py $1 ${@:3}
