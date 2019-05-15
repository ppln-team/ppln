#!/usr/bin/env bash

PYTHONPATH=/ppln python -m torch.distributed.launch --nproc_per_node=$2 train_cifar10.py $1 --launcher pytorch ${@:3}
