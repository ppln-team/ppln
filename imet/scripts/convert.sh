#!/usr/bin/env bash

DATA_PATH=/ppln/data/imet/

PYTHONPATH=/ppln python /ppln/imet/convert.py \
    --labels_path=${DATA_PATH}/labels.csv \
    --annotation_path=${DATA_PATH}/train.csv \
    --output_labels_path=${DATA_PATH}/sub_labels.csv \
    --output_annotation_path=${DATA_PATH}/sub_train.csv
