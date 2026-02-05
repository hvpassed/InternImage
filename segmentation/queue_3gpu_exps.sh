#!/bin/bash
set -e

# Queue DVC experiments for three datasets with explicit names.
# Run from the segmentation directory.

GPUS=3
BATCH=2
CROP="[416,416]"
BASE_LR=4e-5

queue_base() {
    local dataset=$1
    dvc exp run --queue \
        --name "${dataset}_base_g${GPUS}_bs${BATCH}_cs416" \
        -S dataset=${dataset} \
        -S train.gpus=${GPUS} \
        -S train.batch_size=${BATCH} \
        -S "train.crop_size=${CROP}" \
        -S train.optimizer.lr=${BASE_LR}
}

queue_acc() {
    local dataset=$1
    dvc exp run --queue \
        --name "${dataset}_lr2e-5_g${GPUS}_bs${BATCH}_cs416" \
        -S dataset=${dataset} \
        -S train.gpus=${GPUS} \
        -S train.batch_size=${BATCH} \
        -S "train.crop_size=${CROP}" \
        -S train.optimizer.lr=2e-5

    dvc exp run --queue \
        --name "${dataset}_nq100_g${GPUS}_bs${BATCH}_cs416" \
        -S dataset=${dataset} \
        -S train.gpus=${GPUS} \
        -S train.batch_size=${BATCH} \
        -S "train.crop_size=${CROP}" \
        -S model.num_queries=100
}

queue_base landcover
queue_base loveda
queue_base combined

queue_acc landcover
queue_acc loveda
queue_acc combined

echo "Queued experiments. Start with: dvc queue start"
