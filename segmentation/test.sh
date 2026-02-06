GPUS=3
BATCH=2
CROP="[416,416]"
BASE_LR=4e-5

queue_base() {
    local dataset=$1
    dvc exp run --queue --force \
        --name "${dataset}_base_g${GPUS}_bs${BATCH}_cs416" \
        -S dataset=${dataset} \
        -S train.gpus=${GPUS} \
        -S train.batch_size=${BATCH} \
        -S "train.crop_size=${CROP}" \
    -S train.optimizer.lr=${BASE_LR}
}

queue_base combined