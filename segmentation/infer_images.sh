#!/bin/bash

# 设定默认的环境变量
CONFIG_FILE="configs/mask2former_internimage_h_512_80k_custom5cls.py"  # 替换为你的config路径
CHECKPOINT_FILE="work_dirs/combined/best_mIoU.pth"  # 替换为你的checkpoint路径
IMAGE_DIR="exp"  # 替换为你的图片文件夹路径
OUTPUT_DIR="exp/exp_out"  # 输出文件夹路径
DEVICE="cuda:0"  # 默认使用GPU，若使用CPU则为"cpu"
OPACITY=0.5  # 可选的透明度参数
# 执行批量推理
python  scripts/predict_dic.py \
  --config $CONFIG_FILE \
  --checkpoint $CHECKPOINT_FILE\
  --img-dir  $IMAGE_DIR \
  --out $OUTPUT_DIR

echo "All images processed."
