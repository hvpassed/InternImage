#!/bin/bash

# ==================== 路径与基础配置 ====================
CONFIG_FILE="configs/mask2former_internimage_h_512_80k_custom5cls.py"  # config路径
CHECKPOINT_FILE="work_dirs/combined/best_mIoU.pth"                     # checkpoint路径
IMAGE_DIR="exp_large/images"                                                 # 图片文件夹路径
REF_DIR="exp_large/imgRef"                                                   # 参考图像文件夹路径
OUTPUT_DIR="exp_large/pro"                                                   # 输出文件夹路径
LABEL_DIR="exp_large/labels"                                                 # 标签可视化输出路径
DEVICE="cuda:0"                                                        # 默认使用GPU
OPACITY=0.5                                                            # 可视化透明度参数

# ==================== 执行 Python 脚本 ====================
# 注意：假设你将新的 python 代码保存为了 scripts/predict_prob_bin.py
python scripts/predict_prob_bin.py \
  --config ${CONFIG_FILE} \
  --checkpoint ${CHECKPOINT_FILE} \
  --img-dir ${IMAGE_DIR} \
  --imgref ${REF_DIR}/imgRef.txt \
  --out-bin-dir ${OUTPUT_DIR}/bins \
  --device ${DEVICE} \
  --save-dtype float16 \
  --batch-size 12 \
  --gpu-resize \
  --preprocess-workers 4 \
  --prefetch-size 32 \
  --writer-queue-size 64
  # 如果需要输出可视化预测图，请取消下面两行的注释
  # --out-label-dir ${LABEL_DIR} \
  # --opacity ${OPACITY}