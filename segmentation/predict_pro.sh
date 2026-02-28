CONFIG_FILE="configs/mask2former_internimage_h_512_80k_custom5cls.py"  # 替换为你的config路径
CHECKPOINT_FILE="work_dirs/combined/best_mIoU.pth"  # 替换为你的checkpoint路径
IMAGE_DIR="exp/images"  # 替换为你的图片文件夹路径
REF_DIR="exp/imgRef"  # 替换为你的参考图像文件夹路径
OUTPUT_DIR="exp/pro"  # 输出文件夹路径
LABLE_DIR="exp/labels"
DEVICE="cuda:0"  # 默认使用GPU，若使用CPU则为"cpu"
OPACITY=0.5  # 可选的透明度参数

python scripts/predict_pro.py \
  --config ${CONFIG_FILE} \
  --checkpoint ${CHECKPOINT_FILE} \
  --img-dir ${IMAGE_DIR} \
  --imgref ${REF_DIR}/imgRef.txt \
  --out-h5 ${OUTPUT_DIR}/probabilities.h5 \
  --out-label-dir $LABLE_DIR \
  --opacity 0.5