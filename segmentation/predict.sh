CUDA_VISIBLE_DEVICES=2 python scripts/predict_single.py \
  --config configs/mask2former_internimage_h_512_80k_custom5cls.py \
  --checkpoint work_dirs/combined/best_mIoU.pth\
  --img predict_dir/image.png \
  --out predict_dir/out/demo_out \
  --device cuda:0 \
  --opacity 0.5
  