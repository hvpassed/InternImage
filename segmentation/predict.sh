python scripts/predict_single.py \
  --config configs/mask2former_internimage_h_512_80k_custom5cls.py \
  --checkpoint predict_dir/models/loveda/best_mIoU_iter_36000.pth \
  --img predict_dir/True_World.JPG \
  --out predict_dir/out/demo_out \
  --device cuda:0 \
  --opacity 0.12