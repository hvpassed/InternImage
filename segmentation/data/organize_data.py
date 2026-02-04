import os
import shutil
from tqdm import tqdm

# ================= 配置 =================
# 你截图里的原始数据路径 (混杂了jpg和png的文件夹)
SOURCE_TRAIN = "/data/home/cwk/workplace/InternImage/segmentation/data/train/imgs"
SOURCE_VAL   = "/data/home/cwk/workplace/InternImage/segmentation/data/val/imgs"

# 目标路径 (MMSegmentation 标准结构)
TARGET_ROOT = "data/landcover"
# =======================================

def move_files(src_dir, split):
    # 创建目标文件夹
    img_dest = os.path.join(TARGET_ROOT, "img_dir", split)
    ann_dest = os.path.join(TARGET_ROOT, "ann_dir", split)
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(ann_dest, exist_ok=True)
    
    print(f"正在整理 {split} 集...")
    files = os.listdir(src_dir)
    
    for f in tqdm(files):
        src_path = os.path.join(src_dir, f)
        
        if f.endswith('.jpg'):
            # 移动图片 -> img_dir
            shutil.copy(src_path, os.path.join(img_dest, f))
            
        elif f.endswith('.png'):
            # 移动 Mask -> ann_dir
            shutil.copy(src_path, os.path.join(ann_dest, f))

if __name__ == "__main__":
    # 1. 整理训练集
    if os.path.exists(SOURCE_TRAIN):
        move_files(SOURCE_TRAIN, "train")
    else:
        print(f"错误: 找不到源目录 {SOURCE_TRAIN}")

    # 2. 整理验证集
    if os.path.exists(SOURCE_VAL):
        move_files(SOURCE_VAL, "val")
    else:
        print(f"错误: 找不到源目录 {SOURCE_VAL}")
        
    print("\n✅ 整理完成！")
    print(f"数据已准备好，位于: {os.path.abspath(TARGET_ROOT)}")