import os
import shutil
from tqdm import tqdm

# ================= 配置区域 =================
# 根目录 (基于你当前脚本所在位置)
ROOT_DIR = "segmentation/data"

# 源数据集名称 (注意大小写，必须和你文件夹里的一致)
SRC_DATASETS = ["landcover", "loveDA"] 

# 目标数据集名称
TARGET_DIR = os.path.join(ROOT_DIR, "combined")
# ===========================================

def merge_datasets():
    # 1. 准备目标目录结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(TARGET_DIR, "img_dir", split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, "ann_dir", split), exist_ok=True)

    print(f"🚀 开始合并数据到: {TARGET_DIR}")

    # 2. 遍历源数据集
    for ds_name in SRC_DATASETS:
        ds_path = os.path.join(ROOT_DIR, ds_name)
        
        if not os.path.exists(ds_path):
            print(f"❌ 错误: 找不到源数据集 {ds_path}，请检查文件夹名称大小写！")
            continue

        print(f"📦 正在合并: {ds_name} ...")
        
        # 遍历 train 和 val
        for split in ["train", "val"]:
            # 源路径
            src_img_dir = os.path.join(ds_path, "img_dir", split)
            src_ann_dir = os.path.join(ds_path, "ann_dir", split)
            
            # 目标路径
            dst_img_dir = os.path.join(TARGET_DIR, "img_dir", split)
            dst_ann_dir = os.path.join(TARGET_DIR, "ann_dir", split)

            if not os.path.exists(src_img_dir): continue

            # 复制文件
            files = os.listdir(src_img_dir)
            for f in tqdm(files, desc=f"   - {split}集"):
                # 复制图片
                # 为了防止重名(虽然概率很小)，可以加个前缀，但Mask2Former其实不依赖文件名唯一性，只要img和ann对应即可
                # 这里我们直接复制，因为LandCover和LoveDA文件名风格差异很大，基本不会冲突
                shutil.copy(os.path.join(src_img_dir, f), os.path.join(dst_img_dir, f))
                
                # 复制对应的 Mask
                # 注意: 图片可能是 jpg 或 png，但 Mask 肯定是 png
                # 我们需要找到对应的 mask 文件名
                img_name_stem = os.path.splitext(f)[0]
                mask_name = img_name_stem + ".png"
                
                src_mask_path = os.path.join(src_ann_dir, mask_name)
                if os.path.exists(src_mask_path):
                    shutil.copy(src_mask_path, os.path.join(dst_ann_dir, mask_name))
                else:
                    # LoveDA 的 Test 集可能没有 mask，这是正常的，跳过警告
                    pass

    print(f"\n✅ 合并完成！Combined 数据集已准备就绪。")

if __name__ == "__main__":
    merge_datasets()