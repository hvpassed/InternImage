import os
from pathlib import Path

# ================= 配置区域 =================
# 你的数据根目录 (已经分好家的路径)
DATA_ROOT = "/data/home/cwk/workplace/InternImage/segmentation/data/loveDA"

# 定义文件夹名称结构
# MMSegmentation 标准通常是 img_dir/{split} 和 ann_dir/{split}
STRUCTURE = {
    "train": {"img": "img_dir/train", "ann": "ann_dir/train"},
    "val":   {"img": "img_dir/val",   "ann": "ann_dir/val"}
}
# ===========================================

def check_integrity(split_name):
    print(f"\n{'='*20} 正在检查: {split_name} 集 {'='*20}")
    
    img_dir = os.path.join(DATA_ROOT, STRUCTURE[split_name]["img"])
    ann_dir = os.path.join(DATA_ROOT, STRUCTURE[split_name]["ann"])
    
    # 1. 检查目录是否存在
    if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
        print(f"❌ 严重错误: 找不到目录！")
        print(f"   - 图片目录: {img_dir} ({'存在' if os.path.exists(img_dir) else '不存在'})")
        print(f"   - 掩码目录: {ann_dir} ({'存在' if os.path.exists(ann_dir) else '不存在'})")
        return

    # 2. 获取所有文件名
    # 假设图片可能是 .jpg 或 .png，掩码肯定是 .png
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    ann_files = set(os.listdir(ann_dir)) # 转为集合，查找更快
    
    print(f"📂 图片数量: {len(img_files)}")
    print(f"📂 掩码数量: {len(ann_files)}")
    
    # 3. 开始一一核对
    missing_masks = []
    
    for img_name in img_files:
        # 获取文件名主体 (去掉后缀)
        stem = Path(img_name).stem
        
        # 对应的 mask 应该叫 {stem}.png
        expected_mask = f"{stem}.png"
        
        if expected_mask not in ann_files:
            missing_masks.append(img_name)
    
    # 4. 输出结果
    if len(missing_masks) == 0:
        print(f"✅ {split_name} 集校验通过！完美匹配。")
    else:
        print(f"❌ 警告: 发现 {len(missing_masks)} 张图片缺少对应的 Mask！")
        print("   前 5 个缺失的文件:")
        for f in missing_masks[:5]:
            print(f"   - 图片: {f}  ->  缺失 Mask: {Path(f).stem}.png")
        print("   (请检查 ann_dir 中是否存在这些文件，或者文件名是否只有后缀不同)")

if __name__ == "__main__":
    print(f"正在检查数据根目录: {DATA_ROOT}")
    
    # 检查训练集
    check_integrity("train")
    
    # 检查验证集
    check_integrity("val")