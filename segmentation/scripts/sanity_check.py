import sys
import os
from pathlib import Path

# 1. 路径设置
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'mmseg_custom'))

# 2. 导入自定义模块
try:
    import mmseg_custom
    print("✅ 成功导入 mmseg_custom")
except ImportError as e:
    print(f"❌ 无法导入 mmseg_custom: {e}")
    sys.exit(1)

import torch
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

def main():
    print("\n🚀 开始环境与数据完整性自检 (Sanity Check)...")
    
    # 配置文件路径
    config_path = 'configs/mask2former_internimage_h_512_80k_custom5cls.py'
    
    # 【关键修正 1】严格匹配你的实际文件夹名称 (注意 loveDA 大写)
    data_root = 'data/loveDA' 
    
    if not os.path.exists(data_root):
        print(f"❌ 错误: 找不到数据根目录: {data_root}")
        print("   请检查文件夹大小写是否完全一致。")
        return

    cfg = Config.fromfile(config_path)
    
    # 【关键修正 2】直接指向你的 ann_dir (不再加 remapped)
    cfg.data.train.data_root = data_root
    cfg.data.train.img_dir = 'img_dir/train'
    cfg.data.train.ann_dir = 'ann_dir/train' 
    
    # 显存优化
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0 
    cfg.model.backbone.with_cp = True 

    print(f"📂 读取路径: {data_root}/{cfg.data.train.ann_dir}")
    
    try:
        # 构建数据集
        dataset = build_dataset(cfg.data.train)
        print(f"✅ 数据集加载成功! 训练集样本数: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ 警告: 样本数为 0！请检查 img_dir/train 里面是否有图片，或者后缀名是否匹配。")
            return

        # 读取样本
        print("🔍 读取第一个样本...")
        data_batch = dataset[0]
        img = data_batch['img'].data
        gt = data_batch['gt_semantic_seg'].data
        
        print(f"   图像: {img.shape}")
        print(f"   标签: {gt.shape}, 唯一值: {torch.unique(gt).tolist()}")
        
        # 简单校验标签是否在 0-4 之间
        if torch.max(gt) > 4 and torch.max(gt) != 255:
             print("⚠️ 警告: 标签中包含大于 4 的值，请确认是否已完成 6类->5类 映射。")
        else:
             print("✅ 标签值范围正常 (0-4)。")

    except Exception as e:
        print(f"❌ 数据集错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 模型测试 (略过不跑，只要数据能读就说明路径对了)
    print("\n🎉 路径检查通过！")

if __name__ == '__main__':
    main()