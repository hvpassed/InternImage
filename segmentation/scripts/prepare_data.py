#!/usr/bin/env python3
"""
数据预处理脚本
- 验证数据完整性
- 处理 LoveDA 标签重映射
- 生成数据统计信息
"""

import os
import argparse
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter


def load_params():
    """加载 params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def remap_loveda_labels(ann_path: Path, output_path: Path, mapping: dict):
    """重映射 LoveDA 标签"""
    ann = np.array(Image.open(ann_path))
    remapped = np.full_like(ann, 255)  # 默认 ignore
    
    for old_label, new_label in mapping.items():
        old_label = int(old_label)
        remapped[ann == old_label] = new_label
    
    Image.fromarray(remapped.astype(np.uint8)).save(output_path)
    return remapped


def validate_dataset(data_dir: Path, dataset_name: str):
    """验证数据集完整性"""
    img_dir = data_dir / 'img_dir'
    ann_dir = data_dir / 'ann_dir'
    
    errors = []
    stats = {'train': {}, 'val': {}}
    
    for split in ['train', 'val']:
        img_split_dir = img_dir / split
        ann_split_dir = ann_dir / split
        
        if not img_split_dir.exists():
            errors.append(f"Missing: {img_split_dir}")
            continue
        if not ann_split_dir.exists():
            errors.append(f"Missing: {ann_split_dir}")
            continue
        
        # 获取文件列表
        img_files = set()
        for ext in ['*.jpg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.PNG']:
            img_files.update(f.stem for f in img_split_dir.glob(ext))
        
        ann_files = set()
        for ext in ['*.png', '*.tif', '*.tiff']:
            ann_files.update(f.stem for f in ann_split_dir.glob(ext))
        
        missing_ann = img_files - ann_files
        missing_img = ann_files - img_files
        
        if missing_ann:
            errors.append(f"{dataset_name}/{split}: {len(missing_ann)} images missing annotations")
        if missing_img:
            errors.append(f"{dataset_name}/{split}: {len(missing_img)} annotations missing images")
        
        stats[split] = {
            'num_images': len(img_files),
            'num_annotations': len(ann_files),
            'matched': len(img_files & ann_files)
        }
    
    return errors, stats


def process_loveda(data_dir: Path, mapping: dict):
    """处理 LoveDA 数据集的标签重映射"""
    ann_dir = data_dir / 'ann_dir'
    remapped_dir = data_dir / 'ann_dir_remapped'
    
    for split in ['train', 'val']:
        split_ann_dir = ann_dir / split
        split_remapped_dir = remapped_dir / split
        
        if not split_ann_dir.exists():
            print(f"  Warning: {split_ann_dir} not found, skipping...")
            continue
        
        split_remapped_dir.mkdir(parents=True, exist_ok=True)
        
        ann_files = list(split_ann_dir.glob('*.png'))
        print(f"  重映射 {split}: {len(ann_files)} 个标注文件...")
        
        for ann_file in tqdm(ann_files, desc=f"  {split}"):
            output_path = split_remapped_dir / ann_file.name
            remap_loveda_labels(ann_file, output_path, mapping)
    
    return remapped_dir


def compute_class_distribution(ann_dir: Path, num_classes: int = 5, use_remapped: bool = False):
    """计算类别分布"""
    if use_remapped:
        target_dir = ann_dir.parent / 'ann_dir_remapped'
    else:
        target_dir = ann_dir
    
    class_pixels = Counter()
    total_pixels = 0
    
    for split in ['train', 'val']:
        split_dir = target_dir / split
        if not split_dir.exists():
            continue
        
        for ann_file in split_dir.glob('*.png'):
            ann = np.array(Image.open(ann_file))
            for c in range(num_classes):
                class_pixels[c] += np.sum(ann == c)
            total_pixels += np.sum(ann != 255)  # 排除 ignore
    
    distribution = {c: class_pixels[c] / total_pixels if total_pixels > 0 else 0 
                   for c in range(num_classes)}
    return distribution


def main():
    parser = argparse.ArgumentParser(description='数据预处理')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['landcover', 'loveda', 'combined'],
                       help='数据集名称')
    args = parser.parse_args()
    
    params = load_params()
    
    # 数据集目录映射
    data_root_map = {
        'landcover': Path(params['data_root']['landcover']),
        'loveda': Path(params['data_root']['loveda']),
        'combined': Path(params['data_root']['combined'])
    }
    
    data_dir = data_root_map[args.dataset]
    
    print(f"\n{'='*60}")
    print(f"处理数据集: {args.dataset}")
    print(f"数据目录: {data_dir}")
    print(f"{'='*60}\n")
    
    # 1. 验证数据
    print("步骤 1: 验证数据完整性...")
    errors, stats = validate_dataset(data_dir, args.dataset)
    
    if errors:
        print("  发现问题:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  ✓ 数据完整性验证通过")
    
    for split, split_stats in stats.items():
        print(f"    {split}: {split_stats['matched']} 对图像-标注")
    
    # 2. LoveDA 特殊处理
    if args.dataset == 'loveda':
        print("\n步骤 2: 处理 LoveDA 标签重映射...")
        mapping = params['loveda_label_mapping']
        print(f"  映射规则: {mapping}")
        remapped_dir = process_loveda(data_dir, mapping)
        print(f"  ✓ 重映射完成: {remapped_dir}")
    
    # 3. 计算类别分布
    print("\n步骤 3: 计算类别分布...")
    use_remapped = (args.dataset == 'loveda')
    distribution = compute_class_distribution(
        data_dir / 'ann_dir', 
        num_classes=params['num_classes'],
        use_remapped=use_remapped
    )
    
    classes = params['classes']
    print("  类别分布:")
    for c, freq in distribution.items():
        print(f"    {classes[c]}: {freq*100:.2f}%")
    
    # 4. 保存统计信息
    stats_file = data_dir / 'dataset_stats.json'
    stats_data = {
        'dataset': args.dataset,
        'splits': stats,
        'class_distribution': {classes[c]: freq for c, freq in distribution.items()},
        'num_classes': params['num_classes'],
        'classes': classes
    }
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    print(f"\n  统计信息已保存: {stats_file}")
    
    # 5. 创建完成标记
    prepared_marker = data_dir / '.prepared'
    prepared_marker.touch()
    print(f"\n✓ 数据预处理完成!")


if __name__ == '__main__':
    main()
