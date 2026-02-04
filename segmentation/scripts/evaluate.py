#!/usr/bin/env python3
"""
模型评估脚本
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'mmseg_custom'))


def load_params():
    """加载 params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='评估模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['landcover', 'loveda', 'combined'])
    parser.add_argument('--out', type=str, default=None, help='输出目录')
    args = parser.parse_args()
    
    params = load_params()
    
    from mmcv import Config
    from mmseg.apis import single_gpu_test
    from mmseg.datasets import build_dataloader, build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint
    from mmcv.parallel import MMDataParallel
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 更新数据路径
    data_root = params['data_root'].get(args.dataset)
    ann_dir = 'ann_dir_remapped/val' if args.dataset == 'loveda' else 'ann_dir/val'
    
    cfg.data.test.data_root = data_root
    cfg.data.test.img_dir = 'img_dir/val'
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.classes = tuple(params['classes'])
    cfg.data.test.palette = params['palette']
    
    # 构建数据集
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False)
    
    # 构建模型
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    
    # 评估
    results = single_gpu_test(model, data_loader)
    
    # 计算指标
    eval_results = dataset.evaluate(results, metric='mIoU')
    
    # 保存结果
    output_dir = args.out or Path(args.checkpoint).parent
    eval_file = Path(output_dir) / 'eval_metrics.json'
    
    metrics = {
        'dataset': args.dataset,
        'checkpoint': args.checkpoint,
        'mIoU': eval_results.get('mIoU', 0),
        'mAcc': eval_results.get('mAcc', 0),
        'aAcc': eval_results.get('aAcc', 0),
        'class_IoU': {params['classes'][i]: eval_results.get(f'IoU.{params["classes"][i]}', 0) 
                     for i in range(params['num_classes'])}
    }
    
    with open(eval_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"评估结果 - {args.dataset}")
    print(f"{'='*60}")
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"mAcc: {metrics['mAcc']:.4f}")
    print(f"aAcc: {metrics['aAcc']:.4f}")
    print(f"\n类别 IoU:")
    for cls_name, iou in metrics['class_IoU'].items():
        print(f"  {cls_name}: {iou:.4f}")
    print(f"\n结果已保存: {eval_file}")


if __name__ == '__main__':
    main()
