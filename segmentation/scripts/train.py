#!/usr/bin/env python3
"""
InternImage + Mask2Former 训练脚本
支持 DVC 参数管理，通过 --dataset 参数切换数据集

用法:
    python scripts/train.py --config <config> --dataset landcover
    python scripts/train.py --config <config> --dataset loveda
    python scripts/train.py --config <config> --dataset combined
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

# 添加 mmseg_custom 到路径 (InternImage 自定义模块)
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'mmseg_custom'))


def load_params():
    """加载 params.yaml"""
    params_file = Path('params.yaml')
    if params_file.exists():
        with open(params_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def update_config_with_params(cfg, params, dataset):
    """根据 params.yaml 和 dataset 更新配置"""
    from mmcv import Config
    
    # 获取数据集路径
    data_root = params['data_root'].get(dataset, f'data/{dataset}')
    
    # 如果是 loveda，使用重映射后的标注目录
    # if dataset == 'loveda':
    #     ann_dir_train = 'ann_dir_remapped/train'
    #     ann_dir_val = 'ann_dir_remapped/val'
    # else:
    #     ann_dir_train = 'ann_dir/train'
    #     ann_dir_val = 'ann_dir/val'
    ann_dir_train = 'ann_dir/train'
    ann_dir_val = 'ann_dir/val'

    # 更新数据配置
    cfg.data.train.data_root = data_root
    cfg.data.train.img_dir = 'img_dir/train'
    cfg.data.train.ann_dir = ann_dir_train
    cfg.data.train.classes = tuple(params['classes'])
    cfg.data.train.palette = params['palette']
    
    cfg.data.val.data_root = data_root
    cfg.data.val.img_dir = 'img_dir/val'
    cfg.data.val.ann_dir = ann_dir_val
    cfg.data.val.classes = tuple(params['classes'])
    cfg.data.val.palette = params['palette']
    
    cfg.data.test.data_root = data_root
    cfg.data.test.img_dir = 'img_dir/val'
    cfg.data.test.ann_dir = ann_dir_val
    cfg.data.test.classes = tuple(params['classes'])
    cfg.data.test.palette = params['palette']
    
    # 更新训练参数
    train_params = params.get('train', {})
    if 'batch_size' in train_params:
        cfg.data.samples_per_gpu = train_params['batch_size']
    if 'num_workers' in train_params:
        cfg.data.workers_per_gpu = train_params['num_workers']
    
    # 更新优化器
    opt_params = train_params.get('optimizer', {})
    if 'lr' in opt_params:
        cfg.optimizer.lr = opt_params['lr']
    if 'weight_decay' in opt_params:
        cfg.optimizer.weight_decay = opt_params['weight_decay']
    
    # 更新训练迭代次数
    if 'total_iters' in train_params:
        cfg.runner.max_iters = train_params['total_iters']
    if 'val_interval' in train_params:
        cfg.evaluation.interval = train_params['val_interval']
    if 'checkpoint_interval' in train_params:
        cfg.checkpoint_config.interval = train_params['checkpoint_interval']
    
    # 更新 crop_size
    if 'crop_size' in train_params:
        crop_size = tuple(train_params['crop_size'])
        # 更新 pipeline 中的 crop_size
        for item in cfg.data.train.pipeline:
            if item['type'] == 'RandomCrop':
                item['crop_size'] = crop_size
            if item['type'] == 'Pad':
                item['size'] = crop_size
    
    return cfg


def save_metrics(work_dir, metrics):
    """保存指标到 JSON"""
    metrics_file = Path(work_dir) / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='训练 InternImage + Mask2Former')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['landcover', 'loveda', 'combined'],
                       help='数据集名称')
    parser.add_argument('--work-dir', type=str, default=None, help='工作目录')
    parser.add_argument('--resume-from', type=str, default=None, help='恢复训练的checkpoint')
    parser.add_argument('--no-validate', action='store_true', help='不进行验证')
    parser.add_argument('--gpus', type=int, default=1, help='GPU数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 加载参数
    params = load_params()
    
    # 导入 mmcv 和 mmseg
    from mmcv import Config
    from mmcv.runner import init_dist, set_random_seed
    from mmseg.apis import train_segmentor
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmseg.utils import get_root_logger
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 用 params.yaml 更新配置
    cfg = update_config_with_params(cfg, params, args.dataset)
    
    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = f"work_dirs/{args.dataset}"
    
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    
    # 恢复训练
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    # 设置随机种子
    set_random_seed(args.seed, deterministic=False)
    cfg.seed = args.seed
    
    # 保存实验信息
    experiment_info = {
        'dataset': args.dataset,
        'config': args.config,
        'data_root': params['data_root'].get(args.dataset),
        'start_time': datetime.now().isoformat(),
        'params': params
    }
    with open(Path(cfg.work_dir) / 'experiment_info.json', 'w') as f:
        json.dump(experiment_info, f, indent=2, default=str)
    
    # 打印配置
    print(f"\n{'='*60}")
    print(f"训练配置:")
    print(f"  数据集: {args.dataset}")
    print(f"  数据路径: {params['data_root'].get(args.dataset)}")
    print(f"  工作目录: {cfg.work_dir}")
    print(f"  类别数: {params['num_classes']}")
    print(f"  类别: {params['classes']}")
    print(f"{'='*60}\n")
    
    # 构建模型
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    
    # 训练
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=(not args.no_validate),
        meta=dict())
    
    # 保存最终指标
    # 查找 best checkpoint
    best_ckpt = list(Path(cfg.work_dir).glob('best_mIoU_iter_*.pth'))
    if best_ckpt:
        import shutil
        shutil.copy(best_ckpt[0], Path(cfg.work_dir) / 'best_mIoU.pth')
    
    print(f"\n训练完成! 结果保存在: {cfg.work_dir}")


if __name__ == '__main__':
    main()
