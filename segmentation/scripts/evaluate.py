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
sys.path.insert(0, str(Path(__file__).parent.parent / 'mmcv_custom'))

# 触发自定义模块注册（模型/优化器/Assigners 等）
try:
    import mmseg_custom
    import mmseg_custom.models
    import mmseg_custom.core
    import mmcv_custom
except ImportError as e:
    print(f"❌ 无法导入自定义模块: {e}")
    print(f"Sys Path: {sys.path}")
    sys.exit(1)


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
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='分布式启动器')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0],
                        help='单机评估时使用的 GPU ids')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    params = load_params()
    
    from mmcv import Config
    from mmseg.apis import single_gpu_test, multi_gpu_test
    from mmseg.datasets import build_dataloader, build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint, init_dist, get_dist_info
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    
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
    
    # 初始化分布式环境
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    distributed = args.launcher != 'none'
    if distributed:
        dist_params = cfg.get('dist_params', dict(backend='nccl'))
        init_dist(args.launcher, **dist_params)

    # 构建数据集
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=distributed,
        shuffle=False)
    
    # 构建模型
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        results = multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False)
        rank, _ = get_dist_info()
    else:
        model = MMDataParallel(model, device_ids=args.gpu_ids)
        results = single_gpu_test(model, data_loader)
        rank = 0
    
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
    
    if rank == 0:
        with open(eval_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # 打印结果
    if rank == 0:
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
