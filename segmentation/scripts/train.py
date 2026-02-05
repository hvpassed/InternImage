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

# 添加自定义模块到路径 (InternImage 自定义模块)
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
    # 更新 img_scale
    if 'img_scale' in train_params:
        img_scale = tuple(train_params['img_scale'])
        for item in cfg.data.train.pipeline:
            if item['type'] == 'Resize':
                item['img_scale'] = img_scale
        for item in cfg.data.test.pipeline:
            if item.get('type') == 'MultiScaleFlipAug':
                item['img_scale'] = img_scale
    gpu_params = params.get('gpu', {})
    if gpu_params.get('fp16', False):
        print("⚡ 已开启 FP16 混合精度训练")
        cfg.optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', interval=50)
    # --- 动态修改 Backbone (关键) ---
    model_params = params.get('model', {})
    if 'backbone' in cfg.model:
        if 'channels' in model_params:
            cfg.model.backbone.channels = model_params['channels']
        if 'depths' in model_params:
            cfg.model.backbone.depths = model_params['depths']
        if 'groups' in model_params:
            cfg.model.backbone.groups = model_params['groups']
        if 'drop_path_rate' in model_params:
            cfg.model.backbone.drop_path_rate = model_params['drop_path_rate']
        if 'num_queries' in model_params and hasattr(cfg.model, 'decode_head'):
            cfg.model.decode_head.num_queries = model_params['num_queries']
        # 【新增】核心修复：强制修改卷积核大小 (H=5, XL=3)
        if 'dw_kernel_size' in model_params:
            cfg.model.backbone.dw_kernel_size = model_params['dw_kernel_size']
            
        if 'with_cp' in model_params:
            cfg.model.backbone.with_cp = bool(model_params['with_cp'])

    # --- 预训练权重 ---
    if 'pretrained' in model_params and model_params['pretrained']:
        ckpt_path = model_params['pretrained']
        if not os.path.exists(ckpt_path):
            print(f"⚠️ 警告: 预训练权重文件不存在: {ckpt_path}")
        else:
            ckpt_path = os.path.abspath(ckpt_path)
            cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=ckpt_path)
            print(f"🔄 注入预训练权重: {ckpt_path}")

    # --- 修复 Head ---
    if hasattr(cfg.model, 'decode_head'):
        cfg.model.decode_head.in_index = [0, 1, 2, 3]

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
    parser.add_argument('--gpus', type=int, default=1, help='GPU数量(非分布式)')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='分布式启动器')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 加载参数
    params = load_params()
    
    # 导入 mmcv 和 mmseg
    from mmcv import Config
    from mmcv.runner import init_dist, set_random_seed, get_dist_info
    from mmseg.apis import train_segmentor
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmseg.utils import get_root_logger
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 用 params.yaml 更新配置
    cfg = update_config_with_params(cfg, params, args.dataset)

    # 兼容 lr 被解析为序列/字符串的情况
    if hasattr(cfg, 'optimizer') and isinstance(cfg.optimizer, dict):
        lr_val = cfg.optimizer.get('lr', None)
        if lr_val is None and isinstance(params, dict):
            lr_val = params.get('train', {}).get('optimizer', {}).get('lr', None)
        if isinstance(lr_val, (list, tuple)) and len(lr_val) > 0:
            lr_val = lr_val[0]
        if isinstance(lr_val, str):
            lr_val = float(lr_val)
        if isinstance(lr_val, (int, float)):
            cfg.optimizer['lr'] = float(lr_val)

    # 兼容 mmcv Fp16OptimizerHook 不接受 interval 参数
    if hasattr(cfg, 'optimizer_config') and isinstance(cfg.optimizer_config, dict):
        cfg.optimizer_config.pop('interval', None)
        cfg.optimizer_config.pop('loss_scale', None)
        if cfg.optimizer_config.get('type') == 'Fp16OptimizerHook':
            cfg.optimizer_config.pop('type', None)

    # 禁用 fp16，避免 ms_deform_attn_forward 半精度报错
    if hasattr(cfg, 'fp16'):
        cfg.fp16 = None
    
    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = f"work_dirs/{args.dataset}"
    
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    
    # 恢复训练
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    # 初始化分布式环境
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    distributed = args.launcher != 'none'
    if distributed:
        dist_params = cfg.get('dist_params', dict(backend='nccl'))
        init_dist(args.launcher, **dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = list(range(world_size))
        cfg.device = 'cuda'
        # Avoid DDP reduction error when some params are unused in a step
        cfg.find_unused_parameters = True
        # Avoid reentrant backward issues with checkpointing under DDP
        if getattr(cfg.model, 'backbone', None) is not None and cfg.model.backbone.get('with_cp', False):
            if rank == 0:
                print("⚠️ DDP 下关闭 gradient checkpointing 以避免重复反传错误")
            cfg.model.backbone.with_cp = False
    else:
        rank = 0
        cfg.gpu_ids = list(range(args.gpus))
        cfg.device = 'cuda' if args.gpus > 0 else 'cpu'

    # 设置随机种子
    seed = args.seed + rank
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed
    
    # 保存实验信息
    experiment_info = {
        'dataset': args.dataset,
        'config': args.config,
        'data_root': params['data_root'].get(args.dataset),
        'start_time': datetime.now().isoformat(),
        'params': params
    }
    if rank == 0:
        with open(Path(cfg.work_dir) / 'experiment_info.json', 'w') as f:
            json.dump(experiment_info, f, indent=2, default=str)
    
    # 打印配置
    if rank == 0:
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
        distributed=distributed,
        validate=(not args.no_validate),
        meta=dict())
    
    # 保存最终指标
    # 查找 best checkpoint
    if rank == 0:
        best_ckpt = list(Path(cfg.work_dir).glob('best_mIoU_iter_*.pth'))
        if best_ckpt:
            import shutil
            shutil.copy(best_ckpt[0], Path(cfg.work_dir) / 'best_mIoU.pth')

        print(f"\n训练完成! 结果保存在: {cfg.work_dir}")


if __name__ == '__main__':
    main()
