import yaml
import os

# 1. 读取 params.yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)['train_dataset']

# 2. 打印当前使用的参数 (方便调试)
print(f"🚀 Loading Config with Params:")
print(f"   - Data Root: {params['data_root']}")
print(f"   - LR: {params['lr']}")
print(f"   - Crop Size: {params['crop_size']}")

_base_ = ['./mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py']

# 3. 应用参数
crop_size = (params['crop_size'], params['crop_size'])
data_root = params['data_root']

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, params['crop_size']), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, params['crop_size']),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

optimizer = dict(lr=params['lr'])
runner = dict(max_iters=params['max_iters'])
checkpoint_config = dict(by_epoch=False, interval=params['save_interval'])
evaluation = dict(interval=params['save_interval'], metric='mIoU', pre_eval=True)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset', data_root=data_root, 
        img_dir='img_dir/train', ann_dir='ann_dir/train',
        classes=["background", "building", "woodland", "water", "road"],
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0]],
        pipeline=train_pipeline),
    val=dict(
        type='CustomDataset', data_root=data_root, 
        img_dir='img_dir/val', ann_dir='ann_dir/val',
        classes=["background", "building", "woodland", "water", "road"],
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0]],
        pipeline=test_pipeline),
    test=dict(
        type='CustomDataset', data_root='data/loveDA', 
        img_dir='img_dir/val', ann_dir='ann_dir/val',
        classes=["background", "building", "woodland", "water", "road"],
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0]],
        pipeline=test_pipeline)
)