# configs/_base_/datasets/custom_5cls.py
"""
自定义5类语义分割数据集配置
类别: background(0), building(1), woodland(2), water(3), road(4)
"""

# 数据集类型
dataset_type = 'CustomDataset'

# 数据根目录 - 会被训练脚本动态覆盖
data_root = 'data/landcover'

# 类别信息
classes = ('background', 'building', 'woodland', 'water', 'road')
palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# 裁剪尺寸
crop_size = (512, 512)

# 训练数据增强 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),  # 不reduce，保持原标签
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),  # Mask2Former 需要
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
]

# 测试 pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# 数据加载配置
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette))
