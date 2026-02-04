# configs/mask2former_internimage_h_512_80k_custom5cls.py
"""
InternImage-H + Mask2Former 配置文件
用于自定义5类语义分割 (应急救灾场景)
基于官方 mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py 修改
"""

_base_ = [
    './_base_/models/mask2former_internimage.py',
    './_base_/datasets/custom_5cls.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_80k.py'
]

# 类别数量
num_classes = 5

# 预训练权重（骨干）
pretrained = "pretrained/internimage_xl_22k_192to384.pth"
load_from = None

# 模型配置
model = dict(
    type='EncoderDecoderMask2Former',
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=192,
        depths=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.,
        drop_path_rate=0.5,
        norm_layer='LN',
        layer_scale=None,
        offset_scale=1.0,
        post_norm=False,
        dw_kernel_size=5,           # InternImage-H 特有
        res_post_norm=True,          # InternImage-H 特有
        level2_post_norm=True,       # InternImage-H 特有
        level2_post_norm_block_ids=[5, 11, 17, 23, 29],  # InternImage-H 特有
        center_feature_scale=True,   # InternImage-H 特有
        with_cp=False,               # 设为 True 可节省显存
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        feat_channels=1024,
        out_channels=1024,
        num_classes=num_classes,
        num_queries=100,  # 减少queries数量，因为只有5类
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=1024,
                        num_heads=32,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=1024,
                        feedforward_channels=4096,
                        num_fcs=2,
                        ffn_drop=0.0,
                        with_cp=False,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=512, normalize=True),
            init_cfg=None),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=512, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=1024,
                    num_heads=32,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=1024,
                    feedforward_channels=4096,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    with_cp=False,
                    add_identity=True),
                feedforward_channels=4096,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]  # 5类 + 背景
        )
    ),
    test_cfg=dict(mode='whole'))

# 图像配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

# 训练 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
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

# 优化器 - 使用 Layer Decay
optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=0.00002,  # 微调时使用较小学习率
    betas=(0.9, 0.999), 
    weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=39, 
        layer_decay_rate=0.95,
        depths=[5, 5, 24, 5], 
        offset_lr_scale=1.0))

# 学习率配置
lr_config = dict(
    _delete_=True, 
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0, 
    min_lr=0.0, 
    by_epoch=False)

# 数据配置 - 会被训练脚本覆盖
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# 运行配置
runner = dict(type='IterBasedRunner', max_iters=80000)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
