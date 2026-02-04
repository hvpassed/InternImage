# configs/_base_/schedules/schedule_80k.py
"""
80K iterations 训练调度配置
"""

# 优化器
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# 学习率配置
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# 运行器
runner = dict(type='IterBasedRunner', max_iters=80000)

# Checkpoint
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)

# 评估
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')

# 日志
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
