# configs/_base_/default_runtime.py
"""
默认运行时配置 (MMSeg v0.27.0)
"""

# 运行时设置
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# 随机种子
cudnn_benchmark = True
