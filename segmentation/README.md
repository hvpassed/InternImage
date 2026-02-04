# InternImage + Mask2Former 训练方案

## 概述

基于 MMSegmentation v0.27.0 和 DVC 的语义分割训练方案，支持通过参数切换数据集。

**5类分割**: background(0), building(1), woodland(2), water(3), road(4)

## 目录结构

```
segmentation/
├── configs/
│   ├── _base_/
│   │   ├── datasets/custom_5cls.py
│   │   ├── models/mask2former_internimage.py
│   │   ├── schedules/schedule_80k.py
│   │   └── default_runtime.py
│   └── mask2former_internimage_h_512_80k_custom5cls.py  # 主配置
├── data/
│   ├── landcover/{img_dir,ann_dir}/{train,val}/
│   ├── loveDA/{img_dir,ann_dir,ann_dir_remapped}/{train,val}/
│   └── combined/{img_dir,ann_dir}/{train,val}/
├── pretrained/
│   └── mask2former_internimage_h_ade20k.pth
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── params.yaml          # DVC参数 (切换数据集在这里)
├── dvc.yaml             # DVC pipeline
└── run.sh               # 快速启动
```

## 快速开始

### 1. 复制文件到你的 InternImage/segmentation 目录

```bash
# 将以下文件复制到对应位置
cp -r configs/* ~/workplace/InternImage/segmentation/configs/
cp -r scripts/* ~/workplace/InternImage/segmentation/scripts/
cp params.yaml dvc.yaml run.sh ~/workplace/InternImage/segmentation/
```

### 2. 初始化 DVC

```bash
cd ~/workplace/InternImage/segmentation
dvc init
git add .dvc params.yaml dvc.yaml
```

### 3. 准备数据

```bash
# 准备 LoveDA (会自动重映射标签)
bash run.sh prepare loveda

# 准备其他数据集
bash run.sh prepare landcover
bash run.sh prepare combined
```

### 4. 训练

**方法1: 直接训练**
```bash
bash run.sh train landcover
bash run.sh train loveda
bash run.sh train combined
```

**方法2: 使用 DVC (推荐)**
```bash
# 修改 params.yaml 中的 dataset 参数，然后:
bash run.sh dvc-train landcover
```

**方法3: 使用官方脚本分布式训练**
```bash
bash dist_train.sh configs/mask2former_internimage_h_512_80k_custom5cls.py 8
```

### 5. DVC 实验管理

```bash
# 切换数据集训练
dvc exp run -S dataset=landcover
dvc exp run -S dataset=loveda
dvc exp run -S dataset=combined

# 调参实验
dvc exp run -S dataset=landcover -S train.optimizer.lr=0.00001
dvc exp run -S dataset=landcover -S train.batch_size=2

# 查看所有实验
dvc exp show

# 比较实验
dvc exp diff <exp1> <exp2>

# 应用最佳实验
dvc exp apply <best_exp>
```

## LoveDA 标签映射

| 原始标签 | 原始名称 | 新标签 | 新名称 |
|---------|---------|-------|--------|
| 1 | background | 0 | background |
| 2 | building | 1 | building |
| 3 | road | 4 | road |
| 4 | water | 3 | water |
| 5 | barren | 0 | background |
| 6 | forest | 2 | woodland |
| 7 | agriculture | 0 | background |

## 参数配置 (params.yaml)

```yaml
# 切换数据集 - 只需修改这一行
dataset: "landcover"  # landcover / loveda / combined

# 训练参数
train:
  total_iters: 80000
  batch_size: 1
  optimizer:
    lr: 0.00002
```

## 注意事项

1. **预训练权重**: 确保 `pretrained/mask2former_internimage_h_ade20k.pth` 存在

2. **显存不足**: 
   - 减小 `train.batch_size`
   - 在配置中设置 `with_cp=True` 启用 gradient checkpointing

3. **Combined 数据集**: 需要手动合并 LandCover 和 LoveDA 的数据到 `data/combined/`

4. **LoveDA 标签**: 脚本会自动处理标签重映射，结果保存在 `ann_dir_remapped/`
