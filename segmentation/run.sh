#!/bin/bash
# ============================================================================
# InternImage + Mask2Former 训练快速启动脚本
# 
# 用法:
#   bash run.sh prepare <dataset>     # 准备数据
#   bash run.sh train <dataset>       # 训练
#   bash run.sh eval <dataset>        # 评估
#   bash run.sh dvc-train <dataset>   # 使用DVC训练
# ============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

CONFIG="configs/mask2former_internimage_h_512_80k_custom5cls.py"

show_help() {
    echo ""
    echo "用法: bash run.sh <command> <dataset>"
    echo ""
    echo "命令:"
    echo "  prepare <dataset>     准备数据集 (landcover/loveda/combined)"
    echo "  train <dataset>       训练模型"
    echo "  eval <dataset>        评估模型"
    echo "  train-dist <dataset> <gpus>  分布式训练"
    echo "  dvc-train <dataset>   使用DVC训练 (推荐)"
    echo "  dvc-exp               运行DVC实验"
    echo ""
    echo "示例:"
    echo "  bash run.sh prepare loveda"
    echo "  bash run.sh train landcover"
    echo "  bash run.sh dvc-train combined"
    echo ""
}

prepare() {
    local dataset=$1
    echo -e "${GREEN}准备数据集: $dataset${NC}"
    python scripts/prepare_data.py --dataset $dataset
}

train() {
    local dataset=$1
    echo -e "${GREEN}训练模型 - 数据集: $dataset${NC}"
    python scripts/train.py \
        --config $CONFIG \
        --dataset $dataset \
        --work-dir work_dirs/$dataset
}

train_dist() {
    local dataset=$1
    local gpus=$2
    echo -e "${GREEN}分布式训练 - 数据集: $dataset, GPUs: $gpus${NC}"
    
    # 先更新 params.yaml 中的 dataset
    sed -i "s/^dataset:.*/dataset: \"$dataset\"/" params.yaml
    
    bash dist_train.sh $CONFIG $gpus --work-dir work_dirs/$dataset
}

evaluate() {
    local dataset=$1
    echo -e "${GREEN}评估模型 - 数据集: $dataset${NC}"
    python scripts/evaluate.py \
        --config $CONFIG \
        --checkpoint work_dirs/$dataset/best_mIoU.pth \
        --dataset $dataset
}

dvc_train() {
    local dataset=$1
    echo -e "${GREEN}使用DVC训练 - 数据集: $dataset${NC}"
    
    # 更新 params.yaml 中的 dataset 参数
    sed -i "s/^dataset:.*/dataset: \"$dataset\"/" params.yaml
    
    # 运行 DVC pipeline
    dvc repro
}

dvc_exp() {
    echo -e "${GREEN}运行DVC实验${NC}"
    echo "当前参数:"
    grep "^dataset:" params.yaml
    echo ""
    
    # 显示如何运行实验
    echo "运行实验示例:"
    echo "  dvc exp run -S dataset=landcover"
    echo "  dvc exp run -S dataset=loveda"
    echo "  dvc exp run -S dataset=combined"
    echo "  dvc exp run -S train.optimizer.lr=0.00001"
    echo ""
    echo "查看实验结果:"
    echo "  dvc exp show"
}

# 主逻辑
case "${1:-help}" in
    prepare)
        [ -z "$2" ] && echo -e "${RED}请指定数据集${NC}" && exit 1
        prepare $2
        ;;
    train)
        [ -z "$2" ] && echo -e "${RED}请指定数据集${NC}" && exit 1
        train $2
        ;;
    train-dist)
        [ -z "$2" ] && echo -e "${RED}请指定数据集${NC}" && exit 1
        [ -z "$3" ] && echo -e "${RED}请指定GPU数量${NC}" && exit 1
        train_dist $2 $3
        ;;
    eval)
        [ -z "$2" ] && echo -e "${RED}请指定数据集${NC}" && exit 1
        evaluate $2
        ;;
    dvc-train)
        [ -z "$2" ] && echo -e "${RED}请指定数据集${NC}" && exit 1
        dvc_train $2
        ;;
    dvc-exp)
        dvc_exp
        ;;
    help|--help|-h|*)
        show_help
        ;;
esac
