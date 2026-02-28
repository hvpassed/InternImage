#!/usr/bin/env python3
"""
Predict segmentation probabilities for images listed in imgRef and save to H5.
(Modified to handle missing images by writing zero-probability maps)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import mmcv
import numpy as np
import pandas as pd
import torch
import yaml
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose
from tqdm import tqdm

# Add custom modules to path and register
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "mmseg_custom"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mmcv_custom"))
try:
    import mmseg_custom  # noqa: F401
    import mmseg_custom.models  # noqa: F401
    import mmseg_custom.core  # noqa: F401
    import mmcv_custom  # noqa: F401
except ImportError as e:
    print(f"❌ 无法导入自定义模块: {e}")
    print(f"Sys Path: {sys.path}")
    raise


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        if isinstance(results["img"], str):
            results["filename"] = results["img"]
            results["ori_filename"] = results["img"]
        else:
            results["filename"] = None
            results["ori_filename"] = None
        img = mmcv.imread(results["img"])
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


def unwrap_img_metas(img_metas):
    if isinstance(img_metas, list):
        return img_metas[0]
    if hasattr(img_metas, "data"):
        return img_metas.data[0]
    return img_metas


def build_data(model, img_path):
    cfg = model.cfg
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    data = dict(img=img_path)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [next(model.parameters()).device])[0]
        img_metas = unwrap_img_metas(data["img_metas"])
    else:
        data["img_metas"] = [i.data[0] for i in data["img_metas"]]
        img_metas = data["img_metas"][0]
    img_tensor = data["img"]
    if isinstance(img_tensor, (list, tuple)):
        img_tensor = img_tensor[0]
    return img_tensor, img_metas


def load_params(params_path):
    if not params_path:
        return {}
    if not os.path.exists(params_path):
        return {}
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def resolve_image_path(name, img_dir):
    name_path = Path(name)
    if name_path.is_absolute() and name_path.exists():
        return name_path
    candidate = img_dir / name_path
    if candidate.exists():
        return candidate
    candidate = img_dir / name_path.name
    if candidate.exists():
        return candidate
    return None


def load_imgref(imgref_path):
    # 尝试自动检测分隔符
    try:
        df = pd.read_csv(imgref_path, sep=None, engine="python")
    except Exception:
        # 如果第一行读取失败，可能是因为 OpenMVS 有时会生成带逗号的 CSV
        df = pd.read_csv(imgref_path, sep=',')

    columns = {c.lower(): c for c in df.columns}
    
    # 查找列名
    if "name" in columns and "id" in columns:
        name_col = columns["name"]
        id_col = columns["id"]
        # 尝试查找 width/height, 如果没有可能在 imgRef 里叫 W/H 或者 Width/Height
        width_col = columns.get("width", columns.get("w"))
        height_col = columns.get("height", columns.get("h"))
        return df, name_col, id_col, width_col, height_col

    # 如果没有表头，按索引读取
    df = pd.read_csv(imgref_path, sep=None, engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError("imgRef.txt 至少需要两列: Name 和 ID")
    name_col = 0
    id_col = 1
    width_col = 2 if df.shape[1] > 2 else None
    height_col = 3 if df.shape[1] > 3 else None
    return df, name_col, id_col, width_col, height_col


def write_probabilities_h5(model, img_dir, imgref_path, out_h5_path, out_label_dir, opacity):
    df, name_col, id_col, width_col, height_col = load_imgref(imgref_path)
    img_dir = Path(img_dir)
    out_h5_path = Path(out_h5_path)
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    if out_label_dir:
        out_label_dir = Path(out_label_dir)
        out_label_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "module"):
        vis_model = model.module
    else:
        vis_model = model

    # 尝试获取类别数用于生成全黑占位符
    try:
        num_classes = vis_model.decode_head.num_classes
    except AttributeError:
        if hasattr(model, "CLASSES") and model.CLASSES is not None:
             num_classes = len(model.CLASSES)
        else:
             num_classes = model.cfg.model.decode_head.num_classes

    with h5py.File(str(out_h5_path), "w") as h5f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_name = row[name_col]
            image_id = row[id_col]
            img_path = resolve_image_path(str(img_name), img_dir)
            
            # --- 1. 处理图片缺失 ---
            if img_path is None:
                print(f"⚠️ 找不到图片: {img_name} (ID: {image_id}) -> 生成全黑预测占位")
                if width_col is not None and height_col is not None:
                    width = int(row[width_col])
                    height = int(row[height_col])
                    prob = np.zeros((num_classes, height, width), dtype=np.float32)
                    h5f.create_dataset(str(image_id), data=prob, compression="gzip")
                else:
                    print(f"❌ 错误: ID {image_id} 图片缺失且 imgRef 中没有宽高信息，跳过。")
                continue

            # --- 2. 推理 ---
            img_tensor, img_metas = build_data(model, str(img_path))
            with torch.no_grad():
                seg_logit = model.encode_decode(img_tensor, [img_metas])
                # 调整维度
                if seg_logit.dim() == 2:
                    seg_logit = seg_logit.unsqueeze(0).unsqueeze(0)
                elif seg_logit.dim() == 3:
                    seg_logit = seg_logit.unsqueeze(0)
                prob = torch.softmax(seg_logit, dim=1)[0].cpu().numpy()
                if prob.ndim == 2:
                    prob = prob[None, ...]

            # --- 3. 缩放概率图 (关键：使用 imgRef 的宽高) ---
            target_width, target_height = None, None
            if width_col is not None and height_col is not None:
                target_width = int(row[width_col])
                target_height = int(row[height_col])
                
                # 将预测概率图 resize 到 ref 指定的大小 (例如 960x540)
                resized = [
                    cv2.resize(channel, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                    for channel in prob
                ]
                prob = np.stack(resized, axis=0)

            # 保存 H5 (此时 prob 已经是 ref 的形状了)
            h5f.create_dataset(str(image_id), data=prob, compression="gzip")

            # --- 4. 生成可视化 (Overlay) ---
            if out_label_dir:
                # 生成 Label (Mask)
                pred = prob.argmax(axis=0).astype(np.uint8)
                label_path = out_label_dir / f"{image_id}_label.png"
                cv2.imwrite(str(label_path), pred)

                palette = getattr(model, "PALETTE", None)
                if palette is not None:
                    # 【修复的核心逻辑在这】
                    # 读取原始底图
                    img_show = mmcv.imread(str(img_path))
                    
                    # 如果有指定宽高，必须把底图也缩放，否则底图(1920)和Mask(960)无法叠加
                    if target_width is not None and target_height is not None:
                        img_show = cv2.resize(img_show, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                    
                    # 传入【缩放后的底图数组】，而不是路径
                    overlay = vis_model.show_result(
                        img_show,   # 这里传 numpy 数组，不要传路径字符串
                        [pred],
                        palette=palette,
                        show=False,
                        opacity=opacity,
                    )
                    overlay_path = out_label_dir / f"{image_id}_overlay.png"
                    cv2.imwrite(str(overlay_path), overlay)
                else:
                    pass

def main():
    parser = argparse.ArgumentParser(description="Predict probabilities and save to H5")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth")
    parser.add_argument("--img-dir", required=True, help="Input image directory")
    parser.add_argument("--imgref", required=True, help="imgRef.txt path")
    parser.add_argument("--out-h5", required=True, help="Output h5 path")
    parser.add_argument("--out-label-dir", default=None, help="Output label/overlay dir")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--opacity", type=float, default=0.5)
    parser.add_argument("--params", default="params.yaml", help="Params yaml")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        print(f"❌ 输入目录不存在: {img_dir}")
        return

    if not os.path.exists(args.imgref):
        print(f"❌ imgRef 不存在: {args.imgref}")
        return

    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    params = load_params(args.params)
    palette = params.get("palette", getattr(model, "PALETTE", None))
    classes = params.get("classes", getattr(model, "CLASSES", None))
    if isinstance(checkpoint, dict):
        meta = checkpoint.get("meta", {})
        if "CLASSES" in meta:
            classes = meta["CLASSES"]
        if "PALETTE" in meta:
            palette = meta["PALETTE"]
    if classes is not None:
        model.CLASSES = classes
    if palette is not None:
        model.PALETTE = palette

    print(f"📂 使用 imgRef: {args.imgref}")
    print(f"💾 概率将保存至: {args.out_h5}")
    write_probabilities_h5(
        model,
        img_dir,
        args.imgref,
        args.out_h5,
        args.out_label_dir,
        args.opacity,
    )
    print("✅ 概率保存完成。")


if __name__ == "__main__":
    main()