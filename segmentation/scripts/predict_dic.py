#!/usr/bin/env python3
"""
Batch image inference with probability output.
Modified to process a directory of images.
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
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.datasets.pipelines import Compose
from tqdm import tqdm

# 添加自定义模块到路径并注册
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


def load_params(params_path):
    if not params_path:
        return {}
    if not os.path.exists(params_path):
        return {}
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


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


def build_legend(palette, classes, box_size=18, pad=6, font_scale=0.5):
    if not palette or not classes:
        return None
    num = min(len(palette), len(classes))
    height = num * (box_size + pad) + pad
    width = 320
    legend = np.full((height, width, 3), 255, dtype=np.uint8)
    for i in range(num):
        color = palette[i]
        # palette is RGB, cv2 uses BGR
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        y = pad + i * (box_size + pad)
        cv2.rectangle(legend, (pad, y), (pad + box_size, y + box_size), bgr, -1)
        cv2.putText(
            legend,
            str(classes[i]),
            (pad + box_size + pad, y + box_size - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return legend


def process_single_image(model, img_path, out_dir, palette, opacity):
    """处理单张图片的逻辑封装"""
    
    # 为当前图片创建一个子目录，避免文件覆盖
    # 例如: img_path = "data/test.jpg" -> save_dir = "out_dir/test"
    stem = Path(img_path).stem
    save_dir = out_dir / stem
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 计算概率 (Probabilities)
    img_tensor, img_metas = build_data(model, str(img_path))
    with torch.no_grad():
        seg_logit = model.encode_decode(img_tensor, [img_metas])
        if seg_logit.dim() == 2:
            seg_logit = seg_logit.unsqueeze(0).unsqueeze(0)
        elif seg_logit.dim() == 3:
            seg_logit = seg_logit.unsqueeze(0)
        prob = torch.softmax(seg_logit, dim=1)[0].cpu().numpy()
        if prob.ndim == 2:
            prob = prob[None, ...]

    # 2. 推理获取标签 (Labels)
    # Use official inference for stable label map shape
    result = inference_segmentor(model, str(img_path))
    pred = result[0]
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    pred = np.asarray(pred)
    if pred.ndim > 2:
        pred = pred.squeeze()
    if pred.ndim < 2:
        pred = prob.argmax(axis=0)
    pred = pred.astype(np.uint8)

    # 保存概率图
    prob_path = save_dir / "probabilities.npz"
    np.savez_compressed(prob_path, prob=prob)

    # 保存标签图 (raw ids)
    label_path = save_dir / "labels.png"
    cv2.imwrite(str(label_path), pred)

    # 保存彩色标签图
    if palette is not None:
        palette_arr = np.array(palette, dtype=np.uint8)
        color_label = palette_arr[pred]
        color_label = color_label[..., ::-1]  # RGB -> BGR
        color_label_path = save_dir / "labels_color.png"
        cv2.imwrite(str(color_label_path), color_label)

    # 保存叠加图 (Overlay)
    if hasattr(model, "module"):
        vis_model = model.module
    else:
        vis_model = model
        
    overlay = vis_model.show_result(
        str(img_path),
        [pred],
        palette=palette,
        show=False,
        opacity=opacity,
    )
    overlay_path = save_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)


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
    df = pd.read_csv(imgref_path, sep=None, engine="python")
    columns = {c.lower(): c for c in df.columns}
    if "name" in columns and "id" in columns:
        name_col = columns["name"]
        id_col = columns["id"]
        width_col = columns.get("width")
        height_col = columns.get("height")
        return df, name_col, id_col, width_col, height_col

    df = pd.read_csv(imgref_path, sep=None, engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError("imgRef.txt 至少需要两列: Name 和 ID")
    name_col = 0
    id_col = 1
    width_col = 2 if df.shape[1] > 2 else None
    height_col = 3 if df.shape[1] > 3 else None
    return df, name_col, id_col, width_col, height_col


def write_probabilities_h5(model, img_dir, imgref_path, out_h5_path):
    df, name_col, id_col, width_col, height_col = load_imgref(imgref_path)
    img_dir = Path(img_dir)
    out_h5_path = Path(out_h5_path)
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_h5_path), "w") as h5f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_name = row[name_col]
            image_id = row[id_col]
            img_path = resolve_image_path(str(img_name), img_dir)
            if img_path is None:
                print(f"⚠️ 找不到图片: {img_name}")
                continue

            img_tensor, img_metas = build_data(model, str(img_path))
            with torch.no_grad():
                seg_logit = model.encode_decode(img_tensor, [img_metas])
                if seg_logit.dim() == 2:
                    seg_logit = seg_logit.unsqueeze(0).unsqueeze(0)
                elif seg_logit.dim() == 3:
                    seg_logit = seg_logit.unsqueeze(0)
                prob = torch.softmax(seg_logit, dim=1)[0].cpu().numpy()
                if prob.ndim == 2:
                    prob = prob[None, ...]

            if width_col is not None and height_col is not None:
                width = int(row[width_col])
                height = int(row[height_col])
                resized = [
                    cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
                    for channel in prob
                ]
                prob = np.stack(resized, axis=0)

            h5f.create_dataset(str(image_id), data=prob, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description="Batch image inference")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth")
    # 新增参数：imgRef + 输出 h5
    parser.add_argument("--img-dir", required=True, help="Input image directory")
    parser.add_argument("--imgref", default=None, help="imgRef.txt path")
    parser.add_argument("--out-h5", default=None, help="Output h5 path")
    parser.add_argument("--out", default="demo_results", help="Output directory root")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--opacity", type=float, default=0.5)
    parser.add_argument("--params", default="params.yaml", help="Params yaml")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)

    if not img_dir.exists():
        print(f"❌ 输入目录不存在: {img_dir}")
        return

    # 收集图片文件
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    # 递归搜索 (rglob) 或者只搜索当前层 (glob)，这里使用 glob 搜索当前层
    img_files = sorted([
        f for f in img_dir.glob("*") 
        if f.suffix.lower() in valid_extensions and f.is_file()
    ])

    if not img_files:
        print(f"❌ 在 {img_dir} 中未找到图片文件。")
        return

    # Build model (只加载一次)
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # Load palette/classes from params if available
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

    if args.imgref and args.out_h5:
        print(f"📂 使用 imgRef: {args.imgref}")
        print(f"💾 概率将保存至: {args.out_h5}")
        write_probabilities_h5(model, img_dir, args.imgref, args.out_h5)
        print("\n✅ 概率保存完成。")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 找到 {len(img_files)} 张图片，开始处理...")
    print(f"💾 结果将保存至: {out_dir}")

    # 仅生成一次 Legend 并保存在根目录
    legend = build_legend(palette, classes)
    if legend is not None:
        legend_path = out_dir / "legend.png"
        cv2.imwrite(str(legend_path), legend)
        print(f"🏷️ Legend saved to: {legend_path}")

    # 批量推理循环
    prog_bar = mmcv.ProgressBar(len(img_files))
    for img_file in img_files:
        try:
            process_single_image(
                model=model,
                img_path=img_file,
                out_dir=out_dir,
                palette=palette,
                opacity=args.opacity,
            )
        except Exception as e:
            # 捕获异常防止单张图片损坏导致整个任务中断
            print(f"\n❌ 处理图片 {img_file.name} 时出错: {e}")

        prog_bar.update()

    print("\n✅ 所有任务处理完成。")


if __name__ == "__main__":
    main()