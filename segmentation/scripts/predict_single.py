#!/usr/bin/env python3
"""
Single-image inference with probability output.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
import yaml
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.datasets.pipelines import Compose

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


def main():
    parser = argparse.ArgumentParser(description="Single image inference")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth")
    parser.add_argument("--img", required=True, help="Input image path")
    parser.add_argument("--out", default="demo", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--opacity", type=float, default=0.5)
    parser.add_argument("--params", default="params.yaml", help="Params yaml")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model (load checkpoint manually to handle missing meta)
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

    img_tensor, img_metas = build_data(model, args.img)
    with torch.no_grad():
        seg_logit = model.encode_decode(img_tensor, [img_metas])
        if seg_logit.dim() == 2:
            seg_logit = seg_logit.unsqueeze(0).unsqueeze(0)
        elif seg_logit.dim() == 3:
            seg_logit = seg_logit.unsqueeze(0)
        prob = torch.softmax(seg_logit, dim=1)[0].cpu().numpy()
        if prob.ndim == 2:
            prob = prob[None, ...]

    # Use official inference for stable label map shape
    result = inference_segmentor(model, args.img)
    pred = result[0]
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    pred = np.asarray(pred)
    if pred.ndim > 2:
        pred = pred.squeeze()
    if pred.ndim < 2:
        pred = prob.argmax(axis=0)
    pred = pred.astype(np.uint8)

    # Save probability map
    prob_path = out_dir / "probabilities.npz"
    np.savez_compressed(prob_path, prob=prob)

    # Save label map (raw ids)
    label_path = out_dir / "labels.png"
    cv2.imwrite(str(label_path), pred)

    # Save colorized label map
    if palette is not None:
        palette_arr = np.array(palette, dtype=np.uint8)
        color_label = palette_arr[pred]
        color_label = color_label[..., ::-1]  # RGB -> BGR
        color_label_path = out_dir / "labels_color.png"
        cv2.imwrite(str(color_label_path), color_label)

    # Save overlay
    if hasattr(model, "module"):
        vis_model = model.module
    else:
        vis_model = model
    overlay = vis_model.show_result(
        args.img,
        [pred],
        palette=palette,
        show=False,
        opacity=args.opacity,
    )
    overlay_path = out_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    # Save legend
    legend = build_legend(palette, classes)
    if legend is not None:
        legend_path = out_dir / "legend.png"
        cv2.imwrite(str(legend_path), legend)

    print(f"Overlay: {overlay_path}")
    print(f"Labels: {label_path}")
    print(f"Probabilities: {prob_path}")
    if legend is not None:
        print(f"Legend: {legend_path}")
    if palette is not None:
        print(f"Color labels: {color_label_path}")


if __name__ == "__main__":
    main()
