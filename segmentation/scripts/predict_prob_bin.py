#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict segmentation probabilities for images listed in imgRef and save to raw BIN files.

Async pipeline version:
- Reuse MMSeg test pipeline
- Real batch collate + scatter
- CPU preprocessing thread pool
- Single async Writer thread (fast raw binary dump to .bin)
- Generate a meta.json in the output directory to record shapes/dtypes
- Optional AMP(fp16) inference
- Optional GPU resize
- Optional label / overlay export
- Handle missing images by writing zero-probability maps
"""

import argparse
import os
import sys
import time
import queue
import threading
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import cv2
import mmcv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {results['img']}")

        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


def load_params(params_path):
    if not params_path:
        return {}
    if not os.path.exists(params_path):
        return {}
    with open(params_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


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
    try:
        df = pd.read_csv(imgref_path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(imgref_path, sep=",")
        except Exception:
            df = pd.read_csv(imgref_path, header=None)

    columns = {str(c).lower(): c for c in df.columns}

    if "name" in columns and "id" in columns:
        name_col = columns["name"]
        id_col = columns["id"]
        width_col = columns.get("width", columns.get("w"))
        height_col = columns.get("height", columns.get("h"))
        return df, name_col, id_col, width_col, height_col

    df = pd.read_csv(imgref_path, sep=None, engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError("imgRef.txt 至少需要两列: Name 和 ID")

    name_col = 0
    id_col = 1
    width_col = 2 if df.shape[1] > 2 else None
    height_col = 3 if df.shape[1] > 3 else None
    return df, name_col, id_col, width_col, height_col


def get_num_classes(model):
    vis_model = model.module if hasattr(model, "module") else model

    if hasattr(vis_model, "decode_head") and hasattr(vis_model.decode_head, "num_classes"):
        return int(vis_model.decode_head.num_classes)

    if hasattr(model, "CLASSES") and model.CLASSES is not None:
        return len(model.CLASSES)

    return int(model.cfg.model.decode_head.num_classes)


def prepare_test_pipeline(model):
    cfg = model.cfg
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    return Compose(test_pipeline)


def get_save_dtype(dtype_name):
    dtype_name = dtype_name.lower()
    if dtype_name == "float16":
        return np.float16, torch.float16
    if dtype_name == "float32":
        return np.float32, torch.float32
    raise ValueError(f"不支持的 save dtype: {dtype_name}")


def maybe_cuda_sync(model):
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()


def get_autocast_context(enabled):
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def get_free_mem_gb(device):
    if not torch.cuda.is_available():
        return None
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(torch.device(device))
        return free_mem / 1024**3, total_mem / 1024**3
    except Exception:
        return None


def print_cuda_mem(prefix, device):
    mem = get_free_mem_gb(device)
    if mem is not None:
        free_mem, total_mem = mem
        print(f"{prefix} free={free_mem:.2f} GB / total={total_mem:.2f} GB")


def resize_prob_cpu(prob_np, target_width, target_height):
    resized = [
        cv2.resize(channel, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        for channel in prob_np
    ]
    return np.stack(resized, axis=0)


def unwrap_first_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "data"):
        return unwrap_first_tensor(obj.data)
    if isinstance(obj, (list, tuple)):
        for x in obj:
            t = unwrap_first_tensor(x)
            if isinstance(t, torch.Tensor):
                return t
    return None


def normalize_img_metas_for_encode_decode(img_metas):
    obj = img_metas

    while hasattr(obj, "data"):
        obj = obj.data

    while isinstance(obj, (list, tuple)) and len(obj) == 1:
        first = obj[0]
        if isinstance(first, dict):
            break
        obj = first
        while hasattr(obj, "data"):
            obj = obj.data

    if isinstance(obj, tuple):
        obj = list(obj)

    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        return obj

    raise TypeError(f"无法解析 img_metas 结构: type={type(obj)}")


def prepare_sample_cpu(test_pipeline, img_path):
    data = dict(img=img_path)
    data = test_pipeline(data)
    return data


def get_sample_input_hw(sample):
    img_tensor = unwrap_first_tensor(sample["img"])
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("无法从 sample['img'] 中提取张量")
    return int(img_tensor.shape[-2]), int(img_tensor.shape[-1])


def build_batch_data(model, batch_items):
    samples = [item["sample"] for item in batch_items]
    data = collate(samples, samples_per_gpu=len(samples))

    if next(model.parameters()).is_cuda:
        data = scatter(data, [next(model.parameters()).device])[0]

    img_tensor = unwrap_first_tensor(data["img"])
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("无法从 batch data['img'] 中提取张量")

    img_metas = normalize_img_metas_for_encode_decode(data["img_metas"])
    return img_tensor, img_metas


def infer_batch(model, batch_items, amp_enabled, gpu_resize, torch_save_dtype):
    batch_tensor, img_metas = build_batch_data(model, batch_items)

    with torch.inference_mode():
        with get_autocast_context(amp_enabled):
            seg_logit = model.encode_decode(batch_tensor, img_metas)

            if seg_logit.dim() == 2:
                seg_logit = seg_logit.unsqueeze(0).unsqueeze(0)
            elif seg_logit.dim() == 3:
                if len(batch_items) == 1:
                    seg_logit = seg_logit.unsqueeze(0)
                else:
                    raise RuntimeError(
                        f"批量推理返回 3 维张量，无法安全判断维度含义: shape={tuple(seg_logit.shape)}"
                    )

            if seg_logit.dim() != 4:
                raise RuntimeError(f"模型输出维度异常: shape={tuple(seg_logit.shape)}")

            if gpu_resize:
                target_h = batch_items[0]["target_height"]
                target_w = batch_items[0]["target_width"]
                if target_h is not None and target_w is not None:
                    seg_logit = F.interpolate(
                        seg_logit,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )

            prob_t = torch.softmax(seg_logit, dim=1)

    maybe_cuda_sync(model)
    prob_np = prob_t.to(torch_save_dtype).cpu().numpy()

    outputs = []
    for i, item in enumerate(batch_items):
        one_prob = prob_np[i]
        if not gpu_resize:
            target_w = item["target_width"]
            target_h = item["target_height"]
            if target_w is not None and target_h is not None:
                one_prob = resize_prob_cpu(one_prob, target_w, target_h)
        outputs.append(one_prob)

    return outputs


def build_color_mask(pred, palette):
    h, w = pred.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(palette):
        color_mask[pred == cls_id] = color[:3]
    return color_mask


def save_visualization_simple(out_label_dir, img_path, image_id, prob, opacity, target_width, target_height, palette):
    pred = prob.argmax(axis=0).astype(np.uint8)

    label_path = out_label_dir / f"{image_id}_label.png"
    cv2.imwrite(str(label_path), pred)

    if palette is None:
        return

    img_show = mmcv.imread(str(img_path))
    if img_show is None:
        return

    if target_width is not None and target_height is not None:
        img_show = cv2.resize(
            img_show,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    color_mask = build_color_mask(pred, palette)
    overlay = cv2.addWeighted(img_show, 1.0 - opacity, color_mask, opacity, 0.0)

    overlay_path = out_label_dir / f"{image_id}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)


def writer_thread_main(
    out_bin_dir,
    task_queue,
    out_label_dir,
    opacity,
    palette,
    writer_stats,
    metadata_list
):
    out_bin_dir.mkdir(parents=True, exist_ok=True)
    if out_label_dir:
        out_label_dir.mkdir(parents=True, exist_ok=True)

    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        try:
            t0 = time.perf_counter()
            
            prob = task["prob"]
            image_id = task["image_id"]
            image_name = task["image_name"]
            
            # 确保内存连续 (C-contiguous)
            prob = np.ascontiguousarray(prob)
            
            # 直接写入二进制文件
            bin_filename = f"{image_id}.bin"
            bin_path = out_bin_dir / bin_filename
            prob.tofile(str(bin_path))
            
            writer_stats["io"] += time.perf_counter() - t0

            # 收集元数据用于生成 meta.json
            C, H, W = prob.shape
            metadata_list.append({
                "id": int(image_id),
                "name": str(image_name),
                "width": int(W),
                "height": int(H),
                "channels": int(C),
                "dtype": str(prob.dtype),
                "bin_file": bin_filename
            })

            if out_label_dir:
                t1 = time.perf_counter()
                save_visualization_simple(
                    out_label_dir=out_label_dir,
                    img_path=task["img_path"],
                    image_id=image_id,
                    prob=prob,
                    opacity=opacity,
                    target_width=task["target_width"],
                    target_height=task["target_height"],
                    palette=palette,
                )
                writer_stats["viz"] += time.perf_counter() - t1

            writer_stats["written"] += 1

        except Exception as e:
            writer_stats["errors"] += 1
            print(f"⚠️ Writer 处理失败: ID={task.get('image_id')} -> {e}")

        finally:
            task_queue.task_done()


def preprocess_record(record, name_col, id_col, width_col, height_col, img_dir, test_pipeline):
    image_name = record[name_col]
    image_id = record[id_col]
    img_path = resolve_image_path(str(image_name), img_dir)

    target_width, target_height = None, None
    if width_col is not None and height_col is not None:
        try:
            target_width = int(record[width_col])
            target_height = int(record[height_col])
        except Exception:
            target_width, target_height = None, None

    if img_path is None:
        return {
            "status": "missing",
            "image_id": image_id,
            "image_name": image_name,
            "img_path": None,
            "target_width": target_width,
            "target_height": target_height,
        }

    sample = prepare_sample_cpu(test_pipeline, str(img_path))
    input_h, input_w = get_sample_input_hw(sample)
    batch_key = (input_h, input_w, target_height, target_width)

    return {
        "status": "ok",
        "image_id": image_id,
        "image_name": image_name,
        "img_path": str(img_path),
        "target_width": target_width,
        "target_height": target_height,
        "sample": sample,
        "batch_key": batch_key,
    }


def write_probabilities_bin(
    model,
    img_dir,
    imgref_path,
    out_bin_dir,
    out_label_dir=None,
    opacity=0.5,
    save_dtype="float16",
    amp=False,
    log_every=50,
    batch_size=12,
    gpu_resize=True,
    raise_on_single_fail=False,
    preprocess_workers=4,
    prefetch_size=32,
    writer_queue_size=64,
):
    df, name_col, id_col, width_col, height_col = load_imgref(imgref_path)
    records = df.to_dict("records")

    img_dir = Path(img_dir)
    out_bin_dir = Path(out_bin_dir)
    out_bin_dir.mkdir(parents=True, exist_ok=True)

    if out_label_dir:
        out_label_dir = Path(out_label_dir)

    vis_model = model.module if hasattr(model, "module") else model
    _ = vis_model  # 保留变量，便于后续扩展
    num_classes = get_num_classes(model)
    np_save_dtype, torch_save_dtype = get_save_dtype(save_dtype)

    model.eval()
    torch.backends.cudnn.benchmark = True
    test_pipeline = prepare_test_pipeline(model)
    use_cuda = next(model.parameters()).is_cuda
    amp_enabled = bool(amp and use_cuda)
    palette = getattr(model, "PALETTE", None)

    print(f"📌 类别数: {num_classes}")
    print(f"📌 保存精度: {save_dtype}")
    print(f"📌 AMP 推理: {'开启' if amp_enabled else '关闭'}")
    print(f"📌 GPU resize: {'开启' if gpu_resize else '关闭'}")
    print(f"📌 输出可视化: {'开启' if out_label_dir else '关闭'}")
    print(f"📌 批量大小: {batch_size}")
    print(f"📌 预处理线程数: {preprocess_workers}")
    print(f"📌 预取窗口: {prefetch_size}")
    print(f"📌 Writer 队列大小: {writer_queue_size}")

    stats = {
        "prep": 0.0,
        "infer": 0.0,
        "count": 0,
        "missing": 0,
        "skipped": 0,
    }
    writer_stats = {
        "io": 0.0,
        "viz": 0.0,
        "written": 0,
        "errors": 0,
    }

    metadata_list = [] # 用于收集生成的 BIN 文件元数据
    task_queue = queue.Queue(maxsize=writer_queue_size)
    writer_thread = threading.Thread(
        target=writer_thread_main,
        args=(out_bin_dir, task_queue, out_label_dir, opacity, palette, writer_stats, metadata_list),
        daemon=True,
    )
    writer_thread.start()

    current_batch = []
    current_batch_key = None

    def enqueue_write_task(item, prob):
        task_queue.put({
            "image_id": item["image_id"],
            "image_name": item["image_name"],
            "img_path": item["img_path"],
            "prob": prob.astype(np_save_dtype, copy=False),
            "target_width": item["target_width"],
            "target_height": item["target_height"],
        })

    def flush_batch(batch_items):
        if not batch_items:
            return

        t_infer = time.perf_counter()
        try:
            probs = infer_batch(
                model=model,
                batch_items=batch_items,
                amp_enabled=amp_enabled,
                gpu_resize=gpu_resize,
                torch_save_dtype=torch_save_dtype,
            )
        except Exception as e:
            if len(batch_items) > 1:
                ids = [str(x["image_id"]) for x in batch_items]
                print(f"⚠️ 批量推理失败，退回逐张处理。IDs={ids} | error={e}")
                for item in batch_items:
                    flush_batch([item])
                return
            else:
                item = batch_items[0]
                print(f"❌ 单张推理失败: {item['img_path']} (ID: {item['image_id']}) -> {e}")
                stats["skipped"] += 1
                if raise_on_single_fail:
                    raise
                return

        stats["infer"] += time.perf_counter() - t_infer

        for item, prob in zip(batch_items, probs):
            enqueue_write_task(item, prob)
            stats["count"] += 1

    def flush_current():
        nonlocal current_batch, current_batch_key
        if current_batch:
            flush_batch(current_batch)
            current_batch = []
            current_batch_key = None

    pending = deque()

    def submit_record(executor, rec):
        t0 = time.perf_counter()
        fut = executor.submit(
            preprocess_record,
            rec,
            name_col,
            id_col,
            width_col,
            height_col,
            img_dir,
            test_pipeline,
        )
        pending.append((fut, t0))

    with ThreadPoolExecutor(max_workers=max(1, preprocess_workers)) as executor:
        rec_iter = iter(records)

        for _ in range(min(prefetch_size, len(records))):
            try:
                submit_record(executor, next(rec_iter))
            except StopIteration:
                break

        pbar = tqdm(total=len(records), desc="Infer")
        processed_records = 0

        while pending:
            fut, t0 = pending.popleft()

            try:
                item = fut.result()
                stats["prep"] += time.perf_counter() - t0
            except Exception as e:
                flush_current()
                print(f"❌ 预处理失败 -> {e}")
                stats["skipped"] += 1
                processed_records += 1
                pbar.update(1)
                try:
                    submit_record(executor, next(rec_iter))
                except StopIteration:
                    pass
                continue

            processed_records += 1
            pbar.update(1)

            if item["status"] == "missing":
                flush_current()
                print(f"⚠️ 找不到图片: {item['image_name']} (ID: {item['image_id']}) -> 生成全黑预测占位")
                stats["missing"] += 1

                if item["target_width"] is not None and item["target_height"] is not None:
                    zero_prob = np.zeros(
                        (num_classes, item["target_height"], item["target_width"]),
                        dtype=np_save_dtype,
                    )
                    enqueue_write_task(item, zero_prob)
                    stats["count"] += 1
                else:
                    print(f"❌ 错误: ID {item['image_id']} 图片缺失且 imgRef 中没有宽高信息，跳过。")
                    stats["skipped"] += 1

            else:
                batch_key = item["batch_key"]

                if current_batch and batch_key != current_batch_key:
                    flush_current()

                current_batch.append(item)
                current_batch_key = batch_key

                if len(current_batch) >= batch_size:
                    flush_current()

            if log_every > 0 and processed_records % log_every == 0:
                n = max(stats["count"], 1)
                mem = get_free_mem_gb(next(model.parameters()).device) if use_cuda else None
                mem_str = ""
                if mem is not None:
                    free_mem, total_mem = mem
                    mem_str = f", free_mem={free_mem:.2f}/{total_mem:.2f}GB"

                print(
                    f"[{processed_records}/{len(records)}] "
                    f"prep={stats['prep']/n:.4f}s/img, "
                    f"infer={stats['infer']/n:.4f}s/img, "
                    f"io={writer_stats['io']/max(writer_stats['written'], 1):.4f}s/img, "
                    f"viz={writer_stats['viz']/max(writer_stats['written'], 1):.4f}s/img"
                    f"{mem_str}"
                )

            try:
                submit_record(executor, next(rec_iter))
            except StopIteration:
                pass

        pbar.close()

    flush_current()

    task_queue.put(None)
    task_queue.join()
    writer_thread.join()

    # Write metadata JSON file
    meta_json_path = out_bin_dir / "meta.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump({"images": metadata_list}, f, indent=4, ensure_ascii=False)
    print(f"📝 元数据已保存至: {meta_json_path}")

    n = max(stats["count"], 1)
    wn = max(writer_stats["written"], 1)
    print("\n========== 性能统计 ==========")
    print(f"成功提交图片数: {stats['count']}")
    print(f"Writer 实际写入数: {writer_stats['written']}")
    print(f"缺失图片数: {stats['missing']}")
    print(f"跳过图片数: {stats['skipped']}")
    print(f"Writer 错误数: {writer_stats['errors']}")
    print(f"平均预处理耗时: {stats['prep']/n:.4f} s/img")
    print(f"平均推理耗时:   {stats['infer']/n:.4f} s/img")
    print(f"平均BIN写入耗时:{writer_stats['io']/wn:.4f} s/img")
    print(f"平均可视化耗时: {writer_stats['viz']/wn:.4f} s/img")
    print("================================")


def main():
    parser = argparse.ArgumentParser(
        description="Predict segmentation probabilities and save to BIN files (async preprocess + async writer)"
    )
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth")
    parser.add_argument("--img-dir", required=True, help="Input image directory")
    parser.add_argument("--imgref", required=True, help="imgRef.txt path")
    parser.add_argument("--out-bin-dir", required=True, help="Output directory for .bin files and meta.json")
    parser.add_argument("--out-label-dir", default=None, help="Output label/overlay dir")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--opacity", type=float, default=0.5, help="Overlay opacity")
    parser.add_argument("--params", default="params.yaml", help="Params yaml")

    parser.add_argument(
        "--save-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="BIN save dtype",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP(fp16) inference on CUDA",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size for same-shape images",
    )
    parser.add_argument(
        "--gpu-resize",
        action="store_true",
        help="Resize logits on GPU before softmax/save",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print average timing every N images",
    )
    parser.add_argument(
        "--raise-on-single-fail",
        action="store_true",
        help="Raise exception if even single-image inference fails",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=4,
        help="Number of CPU preprocessing worker threads",
    )
    parser.add_argument(
        "--prefetch-size",
        type=int,
        default=32,
        help="How many records to preprocess ahead",
    )
    parser.add_argument(
        "--writer-queue-size",
        type=int,
        default=64,
        help="Max pending BIN write tasks",
    )

    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        print(f"❌ 输入目录不存在: {img_dir}")
        return

    if not os.path.exists(args.imgref):
        print(f"❌ imgRef 不存在: {args.imgref}")
        return

    print_cuda_mem("🚀 Before init_segmentor:", args.device)
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    print_cuda_mem("🚀 After init_segmentor:", args.device)

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
    print(f"💾 概率将保存至: {args.out_bin_dir}")

    write_probabilities_bin(
        model=model,
        img_dir=img_dir,
        imgref_path=args.imgref,
        out_bin_dir=args.out_bin_dir,
        out_label_dir=args.out_label_dir,
        opacity=args.opacity,
        save_dtype=args.save_dtype,
        amp=args.amp,
        log_every=args.log_every,
        batch_size=max(1, args.batch_size),
        gpu_resize=args.gpu_resize,
        raise_on_single_fail=args.raise_on_single_fail,
        preprocess_workers=max(1, args.preprocess_workers),
        prefetch_size=max(1, args.prefetch_size),
        writer_queue_size=max(1, args.writer_queue_size),
    )

    print("✅ 概率保存完成。")


if __name__ == "__main__":
    main()