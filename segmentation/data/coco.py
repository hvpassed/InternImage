import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

# 定义路径
train_image_dir = os.path.join("train", "imgs")
train_mask_dir = os.path.join("train", "imgs")
val_image_dir = os.path.join("val", "imgs")
val_mask_dir = os.path.join("val", "imgs")

# 输出文件
train_output_json = os.path.join("train_annotations.json")
val_output_json = os.path.join("val_annotations.json")

# 定义类别信息
categories = [
    {"id": 0, "name": "background"},
    {"id": 1, "name": "building"},
    {"id": 2, "name": "woodland"},
    {"id": 3, "name": "water"},
    {"id": 4, "name": "road"},
    # 添加更多类别
]

def convert_to_coco(image_dir, mask_dir, output_json):
    # 初始化COCO格式字典
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_id = 1
    annotation_id = 1
    all_file = os.listdir(image_dir)
    imgfile = []
    for i in all_file:
        if(i.endswith('.jpg')):
            imgfile.append(i)
    # 遍历图像和掩码
    for image_name in tqdm(imgfile):
        # 图像信息
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name.replace(".jpg", ".png"))  # 假设图像是JPG格式

        if not os.path.exists(mask_path):
            print(f"Mask file not found for {image_name}, skipping...")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape

        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height,
            "sem_seg_file_name":image_name.replace(".jpg", ".png")
        })

        # 掩码信息
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        for class_id in np.unique(mask):
            # if class_id == 0:  # 跳过背景
            #     continue

            # 提取当前类别的掩码
            class_mask = (mask == class_id).astype(np.uint8)
            fortran_mask = np.asfortranarray(class_mask)
            rle = maskUtils.encode(fortran_mask)
            
            # Ensure 'counts' is a string (bytes to string)
            rle['counts'] = rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
            
            area = maskUtils.area(rle).item()
            bbox = maskUtils.toBbox(rle).tolist()  # Convert to list for JSON compatibility

            # 添加标注信息
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "segmentation": rle,  # rle已经是合适的格式
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    # 保存为JSON文件
    with open(output_json, "w") as f:
        json.dump(coco_data, f)

# 转换训练集
convert_to_coco(train_image_dir, train_mask_dir, train_output_json)
print(f"Train annotations saved to {train_output_json}")

# 转换验证集
convert_to_coco(val_image_dir, val_mask_dir, val_output_json)
print(f"Validation annotations saved to {val_output_json}")
