import json

def read_coco_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def print_first_entry(data):
    if "images" in data and data["images"]:
        print("First image entry:", data["images"][0])
    if "annotations" in data and data["annotations"]:
        print("First annotation entry:", data["annotations"][0])

if __name__ == "__main__":
    file_path = "/data/home/cwk/workplace/mf/data/train_annotations.json"  # 替换为你的JSON文件路径
    coco_data = read_coco_json(file_path)
    print_first_entry(coco_data)