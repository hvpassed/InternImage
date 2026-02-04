import json

def validate_coco(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # 检查关键字段
    assert "images" in data, "Missing 'images' field"
    assert "annotations" in data, "Missing 'annotations' field"
    assert "categories" in data, "Missing 'categories' field"
    
    print(f"{json_file} is valid COCO format.")


validate_coco('train_annotations.json')
validate_coco('val_annotations.json')