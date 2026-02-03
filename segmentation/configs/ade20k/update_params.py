import yaml
import argparse
import sys

def update_yaml(data_root):
    # 读取
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # 修改
    params['train_dataset']['data_root'] = data_root
    
    # 写回
    with open('params.yaml', 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    
    print(f"✅ params.yaml updated: data_root = {data_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    
    update_yaml(args.data_root)