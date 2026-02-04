print("1. 脚本开始运行...") # 第一句必须打印

import sys
import os
print("2. 基础库导入成功")

try:
    import torch
    print(f"3. PyTorch 导入成功: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"❌ PyTorch 挂了: {e}")

try:
    import mmcv
    print(f"4. MMCV 导入成功: {mmcv.__version__}")
except ImportError as e:
    print(f"❌ MMCV 挂了: {e}")

try:
    from mmseg.apis import init_segmentor
    print("5. MMSegmentation 导入成功")
except ImportError as e:
    print(f"❌ MMSegmentation 挂了: {e}")

# 检查路径
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
print(f"6. 当前工作目录: {os.getcwd()}")

if __name__ == '__main__':
    print("7. 进入主函数入口...")
    # 这里放原本的 main() 逻辑，或者暂时只放这一句测试