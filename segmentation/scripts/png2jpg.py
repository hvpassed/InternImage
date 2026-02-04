import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import concurrent.futures

def convert_one(file_info):
    """转换单张图片"""
    png_path, delete_original = file_info
    jpg_path = png_path.with_suffix('.jpg')
    
    try:
        # 如果目标文件已存在，跳过
        if jpg_path.exists():
            return "skipped"

        with Image.open(png_path) as img:
            # 强制转为 RGB (处理可能存在的 RGBA 透明通道)
            rgb_img = img.convert('RGB')
            # 保存为 JPG，质量 95 (兼顾画质和体积)
            rgb_img.save(jpg_path, quality=95)
        
        # 转换成功后再删除源文件
        if delete_original:
            os.remove(png_path)
            
        return "success"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="批量将 PNG 转换为 JPG (仅用于 img_dir)")
    parser.add_argument('dir', type=str, help="要转换的文件夹路径 (例如 data/loveDA/img_dir/train)")
    parser.add_argument('--delete', action='store_true', help="转换成功后删除原 .png 文件")
    parser.add_argument('--workers', type=int, default=8, help="并行处理线程数")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"❌ 错误: 文件夹不存在 - {target_dir}")
        return

    # 1. 扫描所有 png 文件
    print(f"🔍 正在扫描 {target_dir} ...")
    png_files = list(target_dir.glob('*.png'))
    
    if not png_files:
        print("⚠️ 该目录下没有找到 .png 文件。")
        return

    print(f"📦 找到 {len(png_files)} 张 PNG 图片，准备转换...")
    if args.delete:
        print("⚠️ 注意: 转换成功后将【删除】原 PNG 文件！")
    
    # 2. 确认提示
    confirm = input("❓ 是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消。")
        return

    # 3. 多线程并行转换
    tasks = [(p, args.delete) for p in png_files]
    results = {'success': 0, 'skipped': 0, 'errors': []}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 使用 tqdm 显示进度条
        for res in tqdm(executor.map(convert_one, tasks), total=len(tasks), unit="img"):
            if res == "success":
                results['success'] += 1
            elif res == "skipped":
                results['skipped'] += 1
            else:
                results['errors'].append(res)

    # 4. 打印报告
    print(f"\n✅ 处理完成!")
    print(f"   - 成功转换: {results['success']}")
    print(f"   - 跳过(已存在): {results['skipped']}")
    if results['errors']:
        print(f"   - ❌ 失败: {len(results['errors'])}")
        for e in results['errors'][:5]:
            print(f"     {e}")

if __name__ == '__main__':
    main()