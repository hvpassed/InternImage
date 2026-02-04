from PIL import Image
import numpy as np
import os,cv2
from tqdm import tqdm
# 打开三通道的PNG图片
def convert(url):
    
    image = cv2.imread(url)  # 读取图像，默认是BGR格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 定义映射关系
    mapping = {
        (0, 0, 0): 0,
        (1, 1, 1): 1,
        (2, 2, 2): 2,
        (3, 3, 3): 3,
        (4, 4, 4): 4
    }

    # 创建一个空的单通道数组
    single_channel_array = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

    # 遍历图像数组并进行映射
    for i in range(image_rgb.shape[0]):
        for j in range(image_rgb.shape[1]):
            pixel = tuple(image_rgb[i, j])  # 获取RGB值
            if pixel in mapping:
                single_channel_array[i, j] = mapping[pixel]
            else:
                single_channel_array[i, j] = 0  # 或者其他默认值

    # 保存单通道图像
    cv2.imwrite(url, single_channel_array)

img_path = '/data/home/cwk/workplace/mf/data/train/imgs'
all_file = []
for i in os.listdir(img_path):
    if(i.endswith('.png')):
        all_file.append(i)
for i in tqdm(all_file):
    convert(os.path.join(img_path,i))

img_path = '/data/home/cwk/workplace/mf/data/val/imgs'
all_file = []
for i in os.listdir(img_path):
    if(i.endswith('.png')):
        all_file.append(i)
for i in tqdm(all_file):
    convert(os.path.join(img_path,i))