import cv2
import os
def get_image_channels(image_path):
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    if image is not None:
        return image.shape[2] if len(image.shape) == 3 else 1
    else:
        raise FileNotFoundError(f"No image found at {image_path}")

# Example usage
image_path = '/data/home/cwk/workplace/mf/data/output.png'
channels = get_image_channels(image_path)
print(f"The image has {channels} channel(s).")