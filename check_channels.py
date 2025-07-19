import cv2
import os

# 数据集目录
dataset_dir = "/workspace/Yet-Another-EfficientDet-Pytorch-master/datasets/ctir/train2007"
# 或者测试集目录
# dataset_dir = "/workspace/Yet-Another-EfficientDet-Pytorch-master/test_png"

# 遍历目录中的图像文件
for img_file in os.listdir(dataset_dir):
    if img_file.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(dataset_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法加载图像: {img_path}")
            continue
        # 检查图像形状
        shape = img.shape
        channels = shape[2] if len(shape) == 3 else 1
        print(f"图像: {img_path}, 通道数: {channels}")