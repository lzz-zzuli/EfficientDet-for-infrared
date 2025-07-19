import cv2
import numpy as np
import os
import matplotlib.pyplot as plt  # 用于可选的图像显示

# 设置图像路径
input_image_path = "/workspace/Yet-Another-EfficientDet-Pytorch-master/datasets/m3fd/train2007/00040.png"
output_image_path = "/workspace/Yet-Another-EfficientDet-Pytorch-master/00040_grayscale.png"

# 加载图像（模仿 dataset.py 的 load_image 逻辑）
def load_and_convert_image(image_path):
    # 以彩色模式加载图像
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法加载图像: {image_path}")
        return None
    
    # 检查通道数
    if len(img.shape) == 3 and img.shape[2] == 3:  # 三通道图像 (BGR)
        # 转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 计算三通道均值：(R + G + B) / 3
        img = np.mean(img, axis=2).astype(np.float32)
        # 添加通道维度 (H, W, 1)
        img = img[..., np.newaxis]
    else:  # 单通道图像
        img = img[..., np.newaxis]
    
    # 归一化到 [0, 1]
    img = img.astype(np.float32) / 255.
    return img

# 保存单通道图像
def save_grayscale_image(img, output_path):
    if img is None:
        print("无法保存图像：输入图像为 None")
        return
    
    # 移除通道维度并反归一化到 [0, 255]
    img_to_save = img[..., 0] * 255.
    img_to_save = img_to_save.astype(np.uint8)
    
    # 保存为 PNG
    cv2.imwrite(output_path, img_to_save)
    print(f"已保存单通道图像到: {output_path}")

# 可选：显示原图和转换后的灰度图
def display_images(original_path, grayscale_img):
    if grayscale_img is None:
        print("无法显示图像：灰度图像为 None")
        return
    
    # 加载原图
    original_img = cv2.imread(original_path, cv2.IMREAD_COLOR)
    if original_img is None:
        print(f"无法加载原图: {original_path}")
        return
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 准备灰度图
    grayscale_img_display = grayscale_img[..., 0]  # 移除通道维度
    
    # 显示图像
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image (RGB)")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Grayscale Image ((R+G+B)/3)")
    plt.imshow(grayscale_img_display, cmap='gray')
    plt.axis('off')
    
    plt.show()

# 主程序
if __name__ == "__main__":
    # 加载并转换图像
    grayscale_img = load_and_convert_image(input_image_path)
    
    # 保存单通道图像
    save_grayscale_image(grayscale_img, output_image_path)
    
    # 显示原图和灰度图（可选，需在支持 GUI 的环境中运行）
    display_images(input_image_path, grayscale_img)