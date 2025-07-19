# Author: Zylo117
"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

compound_coef = 1  # EfficientDet-D1
force_input_size = None  # 使用默认输入大小
img_path = 'test/000004.jpg'  # 测试图像路径
weight_path = '/workspace/Yet-Another-EfficientDet-Pytorch-master/logs/ctir/efficientdet-d1_15_9500.pth'  # 权重路径
weight_path = '/workspace/Yet-Another-EfficientDet-Pytorch-master/logs/ctir/efficientdet-d1_25_15000.pth'
weight_path = '/workspace/Yet-Another-EfficientDet-Pytorch-master/logs/ctir/efficientdet-d1_44_26500.pth'
weight_path = '/workspace/Yet-Another-EfficientDet-Pytorch-master/logs/ctir/efficientdet-d1_58_35164.pth'
weight_path = '/workspace/Yet-Another-EfficientDet-Pytorch-master/logs/ctir/efficientdet-d1_73_44000.pth'
weight_path = '/workspace/Yet-Another-EfficientDet-Pytorch-master/logs/ctir/efficientdet-d1_99_59600.pth'

# 锚框配置（与 ctir.yml 一致）
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
num_gpus = torch.cuda.device_count() if use_cuda else 0
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['Pedestrian', 'Bus', 'Cyclist', 'Car', 'Truck']
color_list = standard_to_bgr(STANDARD_COLORS)

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# 预处理，适配单通道红外图像
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size, mean=[0.512], std=[0.231], is_grayscale=True)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

# 加载模型
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales, in_channels=1)
model.load_state_dict(torch.load(weight_path, map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus))).cuda()
    else:
        model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)

def display(preds, imgs, imshow=False, imwrite=True):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int64)  # 类型转换避免警告
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            print(f'目标：{obj}，得分：{score},位置:{x1, y1, x2, y2 }')
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            output_path = f'test/img_inferred_d{compound_coef}_this_repo_{i+101}_d1_99_59600.jpg'
            cv2.imwrite(output_path, imgs[i])
            print(f'Result saved to {output_path}')

out = invert_affine(framed_metas, out)
display(out, ori_imgs, imshow=False, imwrite=True)

# 速度测试
print('Running speed test...')
with torch.no_grad():
    print('Test: Model inferring and postprocessing')
    print('Inferring image for 10 times...')
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors = model(x)
        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
        out = invert_affine(framed_metas, out)
    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')