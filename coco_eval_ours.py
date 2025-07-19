# Author: Zylo117
"""
COCO-Style Evaluations with Optional Adapter for Single-Channel Infrared Images

put images here datasets/your_project_name/test/*.jpg
put annotations here datasets/your_project_name/annotations/instances_test.json
put weights here /path/to/your/weights/*.pth
put adapter weights here /path/to/your/adapter_weights/*.pth (optional)
change compound_coef
"""

import json
import os
import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.nn as nn
import numpy as np

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

# 特征适配器
class FeatureAdapter(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureAdapter, self).__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        return self.linear(x)

# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='target', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=1, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default='logs/ctir/efficientdet-d1_99_59600.pth', help='/path/to/weights')
ap.add_argument('--adapter_weights', type=str, default=None, help='/path/to/adapter_weights (optional)')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = args.weights
adapter_weights_path = args.adapter_weights

print(f'Running COCO-style evaluation on project {project_name}, weights {weights_path}' + 
      (f', adapter weights {adapter_weights_path}' if adapter_weights_path else ', no adapter used') + '...')

# 加载配置文件
params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef]

# 加载 EfficientDet 模型
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']), in_channels=1)
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.requires_grad_(False)
model.eval()

# 初始化 FeatureAdapter（仅当 adapter_weights_path 非空时）
adapter = None
feature_dim = None
if adapter_weights_path:
    # 动态计算特征维度
    with torch.no_grad():
        sample_img_path = f'datasets/{project_name}/test2007/000004.png'  # 使用单张图像计算特征维度
        ori_imgs, framed_imgs, framed_metas = preprocess(sample_img_path, max_size=input_size, mean=[0.512], std=[0.231], is_grayscale=True)
        x = torch.from_numpy(framed_imgs[0]).unsqueeze(0).permute(0, 3, 1, 2)
        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()
        sample_features, _, _, _ = model(x)
        feature_dim = sample_features[-1].contiguous().reshape(sample_features[-1].size(0), -1).size(1)
        print(f"Calculated feature_dim: {feature_dim}")

    # 加载 FeatureAdapter
    adapter = FeatureAdapter(feature_dim=feature_dim)
    adapter.load_state_dict(torch.load(adapter_weights_path, map_location=torch.device('cpu')))
    adapter.eval()

if use_cuda:
    model.cuda(gpu)
    if adapter:
        adapter.cuda(gpu)
    if use_float16:
        model.half()
        if adapter:
            adapter.half()

def evaluate_coco(img_path, set_name, image_ids, coco, model, adapter=None, threshold=0.05):
    results = []
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids, desc="Evaluating images"):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_size, mean=[0.512], std=[0.231], is_grayscale=True)
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = model(x)
            if adapter:
                # 适配最后一层特征
                adapted_features = adapter(features[-1].contiguous().reshape(features[-1].size(0), -1))
                features = features[:-1] + [adapted_features.view_as(features[-1])]
            # 后处理
            preds = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,  # COCO category_id 从 1 开始
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('The model does not provide any valid output, check model architecture and the data input')

    # 写入输出
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath) and override_prev_results:
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)
    return filepath

def _eval(coco_gt, image_ids, pred_json_path):
    # 加载预测结果
    coco_pred = coco_gt.loadRes(pred_json_path)

    # 运行 COCO 评估
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    SET_NAME = params.get('test_set', 'test')  # 使用 test_set，默认为 'test'
    VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
        results_file = evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model, adapter)
    else:
        results_file = f'{SET_NAME}_bbox_results.json'

    _eval(coco_gt, image_ids, results_file)