import argparse
import datetime
import os
import traceback
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm
from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
import numpy as np

# CORAL 损失函数
def coral_loss(source, target):
    d = source.size(1)  # 特征维度
    source_cov = torch.matmul(source.T, source) / (source.size(0) - 1)
    target_cov = torch.matmul(target.T, target) / (target.size(0) - 1)
    loss = torch.norm(source_cov - target_cov, p='fro') ** 2 / (4 * d * d)
    return loss

# 特征适配器（修改为2层MLP）
class FeatureAdapter(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureAdapter, self).__init__()
        # 第一层线性层，将输入特征映射到中间维度，这里中间维度设置为和输入维度相同
        self.linear1 = nn.Linear(feature_dim, feature_dim)
        # 激活函数使用ReLU
        self.relu = nn.ReLU()
        # 第二层线性层，将中间特征映射回原始特征维度
        self.linear2 = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # 第一层线性变换
        x = self.linear1(x)
        # 应用ReLU激活函数
        x = self.relu(x)
        # 第二层线性变换
        x = self.linear2(x)
        return x

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch with Domain Adaptation')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=2, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=8, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False, help='whether finetune only the regressor and classifier')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epochs between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0, help='Early stopping\'s parameter')
    parser.add_argument('--es_patience', type=int, default=0, help='Early stopping\'s parameter')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None, help='whether to load weights from a checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False, help='whether visualize the predicted boxes')
    parser.add_argument('--source_data_path', type=str, default='datasets/source/', help='path to source domain dataset')
    parser.add_argument('--target_data_path', type=str, default='datasets/target/', help='path to target domain dataset')
    parser.add_argument('--adapter_epochs', type=int, default=10, help='number of epochs to train adapter')
    parser.add_argument('--adapter_lr', type=float, default=0.001, help='learning rate for adapter')
    return parser.parse_args()

def train_adapter(source_loader, target_loader, model, adapter, opt, params):
    optimizer = torch.optim.Adam(adapter.parameters(), lr=opt.adapter_lr)
    model.eval()  # 固定 EfficientDet 模型
    adapter.train()

    # 选择要使用的层，这里选择最后三层，你可以根据需要调整
    layers_to_use = [-1, -2]
    # 定义每层的权重，这里简单地使用平均权重，你可以根据实验调整
    layer_weights = [1/len(layers_to_use)] * len(layers_to_use)  

    for epoch in range(opt.adapter_epochs):
        epoch_loss = []
        for source_data, target_data in zip(source_loader, target_loader):
            try:
                source_imgs = source_data['img'].cuda()
                target_imgs = target_data['img'].cuda()

                with torch.no_grad():
                    source_features_list, _, _, _ = model(source_imgs)  # 提取源域特征
                    target_features_list, _, _, _ = model(target_imgs)  # 提取目标域特征

                total_loss = 0
                source_combined_features = []
                target_combined_features = []

                for layer in layers_to_use:
                    source_features = source_features_list[layer]
                    target_features = target_features_list[layer]

                    # 检查特征是否包含 nan 或 inf
                    if torch.isnan(source_features).any() or torch.isinf(source_features).any():
                        print("Warning: Source features contain nan or inf")
                        continue
                    if torch.isnan(target_features).any() or torch.isinf(target_features).any():
                        print("Warning: Target features contain nan or inf")
                        continue

                    # 确保张量连续并展平特征图
                    source_features = source_features.contiguous().reshape(source_features.size(0), -1)
                    target_features = target_features.contiguous().reshape(target_features.size(0), -1)

                    source_combined_features.append(source_features)
                    target_combined_features.append(target_features)

                source_combined_features = torch.cat(source_combined_features, dim=1)
                target_combined_features = torch.cat(target_combined_features, dim=1)

                adapted_target_features = adapter(target_combined_features)  # 适配目标域特征
                layer_loss = coral_loss(source_combined_features, adapted_target_features)  # 计算 CORAL 损失

                # 检查损失是否有效
                if torch.isnan(layer_loss) or torch.isinf(layer_loss):
                    print("Warning: Layer loss is nan or inf, skipping layer")
                    continue

                total_loss = layer_loss

                # 检查总损失是否有效
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("Warning: Total loss is nan or inf, skipping batch")
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_loss.append(float(total_loss))
                torch.save(adapter.state_dict(), os.path.join(opt.saved_path, f'3adapter_d{opt.compound_coef}_{epoch+1}.pth'))
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[Error in batch] {traceback.format_exc()}")
                continue

        # 计算 epoch 平均损失
        if len(epoch_loss) > 0:
            print(f'Adapter Epoch [{epoch + 1}/{opt.adapter_epochs}], Loss: {np.mean(epoch_loss):.4f}')
        else:
            print(f'Adapter Epoch [{epoch + 1}/{opt.adapter_epochs}], Loss: nan (no valid batches)')

    # 保存适配器权重
    torch.save(adapter.state_dict(), os.path.join(opt.saved_path, f'3adapter_d{opt.compound_coef}_{epoch+1}.pth'))
def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    # 数据加载：源域和目标域
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    transform = transforms.Compose([
        Normalizer(mean=[0.512], std=[0.231]),  # 红外图像单通道标准化
        Resizer(input_sizes[opt.compound_coef])
    ])
    source_set = CocoDataset(root_dir=opt.source_data_path, set=params.train_set, transform=transform)
    target_set = CocoDataset(root_dir=opt.target_data_path, set=params.train_set, transform=transform)
    print(f"Source dataset size: {len(source_set)}")
    print(f"Target dataset size: {len(target_set)}")
    source_loader = DataLoader(source_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collater, drop_last=True)
    target_loader = DataLoader(target_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collater, drop_last=True)

    # 加载 EfficientDet 模型
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales), in_channels=1)

    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
            model.load_state_dict(torch.load(weights_path), strict=False)
            print(f'[Info] Loaded weights: {os.path.basename(weights_path)}')
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)

    # 动态计算特征维度
    with torch.no_grad():
        sample_batch = next(iter(source_loader))
        sample_imgs = sample_batch['img'].cuda()
        sample_features, _, _, _ = model(sample_imgs)

        # 选择要使用的层，这里选择最后三层
        layers_to_use = [-1, -2]
        total_feature_dim = 0
        for layer in layers_to_use:
            layer_features = sample_features[layer]
            layer_dim = layer_features.contiguous().reshape(layer_features.size(0), -1).size(1)
            total_feature_dim += layer_dim
        print(f"Calculated total feature_dim: {total_feature_dim}")

       

    # 初始化 FeatureAdapter
    adapter = FeatureAdapter(feature_dim=total_feature_dim).cuda()

    # 训练适配器
    train_adapter(source_loader, target_loader, model, adapter, opt, params)

    # 适配器训练完成后退出
    print("Adapter training completed. Exiting.")
    return

if __name__ == '__main__':
    opt = get_args()
    train(opt)