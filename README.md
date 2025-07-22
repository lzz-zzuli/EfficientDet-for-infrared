# EfficientDet 训练与测试流程：从 CTIR 到 M3FD
本文档记录了在 `ctir` 数据集上训练 EfficientDet 模型、在 `m3fd` 数据集上训练适配器（`FeatureAdapter`）并进行 COCO 风格评估的完整流程。
## 1.准备
###  数据集
- **CTIR**：源数据集，单通道红外图像，用于预训练 EfficientDet。
  - 路径：`datasets/ctir`
  - 标注：`datasets/ctir/annotations/instances_train.json`
  - 类别数：5（根据 `projects/ctir.yml` 的 `obj_list`）
- **M3FD**：目标数据集，单通道红外图像，用于适配器训练和测试。
  - 路径：`datasets/m3fd`
  - 标注：`datasets/m3fd/annotations/instances_test.json`
  - 测试集大小：1050 张图像
  - 挑战：小目标占比高（`AP@small=0.001`）


### 依赖安装
参考

## 2. 在 CTIR 数据集上训练 EfficientDet

### 2.1 训练 EfficientDet-D0
- **目标**：在 `ctir` 数据集上训练 EfficientDet-D1，为 `m3fd` 适配器提供预训练权重。
- **命令**：
  ```bash
  python train.py -c 0 -p ctir --batch_size 8 --lr 1e-3 --num_epochs 500
  ```
- **参数**：
  - `-c 0`：EfficientDet-D0

- **输出**：
  - 权重保存：`logs/ctir/efficientdet-d1_<epoch>.pth`（例如，`efficientdet-d1_101_60500.pth`）
- **注意**：
  - 确保 `projects/ctir.yml` 定义了正确的 `obj_list`（5 类）和锚框参数（`anchors_ratios`, `anchors_scales`）。
  - 训练日志显示损失收敛情况，检查是否正常。

### 2.1 训练 EfficientDet-D1
- **目标**：在 `ctir` 数据集上训练 EfficientDet-D1，为 `m3fd` 适配器提供预训练权重。
- **命令**：
  ```bash
  python train.py -c 2 -p ctir --batch_size 8 --lr 1e-3 --num_epochs 500 
  ```
- **参数**：
  - `-c 2`：EfficientDet-D2

- **输出**：
  - 权重保存：`logs/ctir/efficientdet-d1_<epoch>.pth`（例如，`efficientdet-d1_101_60500.pth`）
- **注意**：
  - 确保 `projects/ctir.yml` 定义了正确的 `obj_list`（5 类）和锚框参数（`anchors_ratios`, `anchors_scales`）。
  - 训练日志显示损失收敛情况，检查是否正常。

## 3.迁移

### 训练adapter
- **目标**：将 `ctir` 预训练的 D1 模型适配到 `m3fd` 数据集。
- **命令**：
  ```bash
  python train_adapter2.py --project m3fd --compound_coef 1 --source_data_path /workspace/Yet-Another-EfficientDet-Pytorch-master/datasets/ctir --target_data_path /workspace/Yet-Another-EfficientDet-Pytorch-master/datasets/m3fd --load_weights logs/ctir/efficientdet-d1_99_59600.pth --adapter_epochs 100 --adapter_lr 0.001
  ```
- **适配器结构**（初始）：
  ```python
  class FeatureAdapter(nn.Module):
      def __init__(self, feature_dim):
          super(FeatureAdapter, self).__init__()
          self.linear = nn.Linear(feature_dim, feature_dim)
      def forward(self, x):
          return self.linear(x)
  ```
- **问题**：
  - 适配器只修改 `features[-1]`（`feature_dim=2200`）。
  - mAP 未提升（0.027 vs. 0.026），分类头变化小（`Classification difference ~0.3`）。
  - 训练 epoch（100）或学习率（0.0001）可能不足。

## 4. 在 M3FD 数据集上测试
 **命令**：
  ```bash
  python coco_eval_ours2.py --project m3fd --compound_coef 1 --weights logs/ctir/efficientdet-d1_101_60500.pth --adapter_weights logs/m3fd/adapter_d1_100.pth
  ```
## 5. 适配器多层特征对齐
 **训练命令**：
  ```bash
 python train_adapter3.py --project m3fd --compound_coef 2 --source_data_path /workspace/Yet-Another-EfficientDet-Pytorch-master/datasets/ctir --target_data_path /workspace/Yet-Another-EfficientDet-Pytorch-master/datasets/m3fd --load_weights logs/ctir/efficientdet-d2_150_180000.pth --adapter_epochs 100 --adapter_lr 0.001
  ```
**测试命令**：
  ```bash
  python coco_eval_ours4.py --project m3fd --compound_coef 2 --weights logs/ctir/efficientdet-d2_150_180000.pth  --adapter_weights logs/m3fd/3adapter_d2_1.pth
  ```
