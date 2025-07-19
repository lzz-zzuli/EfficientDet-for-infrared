# EfficientDet 训练与测试流程：从 CTIR 到 M3FD

本文档记录了在 `ctir` 数据集上训练 EfficientDet 模型、在 `m3fd` 数据集上训练适配器（`FeatureAdapter`）并进行 COCO 风格评估的完整流程。

## 1. 环境准备

### 1.1 环境配置


### 1.2 数据集
- **CTIR**：源数据集，单通道红外图像，用于预训练 EfficientDet。
  - 路径：`datasets/ctir`
  - 标注：`datasets/ctir/annotations/instances_train.json`
  - 类别数：5（根据 `projects/ctir.yml` 的 `obj_list`）
- **M3FD**：目标数据集，单通道红外图像，用于适配器训练和测试。
  - 路径：`datasets/m3fd`
  - 标注：`datasets/m3fd/annotations/instances_test.json`
  - 测试集大小：1050 张图像
  - 挑战：小目标占比高（`AP@small=0.001`）

### 1.3 依赖安装
```bash
conda create -n eff python=3.9
conda activate eff
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pycocotools numpy opencv-python tqdm pyyaml
```

## 2. 在 CTIR 数据集上训练 EfficientDet

### 2.1 训练 EfficientDet-D1
- **目标**：在 `ctir` 数据集上训练 EfficientDet-D1，为 `m3fd` 适配器提供预训练权重。
- **命令**：
  ```bash
  python train.py -c 2 -p ctir --batch_size 8 --lr 1e-3 --num_epochs 500 --load_weights weights、efficientdet-d2.pth --head_only True
  ```
- **参数**：
  - `-c 1`：EfficientDet-D1（640x640 分辨率，6.6M 参数，6.1 GFLOPs）
  - `--in_channels 1`：单通道红外图像

- **输出**：
  - 权重保存：`logs/ctir/efficientdet-d1_<epoch>.pth`（例如，`efficientdet-d1_101_60500.pth`）
- **注意**：
  - 确保 `projects/ctir.yml` 定义了正确的 `obj_list`（5 类）和锚框参数（`anchors_ratios`, `anchors_scales`）。
  - 训练日志显示损失收敛情况，检查是否正常。

### 2.2 尝试训练 EfficientDet-D5（失败）
- **目标**：尝试更高性能的 D5（1280x1280 分辨率，34M 参数，135 GFLOPs）。
- **命令**：
  ```bash
  python train.py -c 5 -p ctir --batch_size 8 --lr 1e-3 --num_epochs 500 --load_weights weights/efficientdet-d5.pth --head_only True
  ```
- **问题**：加载预训练权重 `weights/efficientdet-d5.pth`（RGB，3 通道）时，出现维度不匹配错误：
  ```
  RuntimeError: running_mean should contain 32 elements not 48
  ```
  - 原因：预训练权重的 `_conv_stem`（`[48, 3, 3, 3]`）和 `_bn0`（`running_mean` 48 维）基于 3 通道输入，而模型初始化为 `in_channels=1`（`_conv_stem` 输出 32 通道）。
- **修复**：
  - 修改 `train.py` 的权重加载逻辑，忽略不匹配的 BatchNorm 参数：
    ```python
    def load_weights(model, weights_path, in_channels=1):
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model_state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and v.size() == model_state_dict[k].size():
                new_state_dict[k] = v
            else:
                print(f"[Warning] Skipping {k} due to size mismatch: {v.size()} vs {model_state_dict[k].size()}")
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # 重置 BatchNorm 参数
        model.backbone_net.model._bn0.reset_running_stats()
        model.backbone_net.model._bn0.running_mean = torch.zeros(32)
        model.backbone_net.model._bn0.running_var = torch.ones(32)
        print(f"[Info] Loaded weights from {weights_path}")
    ```
  - 重新训练：
    ```bash
    python train.py -c 5 -p ctir --batch_size 8 --lr 1e-3 --num_epochs 500 --load_weights weights/efficientdet-d5.pth --head_only False
    ```

### 2.3 训练 EfficientDet-D2（推荐）
- **原因**：D1 的小目标检测性能差（`AP@small=0.001`），D5 计算量过高（135 GFLOPs）。D2（768x768 分辨率，8.1M 参数，11 GFLOPs）更适合 `m3fd` 的小目标检测。
- **命令**：
  ```bash
  python train.py -c 2 -p ctir --batch_size 8 --lr 1e-3 --num_epochs 500 --in_channels 1 --head_only False
  ```
- **输出**：`logs/ctir/efficientdet-d2_<epoch>.pth`

## 3. 训练 FeatureAdapter（从 CTIR 到 M3FD）

### 3.1 初始适配器训练（D1）
- **目标**：将 `ctir` 预训练的 D1 模型适配到 `m3fd` 数据集。
- **命令**：
  ```bash
  python train_adapter2.py --project m3fd --compound_coef 1 --source_data_path datasets/ctir --target_data_path datasets/m3fd --load_weights logs/ctir/efficientdet-d1_101_60500.pth --adapter_epochs 100 --adapter_lr 0.0001
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

### 3.2 改进适配器训练
- **改进**：
  - 使用多层感知机（MLP）增强适配器表达能力：
    ```python
    class FeatureAdapter(nn.Module):
        def __init__(self, feature_dim):
            super(FeatureAdapter, self).__init__()
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim)
            )
        def forward(self, x):
            return self.mlp(x)
    ```
  - 适配所有特征层（而非仅 `features[-1]`）。
  - 增加 epoch 和学习率。
- **命令**：
  ```bash
  python train_adapter2.py --project m3fd --compound_coef 1 --source_data_path datasets/ctir --target_data_path datasets/m3fd --load_weights logs/ctir/efficientdet-d1_101_60500.pth --adapter_epochs 200 --adapter_lr 0.001
  ```
- **输出**：`logs/m3fd/adapter_d1_<epoch>.pth`

### 3.3 适配器训练（D2）
- **命令**：
  ```bash
  python train_adapter2.py --project m3fd --compound_coef 2 --source_data_path datasets/ctir --target_data_path datasets/m3fd --load_weights logs/ctir/efficientdet-d2_<epoch>.pth --adapter_epochs 200 --adapter_lr 0.001
  ```
- **输出**：`logs/m3fd/adapter_d2_<epoch>.pth`

## 4. 在 M3FD 数据集上测试

### 4.1 测试 D1（初始）
- **命令**：
  ```bash
  python coco_eval_ours2.py --project m3fd --compound_coef 1 --weights logs/ctir/efficientdet-d1_101_60500.pth --adapter_weights logs/m3fd/adapter_d1_100.pth
  ```
- **结果**：
  - mAP@0.50:0.95=0.027（适配器） vs. 0.026（无适配器）
  - `AP@small=0.001`，`AP@medium=0.044`，`AP@large=0.059`
  - `Feature difference (L2 norm)`：28.3804–47.2817
  - `Regression difference (L2 norm)`：5.7777–10.4314
  - `Classification difference (L2 norm)`：0.2741–0.3391
- **问题**：适配器未显著提升 mAP，分类头变化小。

### 4.2 测试 D1（改进适配器）
- **命令**：
  ```bash
  python coco_eval_ours2.py --project m3fd --compound_coef 1 --weights logs/ctir/efficientdet-d1_101_60500.pth --adapter_weights logs/m3fd/adapter_d1_<epoch>.pth --nms_threshold 0.3 --threshold 0.005
  ```
- **改进**：
  - 适配器：多层 MLP，适配所有特征层。
  - 降低 `threshold=0.005`，`nms_threshold=0.3`。
- **预期**：`Classification difference` 增加，mAP 高于 0.027。

### 4.3 测试 D2
- **命令**：
  ```bash
  python coco_eval_ours2.py --project m3fd --compound_coef 2 --weights logs/ctir/efficientdet-d2_<epoch>.pth --adapter_weights logs/m3fd/adapter_d2_<epoch>.pth --nms_threshold 0.3 --threshold 0.005
  ```
- **预期**：D2 的 768x768 分辨率和 5 层 BiFPN 提升 `AP@small` 和整体 mAP。

### 4.4 测试集分析
- **检查 `bbox` 尺寸分布**：
  ```python
  from pycocotools.coco import COCO
  coco = COCO('datasets/m3fd/annotations/instances_test.json')
  anns = coco.loadAnns(coco.getAnnIds())
  areas = [ann['area'] for ann in anns]
  print(f"Area stats: min={min(areas)}, max={max(areas)}, mean={sum(areas)/len(areas)}")
  ```
- **目的**：确认小目标占比，验证是否需要更高分辨率（如 D2/D5）。

## 5. 问题与优化

### 5.1 适配器无效
- **问题**：适配器未提升 mAP，分类头变化小。
- **解决**：
  - 使用 MLP 适配器，适配所有特征层。
  - 增加训练 epoch（200+）和学习率（0.001）。
  - 检查 `train_adapter2.py` 日志，确认损失收敛。

### 5.2 小目标检测瓶颈
- **问题**：`AP@small=0.001`，D1 性能有限。
- **解决**：
  - 切换到 D2（768x768 分辨率）。
  - 增加输入分辨率（`input_sizes[1]=768`）。
  - 调整锚框比例（`projects/m3fd.yml`）以适配小目标。

### 5.3 D5 训练错误
- **问题**：`running_mean` 维度不匹配（48 vs. 32）。
- **解决**：修改 `train.py`，忽略不匹配参数，重置 BatchNorm。

## 6. 后续步骤
- **验证 D2 性能**：完成 D2 训练和测试，比较 mAP 和 `AP@small`。
- **优化适配器**：确保 `Classification difference` 增加。
- **硬件评估**：确认 GPU 内存支持 D2/D5（D2 约 6-8GB，D5 约 12-16GB）。
- **数据检查**：分析 `m3fd` 和 `ctir` 数据分布差异，优化预处理（`mean=[0.512], std=[0.231]`）。