import json
import os

# 设置路径
input_json = "/workspace/Yet-Another-EfficientDet-Pytorch-master/datasets/m3fd/annotations/1_instances_val2007.json"  # 原始 JSON 文件
output_json = "/workspace/Yet-Another-EfficientDet-Pytorch-master/datasets/m3fd/annotations/instances_val2007.json"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_json), exist_ok=True)

# 读取 JSON 文件
try:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误：JSON 文件 {input_json} 不存在")
    exit(1)

# 创建 category_name 到 category_id 的映射
category_map = {cat["name"]: cat["id"] for cat in data["categories"]}

# 添加 id 和 category_id 到 annotations
for i, ann in enumerate(data["annotations"], 1):
    ann["id"] = i  # 添加唯一标注 ID（从 1 开始）
    if "category_name" in ann and ann["category_name"] in category_map:
        ann["category_id"] = category_map[ann["category_name"]]  # 添加 category_id
        del ann["category_name"]  # 删除 category_name（COCO 格式使用 category_id）
    ann["iscrowd"] = 0  # 添加 iscrowd 字段（通常为 0）

# 保存修改后的 JSON 文件
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
print(f"已生成 COCO 格式的 JSON 文件：{output_json}")