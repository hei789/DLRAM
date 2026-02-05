# DLRAM - 多模态预训练

基于对比学习的文本-视觉模态表征对齐预训练框架。

## 环境要求

```bash
# Python 3.7+
# PyTorch 1.9+
# Transformers 4.0+

pip install torch==1.9.0
pip install transformers==4.20.0
pip install Pillow numpy
```

## 模型下载

本项目使用以下预训练模型（首次运行会自动下载）：

| 模型 | 用途 | 下载路径 |
|------|------|----------|
| BERT-base | 文本编码器 | `bert-base-uncased` |
| ViT-B/16 | 视觉编码器 | `google/vit-base-patch16-224` |

如需手动下载：
```python
from transformers import BertModel, ViTModel

# 文本编码器
bert = BertModel.from_pretrained("bert-base-uncased")
bert.save_pretrained("./models/bert-base-uncased")

# 视觉编码器
vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
vit.save_pretrained("./models/vit-base-patch16-224")
```

## 项目结构

```
DLRAM/
├── pretrain.py          # 主训练脚本
├── utils.py             # 工具函数
├── data_prepare.py      # 数据准备脚本
├── README.md            # 本文件
└── example_data/        # 示例数据目录（需自行准备）
    ├── train.json
    ├── images/
    └── masked_images/
```

## 数据格式

### 预训练数据格式 (`train.json`)

```json
[
  {
    "text": "这张图片展示了一只猫在沙发上睡觉。",
    "image": "cat_001.jpg",
    "masked_image": "cat_001_masked.jpg",
    "entities": [
      {
        "text": "猫",
        "type": "animal",
        "bbox": [0.2, 0.3, 0.5, 0.7],
        "start_char": 9,
        "end_char": 10
      },
      {
        "text": "沙发",
        "type": "furniture",
        "bbox": [0.1, 0.5, 0.9, 0.9],
        "start_char": 11,
        "end_char": 13
      }
    ],
    "objects": [
      {
        "label": "cat",
        "bbox": [0.2, 0.3, 0.5, 0.7],
        "confidence": 0.95
      },
      {
        "label": "sofa",
        "bbox": [0.1, 0.5, 0.9, 0.9],
        "confidence": 0.88
      }
    ]
  }
]
```

字段说明：
- `text`: 文本描述
- `image`: 原图文件名
- `masked_image`: 遮盖后的图片文件名（用于负例）
- `entities`: 文本中的实体列表
  - `text`: 实体文本
  - `type`: 实体类型
  - `bbox`: 实体对应的图片区域 [x1, y1, x2, y2]（归一化坐标）
  - `start_char`, `end_char`: 实体在文本中的字符位置
- `objects`: 图片中的目标列表
  - `label`: 目标标签
  - `bbox`: 目标位置 [x1, y1, x2, y2]
  - `confidence`: 检测置信度

## 快速开始

### 1. 准备遮盖图片

```bash
# 使用随机遮盖
python data_prepare.py \
    --task mask_images \
    --image_dir ./data/images \
    --output_dir ./data/masked_images \
    --mask_type random \
    --mask_ratio 0.3

# 使用目标感知遮盖（需要目标检测标注）
python data_prepare.py \
    --task mask_images \
    --image_dir ./data/images \
    --output_dir ./data/masked_images \
    --mask_annotation ./data/object_annotations.json \
    --mask_type object_aware \
    --mask_ratio 0.5
```

### 2. 创建预训练数据

```bash
python data_prepare.py \
    --task create_data \
    --text_file ./data/text_annotations.jsonl \
    --image_dir ./data/images \
    --annotation_file ./data/object_annotations.json \
    --output_file ./data/pretrain_data.json \
    --masked_image_dir ./data/masked_images
```

### 3. 划分训练/验证集

```bash
python data_prepare.py \
    --task split \
    --data_file ./data/pretrain_data.json \
    --train_file ./data/train.json \
    --val_file ./data/val.json \
    --train_ratio 0.9
```

### 4. 运行训练

```bash
python pretrain.py
```

或修改配置后运行：

```python
from pretrain import main, MultiModalEncoder, train
from torch.utils.data import DataLoader
import torch

# 自定义配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalEncoder(
    bert_model_name="bert-base-uncased",
    vit_model_name="google/vit-base-patch16-224"
)

# 准备数据
train_dataset = PretrainDataset(
    data_file="path/to/train.json",
    image_dir="path/to/images",
    masked_image_dir="path/to/masked_images",
    tokenizer=model.text_tokenizer,
    image_processor=model.image_processor
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,  # N=16: 1正例15负例
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, model.text_tokenizer, model.image_processor)
)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
train(model, train_dataloader, optimizer, device, num_epochs=10)
```

## 训练配置

### 粗粒度对比学习（文本-图片）

- **正例**: 文本 - 对应原图
- **负例**: 文本 - 对应遮盖后的图片
- **Batch Size**: 16（对应1正例15负例）

### 细粒度对比学习（实体-目标）

- **正例**: 实体 - 对应视觉目标（通过IoU匹配）
- **负例**: 实体 - 同一图片中的其他视觉目标

## 核心类说明

### MultiModalEncoder
多模态编码器，包含BERT文本编码器和ViT视觉编码器。

**主要方法:**
- `encode_text(text, device)`: 编码单条文本
- `encode_image(image, device)`: 编码单张图片
- `forward(text_inputs, image_inputs)`: 批量前向传播

### MultiGranularityPretrainer
多粒度预训练器，实现粗粒度和细粒度对比学习。

**主要方法:**
- `coarse_grained_loss()`: 计算粗粒度对比损失
- `fine_grained_loss()`: 计算细粒度对比损失
- `train_step()`: 单步训练

### PretrainDataset
预训练数据集类。

## 评估

训练过程中会保存模型检查点到 `./checkpoints/` 目录。

评估检索性能：

```python
from utils import evaluate_retrieval

metrics = evaluate_retrieval(
    model,
    val_dataloader,
    device,
    k_values=[1, 5, 10]
)

print(f"Image-to-Text R@1: {metrics['I2T_R@1']:.2f}")
print(f"Text-to-Image R@1: {metrics['T2I_R@1']:.2f}")
```

## 注意事项

1. **显存需求**: Batch Size=16时，需要约8GB显存
2. **细粒度损失**: 当前实现为简化版本，需根据实际数据格式调整
3. **遮盖策略**: 建议尝试不同的mask_type找到最佳效果

## 引用

```bibtex
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={NAACL-HLT},
  year={2019}
}

@inproceedings{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  booktitle={ICLR},
  year={2021}
}
```
