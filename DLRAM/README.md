# DLRAM - 多模态预训练

基于对比学习的文本-视觉模态表征对齐预训练框架，支持两个层次的对比学习：
1. **文本-图片层次**（粗粒度）
2. **实体-目标层次**（细粒度）

## 环境要求

```bash
# Python 3.8+
# PyTorch 1.9+
# Transformers 4.20+
# NumPy, Pillow

pip install torch>=1.9.0
pip install transformers>=4.20.0
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
├── data_prepare.py      # 数据准备脚本（旧版）
├── data/                # 数据处理目录
│   ├── preprocess.py    # 数据预处理脚本
│   ├── Json/            # 原始JSON数据
│   ├── Images/          # 原始图片
│   ├── Xml/             # 目标检测标注（XML格式）
│   ├── Npz/             # VinVL目标识别结果
│   └── processed/       # 处理后数据集（自动生成）
│       ├── text_image/              # 文本-图片正例
│       ├── text_image_negative/     # 文本-图片负例
│       ├── entity_object/           # 实体-目标正例
│       └── entity_object_negative/  # 实体-目标负例
└── README.md            # 本文件
```

## 数据预处理

### 原始数据格式

#### 1. JSON数据 (`Json/entity_text_test.json`)
```json
[
  {
    "content": "The geometry of plants . Garfield Park Conservatory",
    "img_id": "IMGID:71044",
    "ans": [
      {"ent": "Garfield Park"}
    ]
  }
]
```

#### 2. 图片数据 (`Images/`)
- 存放原始图片，文件名格式：`{img_id}.jpg`
- 支持 `IMGID:xxx` 和 `O_xxx` 两种ID格式

#### 3. XML标注 (`Xml/`)
```xml
<annotation>
  <filename>16_05_01_6.jpg</filename>
  <object>
    <name>JUSTIN BIEBER</name>
    <bndbox>
      <xmin>95</xmin>
      <ymin>329</ymin>
      <xmax>402</xmax>
      <ymax>1024</ymax>
    </bndbox>
  </object>
</annotation>
```

#### 4. VinVL结果 (`Npz/`)
- NPZ格式文件，包含目标检测的bounding boxes和scores
- 用于构建文本-图片层次的负例

### 运行预处理

```bash
cd data
python preprocess.py
```

预处理将生成以下数据集结构：

```
data/processed/
├── text_image/              # 文本-图片层次正例
│   ├── json/text_image_pairs.json
│   └── images/              # 复制的原始图片
├── text_image_negative/     # 文本-图片层次负例
│   └── json/text_image_negatives.json
├── entity_object/           # 实体-目标层次正例
│   ├── json/entity_object_pairs.json
│   └── crops/               # 裁剪的目标图片
└── entity_object_negative/  # 实体-目标层次负例
    └── json/entity_object_negatives.json
```

### 预处理输出格式

#### 文本-图片正例 (`text_image_pairs.json`)
```json
[
  {
    "content": "文本内容",
    "img_id": "IMGID:71044"
  }
]
```

#### 文本-图片负例 (`text_image_negatives.json`)
```json
[
  {
    "content": "文本内容",
    "img_id": "IMGID:71044",
    "mask_boxes": [
      {"box": [xmin, ymin, xmax, ymax], "score": 0.59, "object": "leaf"},
      ...  // 共15个需要遮盖的目标
    ]
  }
]
```

#### 实体-目标正例 (`entity_object_pairs.json`)
```json
[
  {
    "entity": "Garfield Park",
    "crop_img_id": "71044_crop_0.jpg",
    "source_img_id": "IMGID:71044"
  }
]
```

#### 实体-目标负例 (`entity_object_negatives.json`)
```json
[
  {
    "entity": "Garfield Park",
    "positive_img_id": "71044_crop_0.jpg",
    "negative_img_ids": ["xxx_crop_1.jpg", "yyy_crop_5.jpg", ...]  // 15个负例
  }
]
```

## 快速开始

### 1. 准备数据

确保数据目录结构如下：
```
data/
├── Json/entity_text_test.json
├── Images/
├── Xml/
└── Npz/
```

### 2. 运行预处理

```bash
cd data
python preprocess.py
```

预处理将自动：
- 构建文本-图片正例数据集
- 使用VinVL结果构建文本-图片负例（遮盖高score目标）
- 构建实体-目标正例数据集（裁剪目标区域）
- 随机采样构建实体-目标负例

### 3. 运行训练

```bash
# 返回项目根目录
cd ..

# 运行训练
python pretrain.py
```

## 训练配置

### 文本-图片层次对比学习

- **正例**: 文本 ↔ 原始图片
- **负例**: 文本 ↔ 遮盖后的图片（使用VinVL检测到的目标区域进行遮盖）
- **负例数量**: 15个（通过遮盖15个目标区域构建）
- **损失函数**: InfoNCE Loss

### 实体-目标层次对比学习

- **正例**: 实体文本 ↔ 对应的目标裁剪图片
- **负例**: 实体文本 ↔ 其他15个随机采样的目标图片
- **损失函数**: InfoNCE Loss

### 联合训练

```python
# 总损失 = w1 * 文本图片损失 + w2 * 实体目标损失
total_loss = coarse_weight * text_image_loss + fine_weight * entity_object_loss
```

默认配置：`coarse_weight=1.0`, `fine_weight=1.0`

## 核心类说明

### MultiModalEncoder
多模态编码器，包含BERT文本编码器和ViT视觉编码器。

**主要方法:**
- `encode_text(text_inputs)`: 批量编码文本
- `encode_image(image_inputs)`: 批量编码图片
- `forward(text_inputs, image_inputs)`: 批量前向传播

### TextImageDataset
文本-图片层次数据集。

**功能:**
- 同时加载正例和负例JSON
- 根据img_id匹配正负例

### EntityObjectDataset
实体-目标层次数据集。

**功能:**
- 加载正例实体-目标对
- 通过负例JSON获取每个正例对应的15个负例

### ContrastivePretrainer
对比学习预训练器。

**主要方法:**
- `text_image_loss()`: 计算文本-图片层次对比损失
- `entity_object_loss()`: 计算实体-目标层次对比损失
- `train_step_text_image()`: 文本-图片单步训练
- `train_step_entity_object()`: 实体-目标单步训练

## 自定义训练

```python
from pretrain import (
    MultiModalEncoder,
    TextImageDataset,
    EntityObjectDataset,
    collate_fn_text_image,
    collate_fn_entity_object,
    train
)
from torch.utils.data import DataLoader
import torch
from pathlib import Path

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_base_dir = Path("./data/processed")

# 初始化模型
model = MultiModalEncoder(
    bert_model_name="bert-base-uncased",
    vit_model_name="google/vit-base-patch16-224"
)

# 准备文本-图片数据
text_image_dataset = TextImageDataset(
    positive_json=str(data_base_dir / "text_image/json/text_image_pairs.json"),
    negative_json=str(data_base_dir / "text_image_negative/json/text_image_negatives.json"),
    images_dir=str(data_base_dir / "text_image/images"),
    tokenizer=model.text_tokenizer,
    image_processor=model.image_processor
)

text_image_dataloader = DataLoader(
    text_image_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: collate_fn_text_image(batch, model.text_tokenizer, model.image_processor),
    num_workers=4
)

# 准备实体-目标数据
entity_object_dataset = EntityObjectDataset(
    positive_json=str(data_base_dir / "entity_object/json/entity_object_pairs.json"),
    negative_json=str(data_base_dir / "entity_object_negative/json/entity_object_negatives.json"),
    crops_dir=str(data_base_dir / "entity_object/crops"),
    tokenizer=model.text_tokenizer,
    image_processor=model.image_processor
)

entity_object_dataloader = DataLoader(
    entity_object_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: collate_fn_entity_object(batch, model.text_tokenizer, model.image_processor),
    num_workers=4
)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 训练
train(
    model,
    text_image_dataloader,
    entity_object_dataloader,
    optimizer,
    device,
    num_epochs=10,
    save_dir="./checkpoints",
    coarse_weight=1.0,
    fine_weight=1.0,
    save_steps=1000
)
```

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

1. **显存需求**: Batch Size=32时，需要约12GB显存，可根据硬件调整
2. **数据预处理**: 必须先运行 `preprocess.py` 生成处理后的数据集
3. **负例构建**:
   - 文本-图片负例通过遮盖VinVL检测目标构建
   - 实体-目标负例通过随机采样其他目标构建
4. **图片格式**: 预处理时会自动将RGBA图片转换为RGB格式

## 文件说明

| 文件 | 说明 |
|------|------|
| `pretrain.py` | 主训练脚本，包含模型定义、数据集类和训练循环 |
| `utils.py` | 工具函数，包括IoU计算、特征提取、图片遮盖等 |
| `data/preprocess.py` | 数据预处理脚本，构建正负例数据集 |
| `data_prepare.py` | 旧版数据准备脚本（保留备用） |

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
