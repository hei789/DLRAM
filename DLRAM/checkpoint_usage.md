# Checkpoint 使用说明

## Checkpoint 文件内容

保存的 `checkpoint.pt` 文件（或 `best_checkpoint.pt`）包含**整个多模态编码器**（`MultiModalEncoder`）的完整状态：

1. **文本编码器** (`text_encoder`)：BERT-base
2. **视觉编码器** (`visual_encoder`)：ViT-B/16
3. **投影层** (`text_projection`, `visual_projection`)：将特征映射到共享空间
4. **温度参数** (`temperature`)：对比学习的温度系数
5. **优化器状态** (`optimizer_state_dict`)：用于恢复训练

## 如何使用 Checkpoint

### 1. 加载完整模型

```python
import torch
from pretrain import MultiModalEncoder  # 导入模型定义

# 初始化模型（与训练时相同的配置）
model = MultiModalEncoder(
    bert_model_name="./models/bert-base-uncased",
    vit_model_name="./models/vit-base-patch16-224"
)

# 加载 checkpoint
checkpoint_path = "./checkpoints/best_checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 加载模型权重
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # 切换到评估模式

# 查看保存时的信息
print(f"Loaded from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
print(f"Best loss: {checkpoint.get('best_loss', 'N/A')}")
```

### 2. 编码文本

```python
import torch.nn.functional as F

text = "A person is walking on the street."
text_inputs = model.text_tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
)

with torch.no_grad():
    text_features = model.encode_text(text_inputs)  # (1, 256) 归一化后的特征
```

### 3. 编码图片

```python
from PIL import Image

image = Image.open("example.jpg").convert("RGB")
image_inputs = model.image_processor(images=[image], return_tensors="pt")

with torch.no_grad():
    image_features = model.encode_image(image_inputs)  # (1, 256) 归一化后的特征
```

### 4. 计算文本-图片相似度

```python
# 特征已经归一化，直接点积即可得到余弦相似度
similarity = torch.matmul(text_features, image_features.T)
print(f"Text-Image similarity: {similarity.item():.4f}")
```

## 单独使用部分编码器

### 只用文本编码器

```python
text_encoder = model.text_encoder
text_projection = model.text_projection

# 使用方式
text_inputs = model.text_tokenizer(["example text"], return_tensors="pt", padding=True)
outputs = text_encoder(**text_inputs)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
projected = text_projection(cls_embedding)  # 投影到共享空间
```

### 只用视觉编码器

```python
visual_encoder = model.visual_encoder
visual_projection = model.visual_projection

# 使用方式
image_inputs = model.image_processor(images=[image], return_tensors="pt")
outputs = visual_encoder(**image_inputs)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
projected = visual_projection(cls_embedding)  # 投影到共享空间
```

## Checkpoint 文件结构

```python
{
    'epoch': 5,                          # 保存时的 epoch
    'global_step': 10000,                # 全局步数
    'model_state_dict': {...},           # 模型权重
    'optimizer_state_dict': {...},       # 优化器状态（用于恢复训练）
    'best_loss': 0.1234                  # 最佳损失值（仅 best_checkpoint.pt 有）
}
```

## 恢复训练

如果需要从 checkpoint 恢复训练：

```python
model = MultiModalEncoder(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 加载模型和优化器状态
checkpoint = torch.load("./checkpoints/best_checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# 恢复训练起始位置
start_epoch = checkpoint["epoch"] + 1
global_step = checkpoint["global_step"]
```

## 应用场景

1. **文本-图像检索**：计算查询文本与候选图像的相似度，返回最相似的图像
2. **图像-文本检索**：计算查询图像与候选文本的相似度，返回最匹配的文本描述
3. **特征提取**：将文本或图像转换为固定维度的向量表示，用于下游任务
4. **零样本分类**：利用文本描述对图像进行分类
