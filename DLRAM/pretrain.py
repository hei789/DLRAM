"""
多模态预训练模块 - DLRAM
任务：文本-视觉模态表征对齐
使用对比学习进行预训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor
from PIL import Image
import os
import json
from typing import Dict, List, Tuple, Optional
import random


class MultiModalEncoder(nn.Module):
    """
    多模态编码器
    文本编码器: BERT-base
    视觉编码器: ViT-B/16
    """
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        vit_model_name: str = "google/vit-base-patch16-224",
        embedding_dim: int = 768,
        projection_dim: int = 256
    ):
        super(MultiModalEncoder, self).__init__()

        # 文本编码器 - BERT-base
        self.text_encoder = BertModel.from_pretrained(bert_model_name)
        self.text_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # 视觉编码器 - ViT-B/16
        self.visual_encoder = ViTModel.from_pretrained(vit_model_name)
        self.image_processor = ViTImageProcessor.from_pretrained(vit_model_name)

        # 投影层 - 将特征映射到共享空间
        self.text_projection = nn.Linear(embedding_dim, projection_dim)
        self.visual_projection = nn.Linear(embedding_dim, projection_dim)

        # 温度参数
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def encode_text(self, text: str, device: torch.device) -> torch.Tensor:
        """编码文本，返回[CLS]表征"""
        inputs = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.text_encoder(**inputs)
            # 获取[CLS]标记的表征 (batch_size, hidden_size)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            # 投影到共享空间
            projected = self.text_projection(cls_embedding)
            return F.normalize(projected, dim=-1)

    def encode_image(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        """编码图片，返回[CLS]表征"""
        inputs = self.image_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.visual_encoder(**inputs)
            # 获取[CLS]标记的表征 (batch_size, hidden_size)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            # 投影到共享空间
            projected = self.visual_projection(cls_embedding)
            return F.normalize(projected, dim=-1)

    def forward(
        self,
        text_inputs: Dict[str, torch.Tensor],
        image_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Returns:
            text_features: (batch_size, projection_dim)
            image_features: (batch_size, projection_dim)
        """
        # 编码文本
        text_outputs = self.text_encoder(**text_inputs)
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_cls)
        text_features = F.normalize(text_features, dim=-1)

        # 编码图片
        image_outputs = self.visual_encoder(**image_inputs)
        image_cls = image_outputs.last_hidden_state[:, 0, :]
        image_features = self.visual_projection(image_cls)
        image_features = F.normalize(image_features, dim=-1)

        return text_features, image_features

    def encode_text_batch(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """批量编码文本"""
        text_outputs = self.text_encoder(**text_inputs)
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_cls)
        return F.normalize(text_features, dim=-1)

    def encode_image_batch(self, image_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """批量编码图片"""
        image_outputs = self.visual_encoder(**image_inputs)
        image_cls = image_outputs.last_hidden_state[:, 0, :]
        image_features = self.visual_projection(image_cls)
        return F.normalize(image_features, dim=-1)


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数
    支持InfoNCE Loss
    """
    def __init__(self, temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features_a: (batch_size, dim)
            features_b: (batch_size, dim)
            labels: (batch_size,) - 正例的索引，None表示i与i配对
        """
        batch_size = features_a.shape[0]

        # 计算相似度矩阵
        logits = torch.matmul(features_a, features_b.T) / self.temperature

        if labels is None:
            # 默认对角线为正例
            labels = torch.arange(batch_size, device=features_a.device)

        # 对称损失
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)

        return (loss_a + loss_b) / 2


class PretrainDataset(Dataset):
    """
    预训练数据集
    支持粗粒度和细粒度的数据加载
    """
    def __init__(
        self,
        data_file: str,
        image_dir: str,
        masked_image_dir: str,
        tokenizer,
        image_processor,
        max_length: int = 512
    ):
        self.data = self.load_data(data_file)
        self.image_dir = image_dir
        self.masked_image_dir = masked_image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def load_data(self, data_file: str) -> List[Dict]:
        """加载数据文件"""
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # 文本
        text = item['text']

        # 原图路径
        original_image_path = os.path.join(self.image_dir, item['image'])

        # 遮盖后的图片路径（用于负例）
        masked_image_path = os.path.join(self.masked_image_dir, item.get('masked_image', item['image']))

        # 实体和目标信息（用于细粒度）
        entities = item.get('entities', [])  # [{"text": "entity", "bbox": [x1,y1,x2,y2]}, ...]
        objects = item.get('objects', [])    # [{"label": "obj", "bbox": [x1,y1,x2,y2]}, ...]

        return {
            'text': text,
            'original_image_path': original_image_path,
            'masked_image_path': masked_image_path,
            'entities': entities,
            'objects': objects,
            'idx': idx
        }


def collate_fn(batch: List[Dict], tokenizer, image_processor) -> Dict[str, torch.Tensor]:
    """批次数据整理"""
    texts = [item['text'] for item in batch]
    original_image_paths = [item['original_image_path'] for item in batch]
    masked_image_paths = [item['masked_image_path'] for item in batch]
    entities_list = [item['entities'] for item in batch]
    objects_list = [item['objects'] for item in batch]

    # 处理文本
    text_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # 处理原图
    original_images = [Image.open(path).convert('RGB') for path in original_image_paths]
    original_image_inputs = image_processor(images=original_images, return_tensors="pt")

    # 处理遮盖后的图片
    masked_images = [Image.open(path).convert('RGB') for path in masked_image_paths]
    masked_image_inputs = image_processor(images=masked_images, return_tensors="pt")

    return {
        'text_inputs': text_inputs,
        'original_image_inputs': original_image_inputs,
        'masked_image_inputs': masked_image_inputs,
        'entities': entities_list,
        'objects': objects_list,
        'texts': texts
    }


class MultiGranularityPretrainer:
    """
    多粒度预训练器
    包含粗粒度（文本-图片）和细粒度（实体-目标）对比学习
    """
    def __init__(
        self,
        model: MultiModalEncoder,
        device: torch.device,
        temperature: float = 0.07,
        coarse_weight: float = 1.0,
        fine_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight

    def coarse_grained_loss(
        self,
        text_features: torch.Tensor,
        original_image_features: torch.Tensor,
        masked_image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        粗粒度对比学习损失
        正例：文本 - 原图
        负例：文本 - 遮盖后的图片
        """
        batch_size = text_features.shape[0]

        # 构建正负例特征
        # 正例：batch_size个原图
        # 负例：batch_size个遮盖图片
        # 构建 (batch_size, 2*batch_size) 的相似度矩阵

        # 合并图片特征: (2*batch_size, dim)
        all_image_features = torch.cat([original_image_features, masked_image_features], dim=0)

        # 计算相似度: (batch_size, 2*batch_size)
        logits = torch.matmul(text_features, all_image_features.T) / self.contrastive_loss.temperature

        # 标签：正例位于0到batch_size-1的位置
        labels = torch.arange(batch_size, device=self.device)

        return F.cross_entropy(logits, labels)

    def fine_grained_loss(
        self,
        text_features: torch.Tensor,
        entity_features: torch.Tensor,
        object_features: torch.Tensor,
        entity_to_object_mapping: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        细粒度对比学习损失
        正例：实体 - 对应的视觉目标
        负例：实体 - 其他视觉目标
        """
        if len(entity_features) == 0 or len(object_features) == 0:
            return torch.tensor(0.0, device=self.device)

        # 计算所有实体和所有目标之间的相似度
        logits = torch.matmul(entity_features, object_features.T) / self.contrastive_loss.temperature

        # 构建标签
        labels = torch.full((len(entity_features),), -1, device=self.device, dtype=torch.long)
        for e_idx, o_idx in entity_to_object_mapping:
            if e_idx < len(entity_features) and o_idx < len(object_features):
                labels[e_idx] = o_idx

        # 过滤掉没有对应关系的实体
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        return F.cross_entropy(valid_logits, valid_labels)

    def extract_entity_features(
        self,
        text: str,
        entities: List[Dict],
        text_encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        从文本编码输出中提取实体特征
        简化实现：使用实体的token平均表征
        """
        # 注意：这里简化处理，实际应该根据实体的token位置提取
        # 返回文本的整体表征作为实体表征的简化版本
        return text_encoder_output

    def extract_object_features(
        self,
        image: Image.Image,
        objects: List[Dict]
    ) -> torch.Tensor:
        """
        从图片中提取目标特征
        简化实现：使用图片的整体表征
        """
        # 注意：这里简化处理，实际应该根据bbox裁剪图片后编码
        return self.model.encode_image(image, self.device)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        # 将输入移到设备
        text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
        original_image_inputs = {k: v.to(self.device) for k, v in batch['original_image_inputs'].items()}
        masked_image_inputs = {k: v.to(self.device) for k, v in batch['masked_image_inputs'].items()}

        # 编码文本
        text_features = self.model.encode_text_batch(text_inputs)

        # 编码原图和遮盖图
        original_image_features = self.model.encode_image_batch(original_image_inputs)
        masked_image_features = self.model.encode_image_batch(masked_image_inputs)

        # 粗粒度损失
        coarse_loss = self.coarse_grained_loss(
            text_features,
            original_image_features,
            masked_image_features
        )

        # 细粒度损失（如果有实体和目标信息）
        fine_loss = 0.0
        if 'entities' in batch and 'objects' in batch:
            # 提取实体和目标特征
            # 注意：这里需要更复杂的实现来提取局部特征
            pass

        # 总损失
        total_loss = self.coarse_weight * coarse_loss + self.fine_weight * fine_loss

        return {
            'total_loss': total_loss.item(),
            'coarse_loss': coarse_loss.item(),
            'fine_loss': fine_loss if isinstance(fine_loss, float) else fine_loss.item()
        }


def train(
    model: MultiModalEncoder,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    save_dir: str = "./checkpoints"
):
    """训练循环"""
    os.makedirs(save_dir, exist_ok=True)

    pretrainer = MultiGranularityPretrainer(model, device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        coarse_loss_sum = 0.0
        fine_loss_sum = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 准备数据
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            original_image_inputs = {k: v.to(device) for k, v in batch['original_image_inputs'].items()}
            masked_image_inputs = {k: v.to(device) for k, v in batch['masked_image_inputs'].items()}

            # 前向传播
            text_features, original_image_features = model(text_inputs, original_image_inputs)
            _, masked_image_features = model(text_inputs, masked_image_inputs)

            # 计算损失
            coarse_loss = pretrainer.coarse_grained_loss(
                text_features,
                original_image_features,
                masked_image_features
            )

            # 这里可以添加细粒度损失
            total_loss_batch = coarse_loss  # + fine_weight * fine_loss

            # 反向传播
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            coarse_loss_sum += coarse_loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))

    print("Training completed!")


def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = MultiModalEncoder(
        bert_model_name="bert-base-uncased",
        vit_model_name="google/vit-base-patch16-224"
    )

    # 准备数据
    # 注意：这里需要根据实际情况修改路径
    train_dataset = PretrainDataset(
        data_file="data/train.json",
        image_dir="data/images",
        masked_image_dir="data/masked_images",
        tokenizer=model.text_tokenizer,
        image_processor=model.image_processor
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,  # N=16, 即1正例15负例
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model.text_tokenizer, model.image_processor)
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 训练
    train(model, train_dataloader, optimizer, device, num_epochs=10)


if __name__ == "__main__":
    main()
