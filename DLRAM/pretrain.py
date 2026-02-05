"""
多模态预训练模块 - DLRAM
任务：文本-视觉模态表征对齐
使用对比学习进行预训练（支持文本-图片和实体-目标两个层次）
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
import numpy as np
from pathlib import Path

from utils import (
    get_token_positions,
    extract_entity_features_from_output,
    crop_object_from_image,
    mask_image_regions
)


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

    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码文本，返回投影后的表征"""
        text_outputs = self.text_encoder(**text_inputs)
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_cls)
        return F.normalize(text_features, dim=-1)

    def encode_image(self, image_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码图片，返回投影后的表征"""
        image_outputs = self.visual_encoder(**image_inputs)
        image_cls = image_outputs.last_hidden_state[:, 0, :]
        image_features = self.visual_projection(image_cls)
        return F.normalize(image_features, dim=-1)

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
        text_features = self.encode_text(text_inputs)
        image_features = self.encode_image(image_inputs)
        return text_features, image_features


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数 (InfoNCE)
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


class TextImageDataset(Dataset):
    """
    文本-图片层次数据集
    支持正例（原始图片）和负例（遮盖后的图片）
    """
    def __init__(
        self,
        positive_json: str,
        negative_json: str,
        images_dir: str,
        tokenizer,
        image_processor,
        max_length: int = 512
    ):
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        # 加载正例和负例数据
        with open(positive_json, 'r', encoding='utf-8') as f:
            self.positive_data = json.load(f)
        with open(negative_json, 'r', encoding='utf-8') as f:
            self.negative_data = json.load(f)

        # 构建索引映射
        self.data = []
        for pos_item in self.positive_data:
            img_id = pos_item['img_id']
            # 找到对应的负例
            neg_item = None
            for neg in self.negative_data:
                if neg['img_id'] == img_id:
                    neg_item = neg
                    break

            if neg_item:
                self.data.append({
                    'content': pos_item['content'],
                    'img_id': img_id,
                    'mask_boxes': neg_item.get('mask_boxes', [])
                })

    def __len__(self):
        return len(self.data)

    def img_id_to_filename(self, img_id: str) -> str:
        """将img_id转换为文件名"""
        if img_id.startswith('IMGID:'):
            return img_id.replace('IMGID:', '') + '.jpg'
        return img_id + '.jpg'

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        img_id = item['img_id']
        img_filename = self.img_id_to_filename(img_id)
        img_path = os.path.join(self.images_dir, img_filename)

        return {
            'text': item['content'],
            'img_id': img_id,
            'image_path': img_path,
            'mask_boxes': item.get('mask_boxes', [])
        }


class EntityObjectDataset(Dataset):
    """
    实体-目标层次数据集
    支持正例和负例（从其他目标图片中随机选择）
    """
    def __init__(
        self,
        positive_json: str,
        negative_json: str,
        crops_dir: str,
        tokenizer,
        image_processor,
        max_length: int = 512
    ):
        self.crops_dir = crops_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        # 加载正例数据
        with open(positive_json, 'r', encoding='utf-8') as f:
            self.positive_data = json.load(f)

        # 加载负例数据并构建映射
        with open(negative_json, 'r', encoding='utf-8') as f:
            negative_data = json.load(f)

        self.negative_map = {}
        for neg_item in negative_data:
            key = (neg_item['entity'], neg_item['positive_img_id'])
            self.negative_map[key] = neg_item['negative_img_ids']

        # 过滤掉没有负例的数据
        self.data = []
        for pos_item in self.positive_data:
            key = (pos_item['entity'], pos_item['crop_img_id'])
            if key in self.negative_map:
                self.data.append(pos_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        entity = item['entity']
        positive_img_id = item['crop_img_id']
        key = (entity, positive_img_id)

        # 正例图片路径
        positive_img_path = os.path.join(self.crops_dir, positive_img_id)

        # 负例图片路径
        negative_img_ids = self.negative_map.get(key, [])
        negative_img_paths = [
            os.path.join(self.crops_dir, nid)
            for nid in negative_img_ids
        ]

        return {
            'entity': entity,
            'positive_image_path': positive_img_path,
            'negative_image_paths': negative_img_paths,
            'num_negatives': len(negative_img_paths)
        }


def collate_fn_text_image(batch: List[Dict], tokenizer, image_processor) -> Dict:
    """文本-图片层次的批次数据整理"""
    texts = [item['text'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    mask_boxes_list = [item['mask_boxes'] for item in batch]

    # 处理文本
    text_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # 处理原图
    original_images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            original_images.append(img)
        except Exception as e:
            print(f"警告：无法打开图片 {path}: {e}")
            # 使用空白图片替代
            original_images.append(Image.new('RGB', (224, 224), (128, 128, 128)))

    original_image_inputs = image_processor(images=original_images, return_tensors="pt")

    # 生成遮盖后的图片
    masked_images = []
    for img, mask_boxes in zip(original_images, mask_boxes_list):
        if mask_boxes:
            # 提取bbox列表
            bboxes = [mb['box'] for mb in mask_boxes[:15]]  # 最多取15个
            masked_img = mask_image_regions(img.copy(), bboxes, mask_type="black")
            masked_images.append(masked_img)
        else:
            # 如果没有mask_boxes，使用原图
            masked_images.append(img)

    masked_image_inputs = image_processor(images=masked_images, return_tensors="pt")

    return {
        'text_inputs': text_inputs,
        'original_image_inputs': original_image_inputs,
        'masked_image_inputs': masked_image_inputs,
        'texts': texts
    }


def collate_fn_entity_object(batch: List[Dict], tokenizer, image_processor) -> Dict:
    """实体-目标层次的批次数据整理"""
    entities = [item['entity'] for item in batch]
    positive_image_paths = [item['positive_image_path'] for item in batch]
    negative_image_paths_list = [item['negative_image_paths'] for item in batch]

    # 处理实体文本
    entity_inputs = tokenizer(
        entities,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    # 处理正例图片
    positive_images = []
    for path in positive_image_paths:
        try:
            img = Image.open(path).convert('RGB')
            positive_images.append(img)
        except Exception as e:
            print(f"警告：无法打开图片 {path}: {e}")
            positive_images.append(Image.new('RGB', (224, 224), (128, 128, 128)))

    positive_image_inputs = image_processor(images=positive_images, return_tensors="pt")

    # 处理负例图片（展平为列表）
    negative_images = []
    negative_labels = []  # 记录每个实体的负例数量

    for neg_paths in negative_image_paths_list:
        neg_count = 0
        for path in neg_paths:
            try:
                img = Image.open(path).convert('RGB')
                negative_images.append(img)
                neg_count += 1
            except Exception as e:
                print(f"警告：无法打开负例图片 {path}: {e}")
        negative_labels.append(neg_count)

    if negative_images:
        negative_image_inputs = image_processor(images=negative_images, return_tensors="pt")
    else:
        # 如果没有负例，创建一个空张量
        negative_image_inputs = {
            'pixel_values': torch.zeros(0, 3, 224, 224)
        }

    return {
        'entity_inputs': entity_inputs,
        'positive_image_inputs': positive_image_inputs,
        'negative_image_inputs': negative_image_inputs,
        'negative_labels': negative_labels,  # 每个实体的负例数量
        'entities': entities
    }


class ContrastivePretrainer:
    """
    对比学习预训练器
    包含文本-图片和实体-目标两个层次的对比学习
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
        self.temperature = temperature
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight

    def text_image_loss(
        self,
        text_features: torch.Tensor,
        original_image_features: torch.Tensor,
        masked_image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        文本-图片层次对比学习损失
        正例：文本 - 原图
        负例：文本 - 遮盖后的图片（同一批次中的其他样本作为额外负例）
        """
        batch_size = text_features.shape[0]

        # 合并图片特征: (batch_size + batch_size, dim) = (2*batch_size, dim)
        all_image_features = torch.cat([original_image_features, masked_image_features], dim=0)

        # 计算相似度: (batch_size, 2*batch_size)
        logits = torch.matmul(text_features, all_image_features.T) / self.temperature

        # 标签：正例位于0到batch_size-1的位置
        labels = torch.arange(batch_size, device=self.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

    def entity_object_loss(
        self,
        entity_features: torch.Tensor,
        positive_image_features: torch.Tensor,
        negative_image_features: torch.Tensor,
        negative_labels: List[int]
    ) -> torch.Tensor:
        """
        实体-目标层次对比学习损失
        正例：实体 - 对应的目标图片
        负例：实体 - 其他目标图片
        """
        batch_size = entity_features.shape[0]

        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)

        # 计算正例相似度
        pos_sim = torch.sum(entity_features * positive_image_features, dim=-1) / self.temperature  # (batch_size,)

        # 如果存在负例，计算负例相似度
        if negative_image_features.shape[0] > 0:
            # 将负例分组（每个实体对应的负例）
            start_idx = 0
            all_logits = []

            for i, num_neg in enumerate(negative_labels):
                end_idx = start_idx + num_neg

                if num_neg > 0 and end_idx <= negative_image_features.shape[0]:
                    # 获取该实体的负例特征
                    neg_features = negative_image_features[start_idx:end_idx]  # (num_neg, dim)

                    # 计算负例相似度
                    neg_sim = torch.matmul(entity_features[i:i+1], neg_features.T) / self.temperature  # (1, num_neg)

                    # 合并正例和负例相似度
                    logits = torch.cat([pos_sim[i:i+1].unsqueeze(0), neg_sim], dim=1)  # (1, 1 + num_neg)
                    all_logits.append(logits)

                start_idx = end_idx

            if all_logits:
                # 计算损失
                logits = torch.cat(all_logits, dim=0)  # (batch_size, 1 + num_negatives)
                labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(logits, labels)
            else:
                # 如果没有负例，退化为简单的正例损失
                loss = -torch.mean(pos_sim)
        else:
            # 如果没有负例，退化为简单的正例损失
            loss = -torch.mean(pos_sim)

        return loss

    def train_step_text_image(self, batch: Dict[str, torch.Tensor]) -> float:
        """文本-图片层次的单步训练"""
        text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
        original_image_inputs = {k: v.to(self.device) for k, v in batch['original_image_inputs'].items()}
        masked_image_inputs = {k: v.to(self.device) for k, v in batch['masked_image_inputs'].items()}

        # 编码
        text_features = self.model.encode_text(text_inputs)
        original_image_features = self.model.encode_image(original_image_inputs)
        masked_image_features = self.model.encode_image(masked_image_inputs)

        # 计算损失
        loss = self.text_image_loss(
            text_features,
            original_image_features,
            masked_image_features
        )

        return loss

    def train_step_entity_object(self, batch: Dict[str, torch.Tensor]) -> float:
        """实体-目标层次的单步训练"""
        entity_inputs = {k: v.to(self.device) for k, v in batch['entity_inputs'].items()}
        positive_image_inputs = {k: v.to(self.device) for k, v in batch['positive_image_inputs'].items()}
        negative_image_inputs = {k: v.to(self.device) for k, v in batch['negative_image_inputs'].items()}
        negative_labels = batch['negative_labels']

        # 编码
        entity_features = self.model.encode_text(entity_inputs)
        positive_image_features = self.model.encode_image(positive_image_inputs)

        # 编码负例图片
        if negative_image_inputs['pixel_values'].shape[0] > 0:
            negative_image_features = self.model.encode_image(negative_image_inputs)
        else:
            negative_image_features = torch.zeros(0, entity_features.shape[1], device=self.device)

        # 计算损失
        loss = self.entity_object_loss(
            entity_features,
            positive_image_features,
            negative_image_features,
            negative_labels
        )

        return loss


def train(
    model: MultiModalEncoder,
    text_image_dataloader: DataLoader,
    entity_object_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    save_dir: str = "./checkpoints",
    coarse_weight: float = 1.0,
    fine_weight: float = 1.0,
    save_steps: int = 1000
):
    """训练循环"""
    os.makedirs(save_dir, exist_ok=True)

    pretrainer = ContrastivePretrainer(
        model, device,
        coarse_weight=coarse_weight,
        fine_weight=fine_weight
    )

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        text_image_loss_sum = 0.0
        entity_object_loss_sum = 0.0
        total_loss_sum = 0.0

        # 创建迭代器
        text_image_iter = iter(text_image_dataloader)
        entity_object_iter = iter(entity_object_dataloader)

        # 计算每个epoch的步数（取两者最大值）
        max_steps = max(len(text_image_dataloader), len(entity_object_dataloader))

        for step in range(max_steps):
            optimizer.zero_grad()

            total_loss = 0.0

            # 1. 文本-图片层次训练
            try:
                batch_ti = next(text_image_iter)
            except StopIteration:
                text_image_iter = iter(text_image_dataloader)
                batch_ti = next(text_image_iter)

            loss_ti = pretrainer.train_step_text_image(batch_ti)
            total_loss += coarse_weight * loss_ti
            text_image_loss_sum += loss_ti.item()

            # 2. 实体-目标层次训练
            try:
                batch_eo = next(entity_object_iter)
            except StopIteration:
                entity_object_iter = iter(entity_object_dataloader)
                batch_eo = next(entity_object_iter)

            loss_eo = pretrainer.train_step_entity_object(batch_eo)
            total_loss += fine_weight * loss_eo
            entity_object_loss_sum += loss_eo.item()

            # 反向传播
            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            global_step += 1

            # 打印日志
            if global_step % 10 == 0:
                print(f"Epoch {epoch}, Step {global_step}, "
                      f"Total Loss: {total_loss.item():.4f}, "
                      f"TI Loss: {loss_ti.item():.4f}, "
                      f"EO Loss: {loss_eo.item():.4f}")

            # 保存检查点
            if global_step % save_steps == 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, f"checkpoint_step_{global_step}.pt"))

        # Epoch结束，打印平均损失
        avg_total_loss = total_loss_sum / max_steps
        avg_ti_loss = text_image_loss_sum / max_steps
        avg_eo_loss = entity_object_loss_sum / max_steps

        print(f"\nEpoch {epoch} completed.")
        print(f"  Average Total Loss: {avg_total_loss:.4f}")
        print(f"  Average Text-Image Loss: {avg_ti_loss:.4f}")
        print(f"  Average Entity-Object Loss: {avg_eo_loss:.4f}")

        # 保存epoch检查点
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))

    print("Training completed!")


def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据路径配置
    data_base_dir = Path("./data/processed")

    # 正例数据路径
    text_image_positive_json = data_base_dir / "text_image" / "json" / "text_image_pairs.json"
    text_image_images_dir = data_base_dir / "text_image" / "images"

    entity_object_positive_json = data_base_dir / "entity_object" / "json" / "entity_object_pairs.json"
    entity_object_crops_dir = data_base_dir / "entity_object" / "crops"

    # 负例数据路径
    text_image_negative_json = data_base_dir / "text_image_negative" / "json" / "text_image_negatives.json"
    entity_object_negative_json = data_base_dir / "entity_object_negative" / "json" / "entity_object_negatives.json"

    # 检查文件是否存在
    for path in [text_image_positive_json, text_image_negative_json,
                 entity_object_positive_json, entity_object_negative_json]:
        if not path.exists():
            print(f"警告：数据文件不存在 {path}")
            print("请先运行 preprocess.py 构建数据集")
            return

    # 初始化模型
    model = MultiModalEncoder(
        bert_model_name="bert-base-uncased",
        vit_model_name="google/vit-base-patch16-224"
    )

    # 准备文本-图片层次数据
    text_image_dataset = TextImageDataset(
        positive_json=str(text_image_positive_json),
        negative_json=str(text_image_negative_json),
        images_dir=str(text_image_images_dir),
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

    # 准备实体-目标层次数据
    entity_object_dataset = EntityObjectDataset(
        positive_json=str(entity_object_positive_json),
        negative_json=str(entity_object_negative_json),
        crops_dir=str(entity_object_crops_dir),
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

    print(f"Text-Image dataset size: {len(text_image_dataset)}")
    print(f"Entity-Object dataset size: {len(entity_object_dataset)}")

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


if __name__ == "__main__":
    main()
