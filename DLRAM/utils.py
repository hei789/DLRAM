"""
工具函数模块
包含实体-目标对齐、特征提取、数据处理等功能
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple, Optional
import numpy as np


def get_token_positions(
    tokenizer,
    text: str,
    entity_text: str
) -> Tuple[int, int]:
    """
    获取实体在tokenized文本中的位置
    Returns:
        start_pos: 实体起始token位置
        end_pos: 实体结束token位置
    """
    # Tokenize文本
    tokens = tokenizer.tokenize(text)

    # 查找实体token
    entity_tokens = tokenizer.tokenize(entity_text)

    # 简单的子序列匹配
    for i in range(len(tokens) - len(entity_tokens) + 1):
        if tokens[i:i+len(entity_tokens)] == entity_tokens:
            return i, i + len(entity_tokens)

    return -1, -1


def extract_entity_features_from_output(
    hidden_states: torch.Tensor,
    start_pos: int,
    end_pos: int,
    method: str = "mean"
) -> torch.Tensor:
    """
    从BERT输出中提取实体特征
    Args:
        hidden_states: (batch_size, seq_len, hidden_dim)
        start_pos: 实体起始位置
        end_pos: 实体结束位置
        method: 聚合方法 ["mean", "max", "first", "last"]
    Returns:
        entity_feature: (batch_size, hidden_dim)
    """
    if start_pos < 0 or end_pos < 0:
        # 如果找不到位置，返回[CLS]表征
        return hidden_states[:, 0, :]

    entity_hidden = hidden_states[:, start_pos:end_pos, :]

    if method == "mean":
        return entity_hidden.mean(dim=1)
    elif method == "max":
        return entity_hidden.max(dim=1)[0]
    elif method == "first":
        return entity_hidden[:, 0, :]
    elif method == "last":
        return entity_hidden[:, -1, :]
    else:
        return entity_hidden.mean(dim=1)


def crop_object_from_image(
    image: Image.Image,
    bbox: List[float],
    expand_ratio: float = 0.1
) -> Image.Image:
    """
    根据bbox从图片中裁剪出目标
    Args:
        image: PIL Image
        bbox: [x1, y1, x2, y2] 归一化或绝对坐标
        expand_ratio: 扩展比例
    Returns:
        cropped_image: PIL Image
    """
    width, height = image.size

    # 转换为绝对坐标
    if max(bbox) <= 1.0:
        x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    else:
        x1, y1, x2, y2 = bbox

    # 扩展bbox
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * expand_ratio)
    y1 = max(0, y1 - h * expand_ratio)
    x2 = min(width, x2 + w * expand_ratio)
    y2 = min(height, y2 + h * expand_ratio)

    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def create_entity_object_pairs(
    entities: List[Dict],
    objects: List[Dict],
    iou_threshold: float = 0.5
) -> List[Tuple[int, int]]:
    """
    创建实体和目标之间的配对关系
    基于IoU或位置关系进行匹配
    Args:
        entities: [{"text": str, "bbox": [x1,y1,x2,y2], ...}, ...]
        objects: [{"label": str, "bbox": [x1,y1,x2,y2], ...}, ...]
        iou_threshold: IoU阈值
    Returns:
        pairs: [(entity_idx, object_idx), ...]
    """
    pairs = []

    for e_idx, entity in enumerate(entities):
        entity_bbox = entity.get('bbox', None)
        if entity_bbox is None:
            continue

        best_match = -1
        best_iou = 0

        for o_idx, obj in enumerate(objects):
            obj_bbox = obj.get('bbox', None)
            if obj_bbox is None:
                continue

            iou = compute_iou(entity_bbox, obj_bbox)
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = o_idx

        if best_match >= 0:
            pairs.append((e_idx, best_match))

    return pairs


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个bbox的IoU
    Args:
        box1, box2: [x1, y1, x2, y2]
    Returns:
        iou: float
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # 计算并集
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x2_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def hard_negative_mining(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    num_negatives: int = 15
) -> torch.Tensor:
    """
    硬负例挖掘
    选择相似度高但非正例的样本作为负例
    Args:
        text_features: (batch_size, dim)
        image_features: (batch_size, dim)
        num_negatives: 每个正例对应的负例数量
    Returns:
        negative_indices: (batch_size, num_negatives)
    """
    batch_size = text_features.shape[0]

    # 计算相似度矩阵
    similarity = torch.matmul(text_features, image_features.T)

    # 对每个文本，选择最相似的非对应图片
    negative_indices = []
    for i in range(batch_size):
        # 排除正例（对角线）
        sim_scores = similarity[i].clone()
        sim_scores[i] = -float('inf')

        # 选择top-k最相似的作为硬负例
        _, top_indices = torch.topk(sim_scores, min(num_negatives, batch_size - 1))
        negative_indices.append(top_indices)

    return torch.stack(negative_indices)


def info_nce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE损失计算
    Args:
        query: (batch_size, dim)
        positive: (batch_size, dim)
        negatives: (batch_size, num_negatives, dim)
        temperature: 温度参数
    Returns:
        loss: scalar
    """
    batch_size = query.shape[0]

    # 正例相似度
    pos_sim = torch.sum(query * positive, dim=-1) / temperature  # (batch_size,)

    # 负例相似度
    neg_sim = torch.matmul(query, negatives.transpose(1, 2)) / temperature  # (batch_size, num_negatives)

    # 计算logits和labels
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_negatives)
    labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)

    return F.cross_entropy(logits, labels)


def mask_image_regions(
    image: Image.Image,
    bboxes: List[List[float]],
    mask_type: str = "black"
) -> Image.Image:
    """
    遮盖图片的特定区域
    Args:
        image: PIL Image
        bboxes: 需要遮盖的区域列表 [[x1,y1,x2,y2], ...]
        mask_type: 遮盖类型 ["black", "white", "noise", "blur"]
    Returns:
        masked_image: PIL Image
    """
    result = image.copy()
    draw = ImageDraw.Draw(result)

    width, height = image.size

    for bbox in bboxes:
        # 转换为绝对坐标
        if max(bbox) <= 1.0:
            x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
        else:
            x1, y1, x2, y2 = bbox

        if mask_type == "black":
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        elif mask_type == "white":
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        elif mask_type == "noise":
            # 创建噪声区域
            noise = np.random.randint(0, 256, (int(y2-y1), int(x2-x1), 3), dtype=np.uint8)
            noise_img = Image.fromarray(noise)
            result.paste(noise_img, (int(x1), int(y1)))

    return result


def evaluate_retrieval(
    model,
    dataloader,
    device,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    评估图文检索性能
    Args:
        model: 多模态模型
        dataloader: 数据加载器
        device: 设备
        k_values: Recall@K的K值列表
    Returns:
        metrics: 评估指标
    """
    model.eval()

    all_text_features = []
    all_image_features = []

    with torch.no_grad():
        for batch in dataloader:
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            image_inputs = {k: v.to(device) for k, v in batch['original_image_inputs'].items()}

            text_features, image_features = model(text_inputs, image_inputs)

            all_text_features.append(text_features.cpu())
            all_image_features.append(image_features.cpu())

    all_text_features = torch.cat(all_text_features, dim=0)
    all_image_features = torch.cat(all_image_features, dim=0)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(all_text_features, all_image_features.T)

    # 图文检索: 给定文本，检索图片
    i2t_ranks = []
    for i in range(len(all_text_features)):
        sims = similarity_matrix[i]
        ranking = torch.argsort(sims, descending=True)
        rank = (ranking == i).nonzero(as_tuple=True)[0].item()
        i2t_ranks.append(rank)

    # 图文检索: 给定图片，检索文本
    t2i_ranks = []
    for i in range(len(all_image_features)):
        sims = similarity_matrix[:, i]
        ranking = torch.argsort(sims, descending=True)
        rank = (ranking == i).nonzero(as_tuple=True)[0].item()
        t2i_ranks.append(rank)

    # 计算指标
    metrics = {}
    for k in k_values:
        metrics[f'I2T_R@{k}'] = sum(1 for r in i2t_ranks if r < k) / len(i2t_ranks) * 100
        metrics[f'T2I_R@{k}'] = sum(1 for r in t2i_ranks if r < k) / len(t2i_ranks) * 100

    metrics['I2T_MedianRank'] = np.median(i2t_ranks)
    metrics['T2I_MedianRank'] = np.median(t2i_ranks)

    return metrics


def freeze_layers(model, layer_type: str = "text", num_layers: int = 6):
    """
    冻结模型的部分层
    Args:
        model: 多模态模型
        layer_type: "text" 或 "visual"
        num_layers: 冻结的层数
    """
    if layer_type == "text":
        encoder = model.text_encoder
    else:
        encoder = model.visual_encoder

    # 冻结嵌入层
    for param in encoder.embeddings.parameters():
        param.requires_grad = False

    # 冻结指定数量的Transformer层
    for i in range(min(num_layers, len(encoder.encoder.layer))):
        for param in encoder.encoder.layer[i].parameters():
            param.requires_grad = False

    print(f"Frozen first {num_layers} layers of {layer_type} encoder")
