"""
数据准备脚本
用于生成遮盖图片、准备预训练数据格式
"""

import os
import json
import argparse
from PIL import Image, ImageDraw
from typing import List, Dict
import random
from utils import mask_image_regions


def generate_masked_images(
    image_dir: str,
    output_dir: str,
    annotation_file: str = None,
    mask_ratio: float = 0.3,
    mask_type: str = "random"
):
    """
    生成遮盖后的图片
    Args:
        image_dir: 原图目录
        output_dir: 输出目录
        annotation_file: 标注文件（包含目标位置）
        mask_ratio: 遮盖比例
        mask_type: 遮盖类型 ["random", "object_aware", "center"]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载标注
    annotations = {}
    if annotation_file and os.path.exists(annotation_file):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        width, height = image.size

        # 确定遮盖区域
        bboxes = []

        if mask_type == "random":
            # 随机遮盖
            num_patches = max(1, int(mask_ratio * 10))
            for _ in range(num_patches):
                w = width * random.uniform(0.1, 0.3)
                h = height * random.uniform(0.1, 0.3)
                x1 = random.uniform(0, width - w)
                y1 = random.uniform(0, height - h)
                x2 = x1 + w
                y2 = y1 + h
                bboxes.append([x1/width, y1/height, x2/width, y2/height])

        elif mask_type == "object_aware" and img_name in annotations:
            # 遮盖目标区域
            objects = annotations[img_name].get('objects', [])
            num_to_mask = max(1, int(len(objects) * mask_ratio))
            selected = random.sample(objects, min(num_to_mask, len(objects)))
            for obj in selected:
                bboxes.append(obj['bbox'])

        elif mask_type == "center":
            # 遮盖中心区域
            cx, cy = 0.5, 0.5
            w, h = mask_ratio, mask_ratio
            bboxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        # 生成遮盖图片
        masked_image = mask_image_regions(image, bboxes, mask_type="black")

        # 保存
        output_path = os.path.join(output_dir, img_name)
        masked_image.save(output_path)
        print(f"Saved masked image: {output_path}")


def create_pretrain_data(
    text_file: str,
    image_dir: str,
    annotation_file: str,
    output_file: str,
    masked_image_dir: str = None
):
    """
    创建预训练数据文件
    Args:
        text_file: 文本文件，每行一个json {"text": "...", "image": "...", "entities": [...]}
        image_dir: 图片目录
        annotation_file: 目标检测标注文件
        output_file: 输出文件
        masked_image_dir: 遮盖图片目录
    """
    # 加载目标检测标注
    object_annotations = {}
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            object_annotations = json.load(f)

    # 处理文本数据
    data = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())

            img_name = item.get('image', '')

            # 构建数据项
            data_item = {
                'text': item['text'],
                'image': img_name,
                'entities': item.get('entities', []),
                'objects': object_annotations.get(img_name, {}).get('objects', [])
            }

            if masked_image_dir:
                data_item['masked_image'] = img_name

            data.append(data_item)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Created pretrain data with {len(data)} items at {output_file}")


def split_data(
    data_file: str,
    train_file: str,
    val_file: str,
    train_ratio: float = 0.9
):
    """
    划分训练集和验证集
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Split data: {len(train_data)} train, {len(val_data)} val")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for pretraining')
    parser.add_argument('--task', type=str, required=True,
                        choices=['mask_images', 'create_data', 'split'],
                        help='Task to perform')

    # mask_images参数
    parser.add_argument('--image_dir', type=str, help='Image directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--mask_annotation', type=str, help='Annotation file for masking')
    parser.add_argument('--mask_ratio', type=float, default=0.3)
    parser.add_argument('--mask_type', type=str, default='random',
                        choices=['random', 'object_aware', 'center'])

    # create_data参数
    parser.add_argument('--text_file', type=str, help='Text file')
    parser.add_argument('--annotation_file', type=str, help='Object annotation file')
    parser.add_argument('--output_file', type=str, help='Output file')
    parser.add_argument('--masked_image_dir', type=str, help='Masked image directory')

    # split参数
    parser.add_argument('--data_file', type=str, help='Data file to split')
    parser.add_argument('--train_file', type=str, help='Train output file')
    parser.add_argument('--val_file', type=str, help='Val output file')
    parser.add_argument('--train_ratio', type=float, default=0.9)

    args = parser.parse_args()

    if args.task == 'mask_images':
        generate_masked_images(
            args.image_dir,
            args.output_dir,
            args.mask_annotation,
            args.mask_ratio,
            args.mask_type
        )
    elif args.task == 'create_data':
        create_pretrain_data(
            args.text_file,
            args.image_dir,
            args.annotation_file,
            args.output_file,
            args.masked_image_dir
        )
    elif args.task == 'split':
        split_data(
            args.data_file,
            args.train_file,
            args.val_file,
            args.train_ratio
        )


if __name__ == '__main__':
    main()
