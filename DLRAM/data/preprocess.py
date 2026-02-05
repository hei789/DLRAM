#!/usr/bin/env python3
"""
用于构建对比学习数据集的预处理脚本
包含两个层次：
1. 文本-图片层次：将文本和对应的整图配对
2. 实体-目标层次：将实体和对应的目标区域（裁剪后的图片）配对
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import random
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple

# 设置随机种子以保证可重复性
random.seed(42)


def parse_xml(xml_path: str) -> Dict:
    """
    解析XML文件，提取图片信息和目标边界框

    Args:
        xml_path: XML文件路径

    Returns:
        包含文件名和目标列表的字典
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    objects = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        objects.append({
            'name': name,
            'bbox': (xmin, ymin, xmax, ymax)
        })

    return {
        'filename': filename,
        'objects': objects
    }


def img_id_to_filename(img_id: str) -> str:
    """
    将img_id转换为文件名
    例如：IMGID:71044 -> 71044.jpg
             O_2371 -> O_2371.jpg

    Args:
        img_id: JSON中的img_id字段

    Returns:
        对应的图片文件名
    """
    if img_id.startswith('IMGID:'):
        return img_id.replace('IMGID:', '') + '.jpg'
    return img_id + '.jpg'


def build_text_image_dataset(
    json_path: str,
    images_dir: str,
    output_json_dir: str,
    output_images_dir: str
) -> None:
    """
    构建文本-图片层次的数据集

    输出结构：
    - output_json_dir/: 存放包含content和img_id的JSON文件
    - output_images_dir/: 存放对应的完整图片

    Args:
        json_path: 原始JSON文件路径
        images_dir: 原始图片文件夹路径
        output_json_dir: 输出JSON文件夹路径
        output_images_dir: 输出图片文件夹路径
    """
    # 创建输出目录
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)

    # 读取原始JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备新的数据集
    dataset = []
    copied_images = set()

    for item in data:
        img_id = item['img_id']
        img_filename = img_id_to_filename(img_id)
        src_img_path = os.path.join(images_dir, img_filename)

        # 检查图片是否存在
        if not os.path.exists(src_img_path):
            print(f"警告：图片不存在 {src_img_path}，跳过该条目")
            continue

        # 复制图片（只复制一次）
        if img_filename not in copied_images:
            dst_img_path = os.path.join(output_images_dir, img_filename)
            shutil.copy2(src_img_path, dst_img_path)
            copied_images.add(img_filename)

        # 添加到数据集
        dataset.append({
            'content': item['content'],
            'img_id': img_id
        })

    # 保存新的JSON文件
    output_json_path = os.path.join(output_json_dir, 'text_image_pairs.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"文本-图片层次数据集构建完成：")
    print(f"  - 共处理 {len(dataset)} 条文本-图片对")
    print(f"  - JSON文件保存至: {output_json_path}")
    print(f"  - 图片保存至: {output_images_dir}")


def build_entity_object_dataset(
    json_path: str,
    images_dir: str,
    xml_dir: str,
    output_json_dir: str,
    output_crops_dir: str
) -> None:
    """
    构建实体-目标层次的数据集

    输出结构：
    - output_json_dir/: 存放包含entity和crop_img_id的JSON文件
    - output_crops_dir/: 存放裁剪后的目标图片

    Args:
        json_path: 原始JSON文件路径
        images_dir: 原始图片文件夹路径
        xml_dir: XML标注文件夹路径
        output_json_dir: 输出JSON文件夹路径
        output_crops_dir: 输出裁剪图片文件夹路径
    """
    # 创建输出目录
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_crops_dir, exist_ok=True)

    # 读取原始JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 加载所有XML文件，建立映射
    xml_cache = {}
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            xml_info = parse_xml(xml_path)
            # 使用文件名（不含扩展名）作为键
            key = Path(xml_info['filename']).stem
            xml_cache[key] = xml_info

    # 准备数据集和统计信息
    dataset = []
    crop_counter = 0
    total_entities = 0       # 总实体数
    matched_entities = 0     # 成功匹配到视觉目标的实体数
    skipped_no_xml = 0       # 因无XML文件被跳过的实体数
    skipped_no_match = 0     # 因在XML中无匹配目标被跳过的实体数

    for item in data:
        img_id = item['img_id']
        img_filename = img_id_to_filename(img_id)
        img_stem = Path(img_filename).stem

        # 检查是否有对应的XML文件
        if img_stem not in xml_cache:
            # 统计该条目下所有实体（都因无XML而被跳过）
            entities_count = len(item.get('ans', []))
            skipped_no_xml += entities_count
            total_entities += entities_count
            continue

        xml_info = xml_cache[img_stem]
        src_img_path = os.path.join(images_dir, img_filename)

        # 检查原图是否存在
        if not os.path.exists(src_img_path):
            print(f"警告：图片不存在 {src_img_path}，跳过")
            entities_count = len(item.get('ans', []))
            skipped_no_xml += entities_count
            total_entities += entities_count
            continue

        # 打开原图
        try:
            img = Image.open(src_img_path)
        except Exception as e:
            print(f"错误：无法打开图片 {src_img_path}: {e}")
            entities_count = len(item.get('ans', []))
            skipped_no_xml += entities_count
            total_entities += entities_count
            continue

        # 获取该条目中的所有实体
        entities = item.get('ans', [])

        for ent_obj in entities:
            entity_name = ent_obj.get('ent', '')
            total_entities += 1

            if not entity_name:
                skipped_no_match += 1
                continue

            # 在XML中查找匹配的目标
            matched_obj = None
            for obj in xml_info['objects']:
                # 简单的字符串匹配（可以改进为更灵活的匹配方式）
                if entity_name.lower() in obj['name'].lower() or \
                   obj['name'].lower() in entity_name.lower():
                    matched_obj = obj
                    break

            # 忽略没有对应视觉目标的实体
            if matched_obj is None:
                skipped_no_match += 1
                continue

            matched_entities += 1

            # 裁剪目标区域
            xmin, ymin, xmax, ymax = matched_obj['bbox']

            # 确保坐标有效
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img.width, xmax)
            ymax = min(img.height, ymax)

            if xmax <= xmin or ymax <= ymin:
                print(f"警告：无效边界框 {matched_obj['bbox']} for {entity_name}")
                matched_entities -= 1
                skipped_no_match += 1
                continue

            cropped_img = img.crop((xmin, ymin, xmax, ymax))

            # 如果是RGBA模式，转换为RGB（JPEG不支持透明通道）
            if cropped_img.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', cropped_img.size, (255, 255, 255))
                background.paste(cropped_img, mask=cropped_img.split()[3])  # 使用alpha通道作为mask
                cropped_img = background
            elif cropped_img.mode != 'RGB':
                cropped_img = cropped_img.convert('RGB')

            # 保存裁剪后的图片
            crop_filename = f"{img_stem}_crop_{crop_counter}.jpg"
            crop_path = os.path.join(output_crops_dir, crop_filename)
            cropped_img.save(crop_path, 'JPEG')

            # 添加到数据集
            dataset.append({
                'entity': entity_name,
                'crop_img_id': crop_filename,
                'source_img_id': img_id
            })

            crop_counter += 1

        img.close()

    # 保存新的JSON文件
    output_json_path = os.path.join(output_json_dir, 'entity_object_pairs.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n实体-目标层次数据集构建完成：")
    print(f"  - 总实体数: {total_entities}")
    print(f"  - 成功匹配（有视觉目标）: {matched_entities}")
    print(f"  - 被忽略的实体（无视觉目标）: {skipped_no_match + skipped_no_xml}")
    print(f"      - 无XML标注文件: {skipped_no_xml}")
    print(f"      - XML中无匹配目标: {skipped_no_match}")
    print(f"  - 最终生成 {len(dataset)} 条实体-目标对")
    print(f"  - JSON文件保存至: {output_json_path}")
    print(f"  - 裁剪图片保存至: {output_crops_dir}")

    return dataset  # 返回正例数据集，用于构建负例


def build_text_image_negatives(
    json_path: str,
    npz_dir: str,
    output_json_dir: str,
    num_negatives: int = 15
) -> None:
    """
    构建文本-图片层次的负例数据集

    使用VinVL目标识别结果，通过遮盖检测到的目标来构建负例。
    选择score高的目标进行遮盖，如果不足15个则循环使用高score目标补充。

    输出结构：
    - output_json_dir/: 存放包含content、img_id、mask_boxes的JSON文件
      mask_boxes: 需要遮盖的边界框列表

    Args:
        json_path: 原始JSON文件路径
        npz_dir: VinVL npz结果文件夹路径
        output_json_dir: 输出JSON文件夹路径
        num_negatives: 每个样本需要的负例数量（默认15）
    """
    os.makedirs(output_json_dir, exist_ok=True)

    # 读取原始JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 加载所有NPZ文件，建立映射
    npz_cache = {}
    for npz_file in os.listdir(npz_dir):
        if npz_file.endswith('.npz'):
            npz_path = os.path.join(npz_dir, npz_file)
            # 使用文件名（不含扩展名）作为键
            key = Path(npz_file).stem
            npz_data = np.load(npz_path, allow_pickle=True)
            npz_cache[key] = {
                'boxes': npz_data['bounding_boxes'],
                'scores': npz_data['scores'],
                'objects': npz_data['objects']
            }

    # 准备负例数据集
    negative_dataset = []

    for item in data:
        img_id = item['img_id']
        img_stem = img_id_to_filename(img_id).replace('.jpg', '')

        # 检查是否有对应的NPZ文件
        if img_stem not in npz_cache:
            continue

        npz_info = npz_cache[img_stem]
        boxes = npz_info['boxes']
        scores = npz_info['scores']

        num_boxes = len(boxes)
        if num_boxes == 0:
            continue

        # 按score降序排序，获取索引
        sorted_indices = np.argsort(scores)[::-1]

        # 构建负例：选择需要遮盖的bounding box
        mask_boxes_list = []

        for i in range(num_negatives):
            # 循环使用高score的目标
            idx = sorted_indices[i % num_boxes]
            box = boxes[idx].tolist()

            mask_boxes_list.append({
                'box': box,  # [xmin, ymin, xmax, ymax]
                'score': float(scores[idx]),
                'object': str(npz_info['objects'][idx])
            })

        # 添加到负例数据集
        negative_dataset.append({
            'content': item['content'],
            'img_id': img_id,
            'mask_boxes': mask_boxes_list  # 需要遮盖的目标列表
        })

    # 保存负例JSON文件
    output_json_path = os.path.join(output_json_dir, 'text_image_negatives.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(negative_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n文本-图片层次负例数据集构建完成：")
    print(f"  - 共生成 {len(negative_dataset)} 条负例样本")
    print(f"  - 每条负例包含 {num_negatives} 个需要遮盖的目标")
    print(f"  - JSON文件保存至: {output_json_path}")


def build_entity_object_negatives(
    positive_dataset: List[Dict],
    output_json_dir: str,
    num_negatives: int = 15
) -> None:
    """
    构建实体-目标层次的负例数据集

    为每个正例实体-目标对，从其他目标图片中随机挑选指定数量的负例。

    输出结构：
    - output_json_dir/: 存放包含entity、positive_img_id、negative_img_ids的JSON文件

    Args:
        positive_dataset: 正例数据集（实体-目标对列表）
        output_json_dir: 输出JSON文件夹路径
        num_negatives: 每个正例需要的负例数量（默认15）
    """
    os.makedirs(output_json_dir, exist_ok=True)

    if len(positive_dataset) == 0:
        print("警告：正例数据集为空，无法构建负例")
        return

    # 获取所有可用的crop图片ID列表
    all_crop_ids = [item['crop_img_id'] for item in positive_dataset]

    negative_dataset = []

    for pos_item in positive_dataset:
        entity = pos_item['entity']
        positive_crop_id = pos_item['crop_img_id']

        # 从其他crop图片中随机选择负例（排除正例本身）
        available_negatives = [cid for cid in all_crop_ids if cid != positive_crop_id]

        if len(available_negatives) == 0:
            continue

        # 随机选择num_negatives个负例（如果不够就全选）
        if len(available_negatives) >= num_negatives:
            negative_crop_ids = random.sample(available_negatives, num_negatives)
        else:
            # 如果不够，允许重复选择
            negative_crop_ids = random.choices(available_negatives, k=num_negatives)

        negative_dataset.append({
            'entity': entity,
            'positive_img_id': positive_crop_id,
            'negative_img_ids': negative_crop_ids
        })

    # 保存负例JSON文件
    output_json_path = os.path.join(output_json_dir, 'entity_object_negatives.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(negative_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n实体-目标层次负例数据集构建完成：")
    print(f"  - 共生成 {len(negative_dataset)} 条负例样本")
    print(f"  - 每条负例包含 {num_negatives} 个负例图片")
    print(f"  - JSON文件保存至: {output_json_path}")


def main():
    """主函数：配置路径并构建两个层次的数据集（正例+负例）"""

    # ============ 配置路径 ============
    # 原始数据路径
    base_dir = Path(__file__).parent / 'data'
    json_path = base_dir / 'Json' / 'entity_text_test.json'
    images_dir = base_dir / 'Images'
    xml_dir = base_dir / 'Xml'
    npz_dir = base_dir / 'Npz'

    # 输出路径
    output_base = base_dir / 'processed'

    # 正例输出路径
    text_image_json_dir = output_base / 'text_image' / 'json'
    text_image_img_dir = output_base / 'text_image' / 'images'
    entity_object_json_dir = output_base / 'entity_object' / 'json'
    entity_object_crop_dir = output_base / 'entity_object' / 'crops'

    # 负例输出路径
    text_image_neg_dir = output_base / 'text_image_negative' / 'json'
    entity_object_neg_dir = output_base / 'entity_object_negative' / 'json'

    # ============ 执行预处理 ============
    print("=" * 50)
    print("开始构建对比学习数据集")
    print("=" * 50)

    # 1. 构建文本-图片层次正例数据集
    print("\n[1/4] 构建文本-图片层次正例数据集...")
    build_text_image_dataset(
        json_path=str(json_path),
        images_dir=str(images_dir),
        output_json_dir=str(text_image_json_dir),
        output_images_dir=str(text_image_img_dir)
    )

    # 2. 构建实体-目标层次正例数据集
    print("\n[2/4] 构建实体-目标层次正例数据集...")
    positive_dataset = build_entity_object_dataset(
        json_path=str(json_path),
        images_dir=str(images_dir),
        xml_dir=str(xml_dir),
        output_json_dir=str(entity_object_json_dir),
        output_crops_dir=str(entity_object_crop_dir)
    )

    # 3. 构建文本-图片层次负例数据集
    print("\n[3/4] 构建文本-图片层次负例数据集...")
    build_text_image_negatives(
        json_path=str(json_path),
        npz_dir=str(npz_dir),
        output_json_dir=str(text_image_neg_dir),
        num_negatives=15
    )

    # 4. 构建实体-目标层次负例数据集
    print("\n[4/4] 构建实体-目标层次负例数据集...")
    build_entity_object_negatives(
        positive_dataset=positive_dataset,
        output_json_dir=str(entity_object_neg_dir),
        num_negatives=15
    )

    print("\n" + "=" * 50)
    print("所有数据集构建完成！")
    print("=" * 50)
    print(f"\n输出结构：")
    print(f"{output_base}/")
    print(f"├── text_image/              # 文本-图片正例")
    print(f"│   ├── json/text_image_pairs.json")
    print(f"│   └── images/")
    print(f"├── text_image_negative/     # 文本-图片负例")
    print(f"│   └── json/text_image_negatives.json")
    print(f"├── entity_object/           # 实体-目标正例")
    print(f"│   ├── json/entity_object_pairs.json")
    print(f"│   └── crops/")
    print(f"└── entity_object_negative/  # 实体-目标负例")
    print(f"    └── json/entity_object_negatives.json")


if __name__ == '__main__':
    main()
