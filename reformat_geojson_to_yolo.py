import os
import json
import shutil
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from difflib import SequenceMatcher

def label_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize_label(s):
    return s.replace("、", "_").replace(",", "_").replace("，", "_").strip()

# 路径设置
base_dir = 'data/annotation/output'
gold_base_dir = base_dir.rstrip('/\\') + '_gold'
os.makedirs(gold_base_dir, exist_ok=True)
type_counter = defaultdict(int)

# 遍历子文件夹
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    #print (folder)
    if not os.path.isdir(folder_path):
        continue

    image_path = os.path.join(folder_path, 'image', 'image.tif')
    json_path = os.path.join(folder_path, 'meta', 'Drawings.geojson')
    legend_dir = os.path.join(folder_path, 'legend')

    if not os.path.exists(image_path) or not os.path.exists(json_path):
        print(f"Skipping folder {folder} due to missing files.")
        continue

    # 读取 legend 文件名，建立 label -> type
    label_to_type = {}
    if os.path.isdir(legend_dir):
        for file in os.listdir(legend_dir):
            if not file.lower().endswith('.tif'):
                continue
            name, _ = os.path.splitext(file)
            if '_' not in name:
                continue
            label, label_type = name.rsplit('_', 1)
            label = normalize_label(label)

            # ✅ 统一为小写
            label_type = label_type.lower()

            # ✅ 修正常见拼写错误
            if label_type == 'piont':
                label_type = 'point'

            # ✅ 限定在允许的四类，否则归为 unknown
            if label_type not in {'line', 'point', 'poly', 'unknown'}:
                print(f"[Warning] 非法 label_type: '{label_type}' in file '{file}', 将重置为 'unknown'")
                label_type = 'unknown'

            label_to_type[label] = label_type

    # 准备 gold 输出路径
    target_folder = os.path.join(gold_base_dir, folder)
    image_target = os.path.join(target_folder, 'image')
    meta_target = os.path.join(target_folder, 'meta')
    gold_item_dir = os.path.join(target_folder, 'gold_item')
    os.makedirs(gold_item_dir, exist_ok=True)
    shutil.copytree(os.path.join(folder_path, 'image'), image_target, dirs_exist_ok=True)
    shutil.copytree(os.path.join(folder_path, 'meta'), meta_target, dirs_exist_ok=True)

    image = cv2.imread(image_path)
    image_draw = image.copy()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    geo_labels = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        ftype = props.get('type') or props.get('Type') or ''
        if ftype.lower() not in ['legend_patch', 'legend_patcg']:
            continue
        label = props.get('label') or props.get('Label') or props.get('lable') or 'Unknown'
        label = normalize_label(label)
        geo_labels.append(label)

    # ⚠️ 检查数量是否一致
    if len(geo_labels) != len(label_to_type):
        print(f"[Mismatch] Folder '{folder}': {len(geo_labels)} geojson labels vs {len(label_to_type)} legend labels")
        #exit()

    # 精确匹配 + 模糊匹配
    label_mapping = {}
    matched_set = set()
    for label in geo_labels:
        if label in label_to_type:
            label_mapping[label] = label_to_type[label]
            matched_set.add(label)
        else:
            # 模糊匹配
            best_match, best_score = None, 0.0
            for legend_label in label_to_type:
                sim = label_similarity(label, legend_label)
                if sim > best_score:
                    best_match, best_score = legend_label, sim
            if best_score >= 0.5:
                label_mapping[label] = label_to_type[best_match]
                matched_set.add(best_match)
                #print(f"[Fuzzy Match] '{label}' → '{best_match}' ({best_score:.2f})")
            else:
                label_mapping[label] = "unknown"
                print(f"[Unknown Label] folder='{folder}', label='{label}' (best match: '{best_match}' {best_score:.2f})")

    # 遍历数据并生成输出
    output_lines = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        ftype = props.get('type') or props.get('Type') or ''
        if ftype.lower() not in ['legend_patch', 'legend_patcg']:
            continue
        label = props.get('label') or props.get('Label') or props.get('lable') or 'Unknown'
        label = normalize_label(label)
        label_type = label_mapping.get(label, "unknown")

        type_counter[label_type] += 1

        multipolygon = feature.get('geometry', {}).get('coordinates', [])
        for polygon in multipolygon:
            for ring in polygon:
                if len(ring) < 3:
                    continue
                if ring[0] == ring[-1]:
                    ring = ring[:-1]
                if len(ring) != 4:
                    print(f"[Warning] [{folder}] Non-rectangle legend_patch: {len(ring)} pts, label: '{label}'")
                    print("           Coordinates:", ring)
                coords = [f"{v:.6f}" for point in ring for v in point]
                line = f"{label} {label_type} " + ' '.join(coords)
                output_lines.append(line)

                pts = np.array([[int(round(x)), int(round(y))] for x, y in ring], dtype=np.int32)
                cv2.polylines(image_draw, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # 保存图像和文本
    out_img_path = os.path.join(gold_item_dir, 'gold_item_box.png')
    Image.fromarray(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)).save(out_img_path)
    item_txt_path = os.path.join(gold_item_dir, 'item_box.txt')
    if not output_lines:
        print(f"{folder} is empty")
    with open(item_txt_path, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

# 打印统计
print("\n📊 各类别数量统计：")
for label_type, count in sorted(type_counter.items()):
    print(f"  {label_type:>8}: {count}")

print("\n✅ 所有文件处理完成，已保存到 gold 目录中。")