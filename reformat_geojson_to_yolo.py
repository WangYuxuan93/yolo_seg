import os
import json
import shutil
import cv2
import numpy as np
from PIL import Image

# 原始路径
base_dir = 'data/real_geo_map/'
gold_base_dir = base_dir.rstrip('/\\') + '_gold'
os.makedirs(gold_base_dir, exist_ok=True)

# 遍历子文件夹
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    image_path = os.path.join(folder_path, 'image', 'image.tif')
    json_path = os.path.join(folder_path, 'meta', 'Drawings.geojson')

    if not os.path.exists(image_path) or not os.path.exists(json_path):
        print(f"Skipping folder {folder} due to missing files.")
        continue

    # gold 路径结构
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

    output_lines = []

    for feature in data.get('features', []):
        if feature['properties'].get('type') != 'legend_patch':
            continue

        label = feature['properties'].get('label', 'Unknown').replace("\n","")
        multipolygon = feature['geometry'].get('coordinates', [])

        for polygon in multipolygon:
            for ring in polygon:
                if len(ring) < 3:
                    continue

                # 去掉首尾重复点（闭合点）
                if ring[0] == ring[-1]:
                    ring = ring[:-1]

                # 检查是否为4个点
                if len(ring) != 4:
                    print(f"[Warning] Non-rectangle legend_patch in folder '{folder}' with label '{label}' has {len(ring)} points.")
                    print("          Coordinates:", ring)
                
                # 保存输出（保留浮点数）
                coords = [f"{v:.6f}" for point in ring for v in point]
                line = f"{label} " + ' '.join(coords)
                output_lines.append(line)

                # 绘制多边形（绘图仍然转 int）
                pts = np.array([[int(round(x)), int(round(y))] for x, y in ring], dtype=np.int32)
                cv2.polylines(image_draw, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # 保存图像
    out_img_path = os.path.join(gold_item_dir, 'gold_item_box.png')
    Image.fromarray(image_draw).save(out_img_path)

    # 保存文本
    item_txt_path = os.path.join(gold_item_dir, 'item_box.txt')
    if not output_lines:
        print ("{} is empty".format(folder))
    with open(item_txt_path, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

print("✅ 所有文件处理完成，已保存到 gold 目录中。")