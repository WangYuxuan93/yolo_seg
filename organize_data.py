import os
import shutil
import json
import re
from pathlib import Path

def move_and_rename_folders_v0(input_dir, output_dir, json_path):
    os.makedirs(output_dir, exist_ok=True)
    mapping = {}

    for batch_folder in os.listdir(input_dir):
        batch_path = os.path.join(input_dir, batch_folder)
        if not os.path.isdir(batch_path):
            continue

        # 提取批次数字
        match = re.match(r'第(\d+)批', batch_folder)
        if not match:
            continue
        batch_num = match.group(1)

        for subfolder in os.listdir(batch_path):
            subfolder_path = os.path.join(batch_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            if not subfolder.isdigit():
                continue

            new_name = f"part{batch_num}_{subfolder}"
            new_path = os.path.join(output_dir, new_name)

            shutil.move(subfolder_path, new_path)
            mapping[subfolder_path] = new_path

    # 保存路径映射到 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

# 中文数字转阿拉伯数字的映射
chinese_num_map = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
}

def chinese_to_digit(chinese):
    """
    支持“第一”、“第二”、“第十”等中文批次转阿拉伯数字
    """
    chinese = chinese.replace("第", "").replace("批", "")
    if chinese == "十":
        return 10
    elif "十" in chinese:
        parts = chinese.split("十")
        if parts[0] == "":
            tens = 1
        else:
            tens = chinese_num_map.get(parts[0], 0)
        ones = chinese_num_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    else:
        return chinese_num_map.get(chinese, -1)

def move_and_rename_folders(input_dir, output_dir, json_path):
    os.makedirs(output_dir, exist_ok=True)
    mapping = {}

    for batch_folder in os.listdir(input_dir):
        batch_path = os.path.join(input_dir, batch_folder)
        if not os.path.isdir(batch_path):
            continue

        # 中文批次转数字
        batch_num = chinese_to_digit(batch_folder)
        if batch_num == -1:
            print(f"跳过无法识别的批次文件夹: {batch_folder}")
            continue

        for subfolder in os.listdir(batch_path):
            subfolder_path = os.path.join(batch_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            if not subfolder.isdigit():
                continue

            new_name = f"part{batch_num}_{subfolder}"
            new_path = os.path.join(output_dir, new_name)

            shutil.move(subfolder_path, new_path)
            mapping[subfolder_path] = new_path

    # 保存路径映射到 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


# 示例调用
input_dir = r'data/0529_real_map_origin'       # 替换为实际输入路径
output_dir = r'data/0529_real_map'     # 替换为实际输出路径
json_path = r'data/0529_real_map_origin/3_mapping.json'  # 保存映射的 json 文件路径

move_and_rename_folders(input_dir, output_dir, json_path)
