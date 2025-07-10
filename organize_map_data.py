import os
import shutil

def flatten_and_rename_folders(input_root, output_folder):
    """
    将 input_root 目录下所有结构为 date/part*/[数字文件夹] 的文件夹
    复制到 output_folder 下，重命名为 date_partX_num 形式。
    """
    os.makedirs(output_folder, exist_ok=True)

    for date_folder in ['0519', '0523', '0529']:
        date_path = os.path.join(input_root, date_folder)
        if not os.path.isdir(date_path):
            continue

        for part_folder in os.listdir(date_path):
            part_path = os.path.join(date_path, part_folder)
            if not os.path.isdir(part_path):
                continue

            for num_folder in os.listdir(part_path):
                num_path = os.path.join(part_path, num_folder)
                if not os.path.isdir(num_path):
                    continue

                new_folder_name = f"{date_folder}_{part_folder}_{num_folder}"
                dst_path = os.path.join(output_folder, new_folder_name)

                if os.path.exists(dst_path):
                    print(f"目标目录已存在，跳过：{dst_path}")
                else:
                    shutil.copytree(num_path, dst_path)
                    print(f"已复制：{num_path} -> {dst_path}")

# 使用方法
input_root = 'data/annotation/'   # 修改为你的输入根目录路径
output_folder = 'data/annotation/output'  # 修改为输出目录路径
flatten_and_rename_folders(input_root, output_folder)