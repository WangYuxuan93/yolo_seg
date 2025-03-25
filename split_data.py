import os
import shutil
import random

def split_data(input_dir, output_dir, train_size):
    # 输入和输出目录
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    # 创建输出目录的结构
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    
    # 获取所有图片文件的文件名
    image_files = sorted(os.listdir(images_dir))
    
    # 根据训练集大小计算训练集和验证集的分割
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    # 复制训练集数据到输出目录
    for file in train_files:
        image_file = os.path.join(images_dir, file)
        label_file = os.path.join(labels_dir, file.replace('.jpg', '.txt'))
        
        shutil.copy(image_file, os.path.join(train_dir, 'images', file))
        shutil.copy(label_file, os.path.join(train_dir, 'labels', file.replace('.jpg', '.txt')))
    
    # 复制验证集数据到输出目录
    for file in val_files:
        image_file = os.path.join(images_dir, file)
        label_file = os.path.join(labels_dir, file.replace('.jpg', '.txt'))
        
        shutil.copy(image_file, os.path.join(val_dir, 'images', file))
        shutil.copy(label_file, os.path.join(val_dir, 'labels', file.replace('.jpg', '.txt')))

    print(f"Data split complete. Train size: {len(train_files)}, Val size: {len(val_files)}")

# 示例用法
input_directory = 'path/to/your/dataset'  # 替换为你的数据集路径
output_directory = 'path/to/output'  # 替换为输出路径
train_size = 1000  # 设置训练集大小（数量）

split_data(input_directory, output_directory, train_size)
