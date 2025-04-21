import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Segment map images and save YOLO-format results.")
    parser.add_argument('--input_root', type=str, required=True, help="Root directory containing numbered subfolders with images")
    return parser.parse_args()

def load_model():
    #return YOLO("outputs/layout-bs256-gpu8-v0/train2/weights/best.pt")  # 修改为你的模型路径
    return YOLO("../model/layout-bs256-gpu8-v0/best.pt")  # 修改为你的模型路径


def predict_image_segmentation(image: np.ndarray, model_path: str, return_vis: bool = False):
    """
    对输入图像进行分割预测，返回分割轮廓坐标和可视化图像（可选）
    """
    model = YOLO(model_path)
    h_img, w_img = image.shape[:2]
    results = model(image)

    predictions = []
    vis_img = image.copy()

    # 固定类别颜色
    colors = {
        'main map': (255, 0, 0),   # 红
        'legend': (0, 0, 255),     # 蓝
        'item': (0, 255, 0),       # 绿
        'compass': (0, 255, 255),
        'scale': (255, 255, 255),
        'title': (255, 255, 0)
    }

    # 类别优先级
    priority = {
        'main map': 0,
        'legend': 1,
        'item': 2,
        'compass': 3,
        'scale': 4,
        'title': 5
    }

    # 所有掩码合并后的颜色层
    combined_mask = np.zeros_like(vis_img, dtype=np.uint8)

    for result in results:
        if result.masks is None or result.boxes is None or len(result.boxes.cls) == 0:
            continue  # 跳过无预测目标的图片

        # 每个实例打包并排序
        targets = []
        for i, mask in enumerate(result.masks.data):
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]
            targets.append((priority.get(class_name, 6), i, class_id, class_name, mask))

        # 排序：根据类别优先级
        targets.sort(key=lambda x: x[0])

        for _, i, class_id, class_name, mask in targets:
            color = colors.get(class_name, np.random.randint(0, 255, 3).tolist())
            mask_resized = cv2.resize(mask.cpu().numpy(), (w_img, h_img))
            mask_binary = np.uint8(mask_resized * 255)

            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            main_contour = max(contours, key=cv2.contourArea)
            if len(main_contour) < 3:
                continue

            points = [(int(pt[0][0]), int(pt[0][1])) for pt in main_contour]
            predictions.append({
                'class_id': class_id,
                'class_name': class_name,
                'points': points
            })

            # 合并掩码到颜色图层（用于透明叠加）
            combined_mask[mask_resized > 0] = color

            # 画轮廓边框
            cv2.drawContours(vis_img, [main_contour], -1, color, 2)

            # 类别名称
            x, y, w, h = cv2.boundingRect(main_contour)
            cv2.putText(vis_img, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 最后一次性进行透明颜色叠加
    if return_vis:
        vis_img = cv2.addWeighted(vis_img, 0.9, combined_mask, 0.3, 0)

    return (predictions, vis_img) if return_vis else predictions

def predict_image_segmentation_v2(image: np.ndarray, model_path: str, return_vis: bool = False):
    """
    对输入图像进行分割预测，返回分割轮廓坐标和可视化图像（可选）

    参数：
        image (np.ndarray): 输入图像（BGR格式）
        model_path (str): YOLO 模型路径
        return_vis (bool): 是否返回可视化图像

    返回：
        List[Dict]: 每个目标的预测信息，包括：
            - class_id (int)
            - class_name (str)
            - points: List of (x, y) tuples（轮廓坐标）
        vis_img (np.ndarray, optional): 可视化图像（若 return_vis=True）
    """
    model = YOLO(model_path)
    results = model(image)

    predictions = []
    vis_img = image.copy()

    # 固定颜色
    colors = {
        'main map': (255, 0, 0),
        'legend': (0, 0, 255),
        'item': (0, 255, 0),
        'compass': (0, 255, 255),
        'scale': (255, 255, 255),
        'title': (255, 255, 0)
    }

    # 优先级
    priority = {
        'main map': 0,
        'legend': 1,
        'item': 2,
        'compass': 3,
        'scale': 4,
        'title': 5
    }

    # 空白透明图层
    combined_mask = np.zeros_like(vis_img, dtype=np.uint8)

    for result in results:
        if result.masks is None or result.boxes is None or len(result.boxes.cls) == 0:
            continue

        # 提取每个目标及其多边形
        targets = []
        for i in range(len(result.masks.xy)):
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]
            polygon = result.masks.xy[i]  # Nx2 polygon
            targets.append((priority.get(class_name, 6), i, class_id, class_name, polygon))

        # 排序
        targets.sort(key=lambda x: x[0])

        for _, i, class_id, class_name, polygon in targets:
            color = colors.get(class_name, np.random.randint(0, 255, 3).tolist())

            # 储存点坐标
            points = [(int(x), int(y)) for x, y in polygon]
            predictions.append({
                'class_id': class_id,
                'class_name': class_name,
                'points': points
            })

            # 生成掩码图层（填充区域）
            cv2.fillPoly(combined_mask, [polygon.astype(np.int32)], color)

            # 绘制轮廓边框
            cv2.polylines(vis_img, [polygon.astype(np.int32)], isClosed=True, color=color, thickness=2)

            # 添加文字
            x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
            cv2.putText(vis_img, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 所有掩码一次性透明叠加
    if return_vis:
        vis_img = cv2.addWeighted(vis_img, 0.9, combined_mask, 0.3, 0)

    return (predictions, vis_img) if return_vis else predictions


def segment_map(image: np.ndarray, model_path: str="../model/layout-bs256-gpu8-v0/best.pt", return_vis: bool = False):
    """
    通用API函数：输入图像和模型路径，返回分割结果（与 predict_image_segmentation 相同功能）。

    参数:
        image (np.ndarray): 输入图像
        model_path (str): YOLO模型路径
        return_vis (bool): 是否返回可视化图像

    返回:
        predictions (List[Dict]): 每个目标的预测信息
        vis_img (np.ndarray, optional): 可视化图像（若 return_vis=True）
    """
    return predict_image_segmentation(image, model_path=model_path, return_vis=return_vis)

def process_folders(input_root, model_path):
    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)
        if not os.path.isdir(folder_path) or not folder_name.isdigit():
            continue

        image_dir = os.path.join(folder_path, "image")
        layout_dir = os.path.join(folder_path, "layout")
        os.makedirs(layout_dir, exist_ok=True)

        if not os.path.exists(image_dir):
            print(f"No 'image' folder in {folder_path}, skipping.")
            continue

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))]
        if not image_files:
            print(f"No images found in {image_dir}, skipping.")
            continue

        for filename in image_files:
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Failed to read image {image_path}, skipping.")
                continue

            # 调用封装好的 API 函数
            #predictions, vis_img = predict_image_segmentation(img, model_path=model_path, return_vis=True)
            predictions, vis_img = predict_image_segmentation_v2(img, model_path=model_path, return_vis=True)

            # 保存可视化图像
            masked_path = os.path.join(layout_dir, f"masked_{filename}")
            cv2.imwrite(masked_path, vis_img)
            print(f"Saved image: {masked_path}")

            # 保存YOLO格式主轮廓标注
            txt_filename = f"{os.path.splitext(filename)[0]}.txt"
            txt_path = os.path.join(layout_dir, txt_filename)

            yolo_lines = []
            h_img, w_img = img.shape[:2]
            for pred in predictions:
                class_id = pred['class_id']
                norm_points = []
                for x, y in pred['points']:
                    x_norm = round(x / w_img, 6)
                    y_norm = round(y / h_img, 6)
                    norm_points.extend([x_norm, y_norm])
                line = f"{class_id} " + " ".join(map(str, norm_points))
                yolo_lines.append(line)

            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            print(f"Saved YOLO labels: {txt_path}")


def read_yolo_segmentation_file_to_pixel_coords(txt_path, image_width, image_height):
    """
    读取YOLO格式的分割文件，并将归一化坐标转换为像素坐标。

    参数:
        txt_path (str): YOLO格式的txt文件路径
        image_width (int): 原图的宽度
        image_height (int): 原图的高度

    返回:
        List[Dict]: 每个对象为字典，包含：
            - 'class_id': int
            - 'points': List of (x, y) tuples (像素坐标)
    """
    objects = []

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) % 2 != 0:
                raise ValueError(f"Invalid coordinates in line: {line}")

            points = []
            for i in range(0, len(coords), 2):
                x = int(round(coords[i] * image_width))
                y = int(round(coords[i + 1] * image_height))
                points.append((x, y))

            objects.append({
                'class_id': class_id,
                'points': points
            })

    return objects


def main():
    args = parse_args()
    #model = load_model()
    #model_path = "../model/layout_all-bs256-gpu8-v1/best.pt"
    model_path = "../model/layout_item-epoch10-bs256-gpu8-v0/best.pt"
    process_folders(args.input_root, model_path=model_path)

if __name__ == "__main__":
    main()
