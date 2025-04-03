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

def predict_image_segmentation(image: np.ndarray, model: YOLO, return_vis: bool = False):
    """
    对输入图像进行分割预测，返回分割轮廓坐标和可视化图像（可选）

    参数：
        image (np.ndarray): 输入图像（BGR格式）
        model (YOLO): 已加载的YOLO模型对象
        return_vis (bool): 是否返回可视化图像

    返回：
        List[Dict]: 每个目标的预测信息：
            - class_id (int)
            - class_name (str)
            - points: List of (x, y) tuples，轮廓像素坐标点
        vis_img (np.ndarray, optional): 可视化图像（若 return_vis=True）
    """
    h_img, w_img = image.shape[:2]
    results = model(image)

    predictions = []
    vis_img = image.copy()

    colors = {
        'main map': [255, 0, 0],
        'legend': [0, 0, 255],
    }

    for result in results:
        for i, mask in enumerate(result.masks.data):
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]
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

            if return_vis:
                # 可视化
                cv2.drawContours(vis_img, [main_contour], -1, color, 2)
                x, y, w, h = cv2.boundingRect(main_contour)
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis_img, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if return_vis:
        return predictions, vis_img
    return predictions

"""
def process_folders(input_root, model):
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
            results = model(image_path)
            img = cv2.imread(image_path)
            h_img, w_img = img.shape[:2]

            colors = {
                'main map': [255, 0, 0],
                'legend': [0, 0, 255],
            }

            for result in results:
                yolo_lines = []

                for i, mask in enumerate(result.masks.data):
                    class_id = int(result.boxes.cls[i])
                    class_name = result.names[class_id]
                    color = colors.get(class_name, np.random.randint(0, 255, 3).tolist())
                    mask_resized = cv2.resize(mask.cpu().numpy(), (w_img, h_img))
                    mask_resized = np.uint8(mask_resized * 255)

                    # 可视化叠加
                    mask_overlay = np.zeros_like(img, dtype=np.uint8)
                    mask_overlay[mask_resized > 0] = color
                    img = cv2.addWeighted(img, 0.85, mask_overlay, 0.15, 0)

                    # 提取最大轮廓
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        if len(main_contour) >= 3:
                            normalized_points = []
                            for pt in main_contour.squeeze():
                                x, y = pt[0], pt[1]
                                x_norm = round(x / w_img, 6)
                                y_norm = round(y / h_img, 6)
                                normalized_points.extend([x_norm, y_norm])
                            line = f"{class_id} " + " ".join(map(str, normalized_points))
                            yolo_lines.append(line)

                        # 可视化轮廓
                        cv2.drawContours(img, [main_contour], -1, color, 2)

                    # 类别文本显示
                    x_min, y_min, x_max, y_max = result.boxes.xyxy[i].cpu().numpy()
                    (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_y = max(int(y_min) + h + 5, h + 5)
                    cv2.rectangle(img, (int(x_min), text_y - h), (int(x_min) + w, text_y), color, -1)
                    cv2.putText(img, class_name, (int(x_min), text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # 保存可视化图像
                masked_path = os.path.join(layout_dir, f"masked_{filename}")
                cv2.imwrite(masked_path, img)
                print(f"Saved image: {masked_path}")

                # 保存YOLO格式主轮廓标注
                txt_filename = f"{os.path.splitext(filename)[0]}.txt"
                txt_path = os.path.join(layout_dir, txt_filename)
                with open(txt_path, 'w') as f:
                    f.write("\n".join(yolo_lines))
                print(f"Saved YOLO labels: {txt_path}")
"""

def process_folders(input_root, model):
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
            predictions, vis_img = predict_image_segmentation(img, model, return_vis=True)

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
    model = load_model()
    process_folders(args.input_root, model)

if __name__ == "__main__":
    main()