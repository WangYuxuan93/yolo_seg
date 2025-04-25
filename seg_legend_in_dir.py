import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO


def predict_image_segmentation_with_opencv(image: np.ndarray, return_vis: bool = False):
    """
    使用 OpenCV 分割图像，先去除主图大框，再查找 legend/scale/title 等结构。

    参数:
        image (np.ndarray): 输入图像（BGR）
        return_vis (bool): 是否返回可视化图像

    返回:
        predictions (List[Dict]): 每个目标包含：
            - class_id: 0
            - class_name: "region"
            - points: List[(x, y)] 多边形坐标
        vis_img (optional): 可视化图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = image.shape[:2]
    total_area = h_img * w_img

    # 步骤1：初步二值化
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 步骤2：找最大轮廓（主图）并抹去
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(binary, [main_contour], -1, 0, thickness=cv2.FILLED)  # 黑色填充主图

    # 步骤3：查找去除主图后的其余框（图例等）
    binary_mask = binary.copy()
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    predictions = []
    vis_img = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / total_area

        # 过滤规则（根据相对面积）
        #if area_ratio < 0.005 or area_ratio > 0.4:
        #    continue

        # 简化轮廓点
        approx = cv2.approxPolyDP(cnt, epsilon=3.0, closed=True)
        points = [(pt[0][0], pt[0][1]) for pt in approx]

        predictions.append({
            'class_id': 0,
            'class_name': 'region',
            'points': points
        })

        if return_vis:
            color = (0, 255, 0)
            cv2.drawContours(vis_img, [approx], -1, color, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(vis_img, 'region', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if return_vis:
        return predictions, vis_img
    return predictions


def refine_box_with_opencv(cropped_region: np.ndarray, return_polygon: bool = False) -> list:
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize=15, C=10)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    main_contour = max(contours, key=cv2.contourArea)

    if return_polygon:
        approx = cv2.approxPolyDP(main_contour, epsilon=2.0, closed=True)
        return [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
    else:
        x, y, w, h = cv2.boundingRect(main_contour)
        return [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]


def refine_box_with_opencv_v2(cropped_region: np.ndarray, return_polygon: bool = False) -> list:
    """
    对区域进行轮廓提取，要求最终结果必须是四边形，否则返回空。
    使用最大面积轮廓，过滤掉非矩形。
    """
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize=15, C=10)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # 找最大面积轮廓
    best_cnt = max(contours, key=cv2.contourArea)

    # 判断是否近似为四边形
    approx = cv2.approxPolyDP(best_cnt, epsilon=2.0, closed=True)
    #if len(approx) != 4:
    #    return []  # ❌ 非矩形

    if return_polygon:
        return [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
    else:
        x, y, w, h = cv2.boundingRect(approx)
        return [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]

def visualize_boxes(image, x, y, w, h, refined, refined2=None, expand_bbox_px=0, output_dir="output"):
    """
    可视化：绘制原始框，第一次尝试扩张框，第二次尝试框（如果有）。
    将可视化的图像保存到文件夹。
    """
    # 计算扩展后的框的坐标，第一次尝试
    if expand_bbox_px > 0:
        x1 = max(0, x - expand_bbox_px)
        y1 = max(0, y - expand_bbox_px)
        x2 = min(image.shape[1] - 1, x + w + expand_bbox_px)
        y2 = min(image.shape[0] - 1, y + h + expand_bbox_px)
    else:
        # 第二次尝试时，不扩展，使用原始框的坐标
        x1, y1, x2, y2 = x, y, x + w, y + h

    # 创建一个与原图大小相同的透明图层
    vis_img = image.copy()
    overlay = np.zeros_like(vis_img, dtype=np.uint8)

    # 绘制原始框（红色，透明）
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框

    # 绘制第一次尝试的扩张框（蓝色，透明）
    if refined:
        # 加上偏移量（x1, y1）将 refined 坐标放回原图位置
        x_ref, y_ref, w_ref, h_ref = cv2.boundingRect(np.array(refined))
        x_ref += x1  # 加上偏移量 x1
        y_ref += y1  # 加上偏移量 y1
        cv2.rectangle(overlay, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), (255, 0, 0), 2)  # 蓝色框

    # 绘制第二次尝试的框（绿色，透明）
    if refined2:
        # 获取最小矩形框并加上偏移量
        x_ref2, y_ref2, w_ref2, h_ref2 = cv2.boundingRect(np.array(refined2))
        x_ref2 += x  # 对于第二次尝试，直接用原始的 x, y 作为偏移量
        y_ref2 += y  # 对于第二次尝试，直接用原始的 y, y 作为偏移量
        cv2.rectangle(overlay, (x_ref2, y_ref2), (x_ref2 + w_ref2, y_ref2 + h_ref2), (0, 255, 0), 2)  # 绿色框

    # 叠加透明效果：cv2.addWeighted() 用于将 overlay 叠加到原图上
    vis_img = cv2.addWeighted(vis_img, 0.6, overlay, 0.8, 0)  # 0.8 为透明度，值越小越透明

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    output_file = os.path.join(output_dir, f"visualized_{x}_{y}_{w}_{h}.jpg")
    cv2.imwrite(output_file, vis_img)
    print(f"Saved visualized image to {output_file}")




def predict_image_segmentation(image: np.ndarray, model_path: str,
                               return_vis: bool = False,
                               debug=False, smooth_contours: bool = True, epsilon_factor: float = 0.02):
    model = YOLO(model_path)
    results = model(image)
    predictions = []
    vis_img = image.copy()

    colors = {
        'main map': (0, 255, 0),
        'legend': (255, 0, 0),
        'item': (0, 0, 255),
        'compass': (0, 255, 255),
        'scale': (255, 255, 255),
        'title': (255, 255, 0)
    }

    priority = {
        'main map': 0,
        'legend': 1,
        'item': 2,
        'compass': 3,
        'scale': 4,
        'title': 5
    }

    h_img, w_img = image.shape[:2]
    combined_mask = np.zeros_like(vis_img, dtype=np.uint8)

    for result in results:
        if result.masks is None or result.boxes is None or len(result.boxes.cls) == 0:
            continue

        targets = []
        for i in range(len(result.masks.xy)):
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]
            polygon = result.masks.xy[i]
            targets.append((priority.get(class_name, 6), i, class_id, class_name, polygon))

        targets.sort(key=lambda x: x[0])

        for _, i, class_id, class_name, polygon in targets:
            color = colors.get(class_name, tuple(np.random.randint(0, 255, 3).tolist()))

            if smooth_contours:
                # --- 凸包 + 平滑 ---
                points_array = polygon.astype(np.int32)
                hull = cv2.convexHull(points_array)
                arc_length = cv2.arcLength(hull, closed=True)
                epsilon = epsilon_factor * arc_length
                approx = cv2.approxPolyDP(hull, epsilon=epsilon, closed=True)
                points = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
                polygon_draw = approx
            else:
                # --- 不处理顺序、直接绘制原始 polygon ---
                points = [(int(x), int(y)) for x, y in polygon]
                polygon_draw = np.array(points, dtype=np.int32)

            predictions.append({
                'class_id': class_id,
                'class_name': class_name,
                'points': points
            })

            cv2.fillPoly(combined_mask, [polygon_draw], color)
            cv2.polylines(vis_img, [polygon_draw], isClosed=True, color=color, thickness=2)
            x_text, y_text, _, _ = cv2.boundingRect(polygon_draw)
            cv2.putText(vis_img, class_name, (x_text, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

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

def process_folders(input_root, model_path, use_bbox=True):
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
            predictions, vis_img = predict_image_segmentation(img, model_path=model_path, return_vis=True, use_bbox=use_bbox)

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


def process_folders_to_ouput_dir(input_root, model_path, output_dir, use_opencv=False, smooth_contours=True, epsilon_factor=0.02):
    os.makedirs(output_dir, exist_ok=True)

    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)
        if not os.path.isdir(folder_path) or not folder_name.isdigit():
            continue

        image_dir = os.path.join(folder_path, "image")
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

            # 选择 YOLO 或 OpenCV 分割
            if use_opencv:
                predictions, vis_img = predict_image_segmentation_with_opencv(img, return_vis=True)
            else:
                predictions, vis_img = predict_image_segmentation(
                    img,
                    model_path=model_path,
                    return_vis=True,
                    smooth_contours=smooth_contours, 
                    epsilon_factor=epsilon_factor
                )

            prefix = f"{folder_name}_{os.path.splitext(filename)[0]}"
            masked_path = os.path.join(output_dir, f"masked_{prefix}.jpg")
            cv2.imwrite(masked_path, vis_img)
            print(f"Saved image: {masked_path}")

            txt_path = os.path.join(output_dir, f"{prefix}.txt")
            h_img, w_img = img.shape[:2]
            yolo_lines = []
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
    parser = argparse.ArgumentParser(description="Segment map images using YOLO or OpenCV.")
    parser.add_argument('--input_root', type=str, required=True, help="Root directory containing numbered subfolders with images")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for saving masks and labels")
    parser.add_argument('--model_path', type=str, default="../model/layout-bs256-gpu8-v0/best.pt", help="Path to YOLO model")
    parser.add_argument('--use_opencv', action='store_true', help="Use OpenCV instead of YOLO for segmentation")
    parser.add_argument('--smooth_contours', action='store_true', help="smooth contours")
    parser.add_argument('--epsilon_factor', type=float, default=0.02, help="epsilon factor for smooth contours")

    args = parser.parse_args()

    process_folders_to_ouput_dir(
        input_root=args.input_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_opencv=args.use_opencv,
        smooth_contours=args.smooth_contours, 
        epsilon_factor=args.epsilon_factor
    )
if __name__ == "__main__":
    main()
