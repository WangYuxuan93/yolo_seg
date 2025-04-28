import cv2
import numpy as np
import os
import glob
import argparse
from ultralytics import YOLO

def predict_image_segmentation(image: np.ndarray, model_path: str, 
                                return_vis: bool = False,
                                debug: bool = False,
                                smooth_contours: bool = False,
                                epsilon_factor: float = 0.02,
                                merge_same_class_boxes: bool = False):
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

    class_polygons = {}

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
            if class_name not in class_polygons:
                class_polygons[class_name] = []
            class_polygons[class_name].append((class_id, polygon))

    merged_targets = []

    for class_name, instances in class_polygons.items():
        if merge_same_class_boxes and len(instances) > 1:
            class_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            for _, polygon in instances:
                pts = polygon.astype(np.int32)
                cv2.fillPoly(class_mask, [pts], 255)

            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.shape[1] == 1:
                    contour = contour.squeeze(1)
                merged_targets.append((instances[0][0], class_name, contour))
        else:
            for class_id, polygon in instances:
                pts = polygon.astype(np.int32)
                merged_targets.append((class_id, class_name, pts))

    for class_id, class_name, polygon in merged_targets:
        color = colors.get(class_name, tuple(np.random.randint(0, 255, 3).tolist()))

        if smooth_contours:
            hull = cv2.convexHull(polygon)
            arc_length = cv2.arcLength(hull, closed=True)
            epsilon = epsilon_factor * arc_length
            approx = cv2.approxPolyDP(hull, epsilon=epsilon, closed=True)
            points = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
            polygon_draw = approx
        else:
            points = [(int(x), int(y)) for x, y in polygon]
            polygon_draw = polygon.reshape(-1, 1, 2)

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


def obtain_legend_rectangle_bbox(main_img, legend_area, area_min_factor=0.01, area_max_factor=0.5, binary_image_filename=None):
    target_img = np.array(main_img)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(target_img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 5, 1)

    # 保存二值化图像到传入的路径
    if binary_image_filename is not None:
        cv2.imwrite(binary_image_filename, thresh)  # 保存二值化图像
        print(f"二值化图已保存为 {binary_image_filename}")

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
        #if True:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            item_area = w * h
            print (f"{legend_area}:{item_area}")
            if (1 <= aspect_ratio <= 2.5) and (legend_area * area_min_factor <= item_area <= legend_area * area_max_factor):
            #if (1 <= aspect_ratio <= 2.5):
                rectangles.append([x, y, x + w, y + h, 1.0])

    return rectangles


def process_image(image_path, output_image_path, output_txt_path, model_path, area_min_factor=0.001, area_max_factor=0.5, expand_pixel=30):
    # 获取 output_dir
    output_dir = os.path.dirname(output_image_path)
    
    # 从文件名中提取编号部分（例如，假设图像名是 "3.png"，我们提取 "3"）
    image_name = os.path.basename(output_image_path)
    image_base_name, _ = os.path.splitext(image_name)  # 获取文件名部分，不带扩展名

    main_img = cv2.imread(image_path)
    if main_img is None:
        raise FileNotFoundError(f"无法读取图像文件：{image_path}")

    h_img, w_img = main_img.shape[:2]
    vis_img = main_img.copy()
    overlay = vis_img.copy()  # 用来画半透明区域

    predictions = predict_image_segmentation(main_img, model_path=model_path, smooth_contours=False,
                                              epsilon_factor=0.002,
                                              merge_same_class_boxes=False)

    all_boxes = []
    found_legend = False
    legend_counter = 0  # 用于编号

    for pred in predictions:
        if pred['class_name'] == 'legend':
            found_legend = True

            pts = np.array(pred['points'], dtype=np.int32)

            # --- 画原始legend polygon（蓝色） ---
            cv2.polylines(vis_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # --- 创建mask并膨胀 ---
            mask = np.zeros(main_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

            kernel = np.ones((expand_pixel * 2 + 1, expand_pixel * 2 + 1), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                raise ValueError("膨胀后的mask没有找到任何轮廓")

            expanded_pts = max(contours, key=cv2.contourArea)

            # --- 画扩展后的legend多边形（紫色） ---
            cv2.polylines(vis_img, [expanded_pts], isClosed=True, color=(255, 0, 255), thickness=3)

            expanded_mask = np.zeros_like(dilated_mask)
            cv2.fillPoly(expanded_mask, [expanded_pts], 255)

            legend_crop = cv2.bitwise_and(main_img, main_img, mask=expanded_mask)

            x, y, w, h = cv2.boundingRect(expanded_pts)
            legend_crop = legend_crop[y:y+h, x:x+w]
            cropped_mask = expanded_mask[y:y+h, x:y+h]

            legend_area = cv2.countNonZero(cropped_mask)

            legend_counter += 1  # 增加图例编号

            # 构建文件名并传递到函数中
            binary_image_filename = os.path.join(output_dir, f"{image_base_name}_{legend_counter}_binary.png")
            
            # 保存二值化图像
            rectangles = obtain_legend_rectangle_bbox(legend_crop, legend_area, area_min_factor=area_min_factor, area_max_factor=area_max_factor, binary_image_filename=binary_image_filename)

            def box_area(rect):
                return (rect[2] - rect[0]) * (rect[3] - rect[1])

            def boxes_overlap(a, b):
                return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

            rectangles.sort(key=box_area, reverse=True)

            selected = []
            for rect in rectangles:
                overlap = False
                for kept in selected:
                    if boxes_overlap(rect, kept):
                        overlap = True
                        break
                if not overlap:
                    selected.append(rect)

            for rect in selected:
                x1, y1, x2, y2, _ = rect

                # 在overlay上填充矩形（绿色填充）
                cv2.rectangle(overlay, (x + x1, y + y1), (x + x2, y + y2), (0, 255, 0), -1)

                # 在vis_img上画矩形边框
                cv2.rectangle(vis_img, (x + x1, y + y1), (x + x2, y + y2), (0, 255, 0), 2)

                points = [
                    (x + x1, y + y1),
                    (x + x2, y + y1),
                    (x + x2, y + y2),
                    (x + x1, y + y2)
                ]
                all_boxes.append(points)

    if not found_legend:
        print(f"未检测到legend，直接在全图进行矩形检测：{image_path}")

        legend_crop = main_img
        legend_area = w_img * h_img

        rectangles = obtain_legend_rectangle_bbox(legend_crop, legend_area, area_min_factor=area_min_factor, area_max_factor=area_max_factor, binary_image_filename=os.path.join(output_dir, f"{image_base_name}_legend_binary.png"))

        def box_area(rect):
            return (rect[2] - rect[0]) * (rect[3] - rect[1])

        def boxes_overlap(a, b):
            return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

        rectangles.sort(key=box_area, reverse=True)

        selected = []
        for rect in rectangles:
            overlap = False
            for kept in selected:
                if boxes_overlap(rect, kept):
                    overlap = True
                    break
            if not overlap:
                selected.append(rect)

        for rect in selected:
            x1, y1, x2, y2, _ = rect

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            points = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ]
            all_boxes.append(points)

    # --- 融合overlay到vis_img，产生半透明填充效果 ---
    alpha = 0.2  # 半透明程度
    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

    cv2.imwrite(output_image_path, vis_img)

    with open(output_txt_path, 'w') as f:
        for points in all_boxes:
            line = ",".join([f"{px},{py}" for (px, py) in points])
            f.write(line + "\n")

    print(f"处理完成，保存到 {output_image_path} 和 {output_txt_path}")



def main():
    parser = argparse.ArgumentParser(description="批量处理地图图例区域提取")
    parser.add_argument('--input_dir', type=str, help="输入文件夹路径")
    parser.add_argument('--output_dir', type=str, help="输出文件夹路径")
    parser.add_argument('--model_path', type=str, help="YOLO模型路径")
    parser.add_argument('--area_min_factor', type=float, default=0.001, help="小矩形面积占legend面积的最小比例，默认0.1%")
    parser.add_argument('--area_max_factor', type=float, default=0.5, help="小矩形面积占legend面积的最比例，默认50%")
    parser.add_argument('--expand_pixel', type=int, default=30, help="legend裁剪扩展固定像素，默认30px")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    subdirs = [d for d in subdirs if d.isdigit()]

    for subdir in subdirs:
        image_folder = os.path.join(args.input_dir, subdir, 'image')
        tif_files = glob.glob(os.path.join(image_folder, '*.tif'))
        if not tif_files:
            print(f"跳过 {subdir}，没有.tif文件")
            continue
        image_path = tif_files[0]

        output_image_path = os.path.join(args.output_dir, f"{subdir}.png")
        output_txt_path = os.path.join(args.output_dir, f"{subdir}.txt")

        process_image(image_path, output_image_path, output_txt_path, args.model_path,
                      area_min_factor=args.area_min_factor, area_max_factor=args.area_max_factor, expand_pixel=args.expand_pixel)


if __name__ == "__main__":
    main()
