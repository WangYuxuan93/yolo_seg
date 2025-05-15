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


def is_box_surrounded_by_uniform_color(image, box, initial_expand=1, border_thickness=1,
                                       color_tolerance=15, debug_dir=None, file_name=None, index=None):
    x1, y1, x2, y2, _ = box
    h, w = image.shape[:2]

    # 扩展 box
    x1_ext = max(x1 - initial_expand, 0)
    y1_ext = max(y1 - initial_expand, 0)
    x2_ext = min(x2 + initial_expand, w - 1)
    y2_ext = min(y2 + initial_expand, h - 1)

    # 外扩用于边缘提取
    x1_outer = max(x1_ext - border_thickness, 0)
    y1_outer = max(y1_ext - border_thickness, 0)
    x2_outer = min(x2_ext + border_thickness, w - 1)
    y2_outer = min(y2_ext + border_thickness, h - 1)

    # 四个边区域
    top = image[y1_outer:y1_ext, x1_outer:x2_outer]
    bottom = image[y2_ext:y2_outer, x1_outer:x2_outer]
    left = image[y1_outer:y2_outer, x1_outer:x1_ext]
    right = image[y1_outer:y2_outer, x2_ext:x2_outer]

    if any(arr.size == 0 for arr in [top, bottom, left, right]):
        return False

    # 拼图调试部分
    if debug_dir and index is not None:
        os.makedirs(debug_dir, exist_ok=True)

        label_list = ["TOP", "BOTTOM", "LEFT", "RIGHT", "ITEM"]
        img_list = [
            top,
            bottom,
            left,
            right,
            image[y1:y2, x1:x2]
        ]

        label_height = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        min_visual_width = 40

        labeled_blocks = []
        max_height = 0

        for label, img in zip(label_list, img_list):
            safe_img = img if img.size != 0 else np.full((30, 30, 3), 255, dtype=np.uint8)

            bordered_img = cv2.copyMakeBorder(
                safe_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(180, 180, 180)
            )

            h, w = bordered_img.shape[:2]

            # 宽度不足的（如LEFT/RIGHT），pad白边填充
            if w < min_visual_width:
                pad_total = min_visual_width - w
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                bordered_img = cv2.copyMakeBorder(
                    bordered_img, 0, 0, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
                w = min_visual_width

            # 生成标签区
            label_img = np.full((label_height, w, 3), 255, dtype=np.uint8)
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (label_height + text_size[1]) // 2
            cv2.putText(label_img, label, (text_x, text_y), font, font_scale,
                        (0, 0, 255), font_thickness, cv2.LINE_AA)

            combined = np.vstack([label_img, bordered_img])
            labeled_blocks.append(combined)
            max_height = max(max_height, combined.shape[0])

        # 统一高度
        resized_blocks = []
        for block in labeled_blocks:
            h, w = block.shape[:2]
            if h != max_height:
                resized_block = cv2.resize(block, (int(w * max_height / h), max_height))
            else:
                resized_block = block
            resized_blocks.append(resized_block)

        debug_image = np.hstack(resized_blocks)

    # 边缘像素颜色一致性判断
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)

    mean_color = np.mean(border_pixels, axis=0)
    color_diff = np.linalg.norm(border_pixels - mean_color, axis=1)
    mean_diff = np.mean(color_diff)
    result = mean_diff <= color_tolerance

    # 添加指标说明文字
    if debug_dir and index is not None:
        annotated = debug_image.copy()
        text_lines = [
            f"Mean RGB: ({int(mean_color[0])}, {int(mean_color[1])}, {int(mean_color[2])})",
            f"Mean Diff: {mean_diff:.2f}",
            f"Tolerance: {color_tolerance}",
            f"Pass: {'Yes' if result else 'No'}"
        ]
        for i, line in enumerate(text_lines):
            y = 15 + i * 18
            cv2.putText(annotated, line, (10, y), font, font_scale,
                        (0, 128, 0) if result else (0, 0, 255), font_thickness, cv2.LINE_AA)

        debug_path = os.path.join(debug_dir, f"{file_name}_box_{index}_debug.png")
        cv2.imwrite(debug_path, annotated)

    return result


def is_rectangle(approx, angle_tolerance=15):
    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    pts = approx.reshape(4, 2)
    vectors = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    angles = [angle_between(vectors[i], vectors[(i + 1) % 4]) for i in range(4)]

    return all(abs(a - 90) <= angle_tolerance for a in angles)


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

        if len(approx) == 4 and is_rectangle(approx):
        #if True:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            item_area = w * h
            #print (f"{legend_area}:{item_area}")
            #if (1 <= aspect_ratio <= 2.5) and (legend_area * area_min_factor <= item_area <= legend_area * area_max_factor):
            if (aspect_ratio > 1) and (legend_area * area_min_factor <= item_area <= legend_area * area_max_factor):
            #if (1 <= aspect_ratio <= 2.5):
                rectangles.append([x, y, x + w, y + h, 1.0])

    return rectangles


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def filter_isolated_boxes_by_clustering_auto_eps(boxes, min_samples=3, eps_scale=1.5):
    """
    自动估算 eps 参数的 box 聚类筛选方法。

    参数:
    - boxes: List[List[Tuple[int, int]]], 每个 box 为 4 个点坐标
    - min_samples: 聚类中最小邻居数
    - eps_scale: 基于平均第K邻居距离的放缩比例，默认扩大 20%

    返回:
    - filtered_boxes: 聚类保留下来的 box 列表
    - labels: 所有 box 对应的聚类标签（-1 表示被排除）
    """
    if len(boxes) == 0:
        return [], []

    centers = np.array([
        [np.mean([p[0] for p in box]), np.mean([p[1] for p in box])]
        for box in boxes
    ])

    if len(centers) <= min_samples:
        return boxes, [0] * len(boxes)

    # 计算每个点到其第 k 个邻居的距离
    k = min_samples
    nbrs = NearestNeighbors(n_neighbors=k).fit(centers)
    distances, _ = nbrs.kneighbors(centers)
    kth_distances = distances[:, -1]  # 取第 k 个邻居的距离
    #estimated_eps = np.median(kth_distances) * eps_scale
    #estimated_eps = np.mean(kth_distances) * eps_scale
    estimated_eps = np.percentile(kth_distances, 75) * eps_scale
    print (f"kth_distances: {kth_distances}\nestimated_eps: {estimated_eps}")

    # 聚类
    clustering = DBSCAN(eps=estimated_eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    print (f"labels:{labels}")

    valid_labels = {label for label in set(labels)
                    if label != -1 and list(labels).count(label) >= min_samples}

    filtered_boxes = [box for box, label in zip(boxes, labels) if label in valid_labels]

    return filtered_boxes, labels.tolist()



def process_image(image_path, output_image_path, output_txt_path, model_path,
                  legend_area_min_factor=0.001, legend_area_max_factor=0.1,
                  global_area_min_factor=0.0001, global_area_max_factor=0.01,
                  expand_pixel=30):
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

    rec_id = 0

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
            
            legend_area = cv2.contourArea(expanded_pts)
            #print (f"expanded_pts: {expanded_pts}\nlegend_area: {legend_area}")

            # --- 画扩展后的legend多边形（紫色） ---
            cv2.polylines(vis_img, [expanded_pts], isClosed=True, color=(255, 0, 255), thickness=3)

            expanded_mask = np.zeros_like(dilated_mask)
            cv2.fillPoly(expanded_mask, [expanded_pts], 255)

            legend_crop = cv2.bitwise_and(main_img, main_img, mask=expanded_mask)

            x, y, w, h = cv2.boundingRect(expanded_pts)
            legend_crop = legend_crop[y:y+h, x:x+w]
            #cropped_mask = expanded_mask[y:y+h, x:y+h]

            #legend_area = w * h

            legend_counter += 1  # 增加图例编号

            # 构建文件名并传递到函数中
            binary_image_filename = os.path.join(output_dir, f"{image_base_name}_{legend_counter}_binary.png")
            
            # 保存二值化图像
            rectangles = obtain_legend_rectangle_bbox(legend_crop, legend_area,
                                                        area_min_factor=legend_area_min_factor,
                                                        area_max_factor=legend_area_max_factor,
                                                        binary_image_filename=binary_image_filename
                                                    )

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
                rec_id += 1
                x1, y1, x2, y2, score = rect
                adjusted_rect = [x + x1, y + y1, x + x2, y + y2, score]
                if not is_box_surrounded_by_uniform_color(main_img, adjusted_rect, border_thickness=1, color_tolerance=50, debug_dir=output_dir, file_name=f"{image_base_name}_{legend_counter}", index=rec_id):
                    continue  # 跳过不合格的 box
                #x1, y1, x2, y2, _ = rect

                # 在overlay上填充矩形（绿色填充）
                #cv2.rectangle(overlay, (x + x1, y + y1), (x + x2, y + y2), (0, 255, 0), -1)

                # 在vis_img上画矩形边框
                #cv2.rectangle(vis_img, (x + x1, y + y1), (x + x2, y + y2), (0, 255, 0), 2)

                points = [
                    (x + x1, y + y1),
                    (x + x2, y + y1),
                    (x + x2, y + y2),
                    (x + x1, y + y2)
                ]
                all_boxes.append(points)

    if not found_legend or not all_boxes:
        if not found_legend:
            print(f"未检测到 legend，直接在全图进行矩形检测：{image_path}")
        else:
            print(f"已检测到 legend，但未在 legend 区域中找到有效的 item box，回退至整图检测：{image_path}")

        legend_crop = main_img
        legend_area = w_img * h_img

        rectangles = obtain_legend_rectangle_bbox(legend_crop, legend_area,
                                                    area_min_factor=global_area_min_factor,
                                                    area_max_factor=global_area_max_factor,
                                                    binary_image_filename=os.path.join(output_dir, f"{image_base_name}_legend_binary.png")
                                                    )

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
            rec_id += 1
            if not is_box_surrounded_by_uniform_color(main_img, rect, border_thickness=1, color_tolerance=15, debug_dir=output_dir, file_name=f"{image_base_name}_{legend_counter}", index=rec_id):
                continue  # 跳过不合格的 box
            x1, y1, x2, y2, _ = rect

            #cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            #cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            points = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ]
            all_boxes.append(points)

    # --- 融合overlay到vis_img，产生半透明填充效果 ---
    #filtered_boxes, labels = filter_isolated_boxes_by_clustering_auto_eps(all_boxes, eps_scale=2, min_samples=3)
    #all_boxes = filtered_boxes

    # ✅ Step 2：统一绘制 overlay 和 vis_img 中的框
    for box in all_boxes:
        # 每个 box 是四个点 [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        x1, y1 = box[0]
        x3, y3 = box[2]

        # 填充区域
        cv2.rectangle(overlay, (x1, y1), (x3, y3), (0, 255, 0), -1)
        # 边框
        cv2.rectangle(vis_img, (x1, y1), (x3, y3), (0, 255, 0), 2)


    alpha = 0.2  # 半透明程度
    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

    cv2.imwrite(output_image_path, vis_img)

    with open(output_txt_path, 'w') as f:
        for points in all_boxes:
            line = ",".join([f"{px},{py}" for (px, py) in points])
            f.write(line + "\n")

    print(f"处理完成，保存到 {output_image_path} 和 {output_txt_path}")



def main():
    parser = argparse.ArgumentParser(description="Batch processing for extracting map legend regions and detecting item boxes")
    parser.add_argument('--input_dir', type=str, help="Path to the input root directory containing numbered subfolders")
    parser.add_argument('--output_dir', type=str, help="Path to the output directory where results will be saved")
    parser.add_argument('--model_path', type=str, help="Path to the YOLO model used for semantic segmentation")
    parser.add_argument('--legend_area_min_factor', type=float, default=0.001,
                        help="Minimum ratio of item box area to legend area when a legend is detected (default: 0.001)")
    parser.add_argument('--legend_area_max_factor', type=float, default=0.1,
                        help="Maximum ratio of item box area to legend area when a legend is detected (default: 0.1)")
    parser.add_argument('--global_area_min_factor', type=float, default=0.001,
                        help="Minimum ratio of item box area to full image area when no legend is detected (default: 0.0001)")
    parser.add_argument('--global_area_max_factor', type=float, default=0.01,
                        help="Maximum ratio of item box area to full image area when no legend is detected (default: 0.01)")
    parser.add_argument('--expand_pixel', type=int, default=30,
                        help="Number of pixels to expand around the detected legend region for further processing (default: 30px)")


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
              legend_area_min_factor=args.legend_area_min_factor,
              legend_area_max_factor=args.legend_area_max_factor,
              global_area_min_factor=args.global_area_min_factor,
              global_area_max_factor=args.global_area_max_factor,
              expand_pixel=args.expand_pixel)


if __name__ == "__main__":
    main()
