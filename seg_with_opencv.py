import cv2
import numpy as np
import os
import glob
import argparse
from ultralytics import YOLO
from collections import defaultdict

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
                                       color_tolerance=15, default_bg_color=None, 
                                       debug_dir=None, file_name=None, index=None):
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

    # 提取边缘所有像素
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)

    # 判断颜色一致性（使用默认颜色或平均值）
    if default_bg_color is not None:
        ref_color = np.array(default_bg_color)
        color_diff = np.linalg.norm(border_pixels - ref_color, axis=1)
        mean_diff = np.mean(color_diff)
    else:
        ref_color = np.mean(border_pixels, axis=0)
        color_diff = np.linalg.norm(border_pixels - ref_color, axis=1)
        mean_diff = np.mean(color_diff)

    result = mean_diff <= color_tolerance

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

            if w < min_visual_width:
                pad_total = min_visual_width - w
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                bordered_img = cv2.copyMakeBorder(
                    bordered_img, 0, 0, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
                w = min_visual_width

            label_img = np.full((label_height, w, 3), 255, dtype=np.uint8)
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (label_height + text_size[1]) // 2
            cv2.putText(label_img, label, (text_x, text_y), font, font_scale,
                        (0, 0, 255), font_thickness, cv2.LINE_AA)

            combined = np.vstack([label_img, bordered_img])
            labeled_blocks.append(combined)
            max_height = max(max_height, combined.shape[0])

        resized_blocks = []
        for block in labeled_blocks:
            h, w = block.shape[:2]
            if h != max_height:
                resized_block = cv2.resize(block, (int(w * max_height / h), max_height))
            else:
                resized_block = block
            resized_blocks.append(resized_block)

        debug_image = np.hstack(resized_blocks)

        annotated = debug_image.copy()
        ref_r, ref_g, ref_b = [int(x) for x in ref_color]
        text_lines = [
            f"Reference RGB: ({ref_r}, {ref_g}, {ref_b})",
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


def filter_boxes_by_uniform_color(boxes, image, offset_xy=(0, 0),
                                   initial_expand=1,
                                   border_thickness=1,
                                   default_bg_color=None,
                                   color_tolerance=15,
                                   debug=False,
                                   debug_dir=None,
                                   file_name=None,
                                   legend_counter=None,
                                   start_index=0):
    """
    对 box 列表进行颜色一致性判断，过滤掉边缘颜色不符合的 box。

    参数：
    - boxes: List[List[int]]，每个为 [x1, y1, x2, y2, score]
    - image: 原图
    - offset_xy: (x_offset, y_offset)，box 偏移量（如 legend 裁剪偏移）
    - initial_expand, border_thickness, default_bg_color, color_tolerance: 颜色判断参数
    - debug: 是否保存 debug 图
    - debug_dir, file_name, legend_counter: 用于保存调试图命名
    - start_index: box 序号起始值（用于 debug 命名）

    返回：
    - kept_boxes: List[List[Tuple[int, int]]]，通过过滤的 box（四点格式）
    - filtered_out_boxes: List[List[Tuple[int, int]]]，被过滤掉的 box（四点格式）
    """
    print ("\n[Color] Filtering boxes by surrounded color")

    x_offset, y_offset = offset_xy
    kept_boxes = []
    filtered_out_boxes = []

    for i, rect in enumerate(boxes):
        rec_id = start_index + i + 1
        x1, y1, x2, y2, score = rect
        adjusted_rect = [x_offset + x1, y_offset + y1, x_offset + x2, y_offset + y2, score]

        points = [
            (x_offset + x1, y_offset + y1),
            (x_offset + x2, y_offset + y1),
            (x_offset + x2, y_offset + y2),
            (x_offset + x1, y_offset + y2)
        ]

        passed = is_box_surrounded_by_uniform_color(
            image, adjusted_rect,
            initial_expand=initial_expand,
            border_thickness=border_thickness,
            default_bg_color=default_bg_color,
            color_tolerance=color_tolerance,
            debug_dir=debug_dir if debug else None,
            file_name=f"{file_name}_{legend_counter}" if legend_counter is not None else file_name,
            index=rec_id
        )

        if passed:
            kept_boxes.append(points)
        else:
            filtered_out_boxes.append(points)

    print ("Background Color: {}".format("Mean" if default_bg_color is None else default_bg_color))
    print ("Input: {}, Output: {}".format(len(boxes), len(kept_boxes)))

    return kept_boxes, filtered_out_boxes


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
    print(f"\n[Search] Searching candidate item boxes")
    target_img = np.array(main_img)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(target_img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 5, 1)

    # 保存二值化图像到传入的路径
    if binary_image_filename is not None:
        cv2.imwrite(binary_image_filename, thresh)  # 保存二值化图像
        print(f"Binary image saved to: {binary_image_filename}")

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
    print ("Found {} candidates".format(len(rectangles)))

    return rectangles



def recover_deleted_boxes_by_size_consistency(all_boxes, labels, outlier_boxes, size_tolerance=0.1):
    """
    Recover outlier boxes whose size (width and height) closely match the standard size
    derived from each DBSCAN cluster.

    Parameters:
    - all_boxes: List of all boxes (each box is a list of 4 (x, y) points)
    - labels: List of DBSCAN cluster labels for each box (-1 means outlier)
    - outlier_boxes: List of boxes labeled as outliers
    - size_tolerance: Allowed relative difference for matching width/height (e.g., 0.1 = 10%)

    Returns:
    - recovered_boxes: List of recovered boxes from outliers
    """

    # Step 1: Measure sizes (width, height) of boxes in each cluster
    cluster_sizes = defaultdict(list)
    for box, label in zip(all_boxes, labels):
        if label == -1:
            continue
        w = np.linalg.norm(np.array(box[1]) - np.array(box[0]))
        h = np.linalg.norm(np.array(box[3]) - np.array(box[0]))
        cluster_sizes[label].append((w, h))

    # Step 2: Compute standard size for each cluster using median
    cluster_standards = {}
    print("\nStandard width and height for each cluster:")
    for label, sizes in cluster_sizes.items():
        widths = [wh[0] for wh in sizes]
        heights = [wh[1] for wh in sizes]
        std_w = np.median(widths)
        std_h = np.median(heights)
        cluster_standards[label] = (std_w, std_h)
        print(f" - Cluster {label}: width = {std_w:.1f}, height = {std_h:.1f} ({len(sizes)} boxes)")

    # Step 3: Try to recover outliers whose size matches any cluster
    recovered_boxes = []
    print("\nChecking outliers for possible recovery:")
    for box in outlier_boxes:
        w = np.linalg.norm(np.array(box[1]) - np.array(box[0]))
        h = np.linalg.norm(np.array(box[3]) - np.array(box[0]))
        for label, (std_w, std_h) in cluster_standards.items():
            if (
                abs(w - std_w) / std_w <= size_tolerance and
                abs(h - std_h) / std_h <= size_tolerance
            ):
                recovered_boxes.append(box)
                print(f"Recovered box: width = {w:.1f}, height = {h:.1f} -> matches cluster {label}")
                break

    if not recovered_boxes:
        print("No recoverable boxes found.")

    return recovered_boxes


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def filter_isolated_boxes_by_clustering_auto_eps(boxes, min_samples=3, eps_scale=2.0):
    """
    自动估算 eps 的 DBSCAN 聚类过滤方法。

    参数：
    - boxes: List[List[Tuple[int, int]]]，每个 box 是 4 个点
    - min_samples: 最小邻居数（DBSCAN）
    - eps_scale: 第k邻居距离乘以的比例，用于估算 eps

    返回：
    - clustered_boxes: 聚类保留下来的 box 列表
    - labels: 每个 box 对应的聚类标签（-1 表示离群）
    - outlier_boxes: 被 DBSCAN 标记为离群的 box
    - cluster_standards: Dict[label] = (width, height)，每个聚类的标准尺寸
    """
    print("\n[Cluster] Clustring boxes for distance filtering:")
    if len(boxes) == 0:
        return [], [], [], {}

    centers = np.array([
        [np.mean([p[0] for p in box]), np.mean([p[1] for p in box])]
        for box in boxes
    ])

    if len(centers) <= min_samples:
        return boxes, [0] * len(boxes), [], {}

    # 计算第 k 个邻居距离
    k = min_samples
    nbrs = NearestNeighbors(n_neighbors=k).fit(centers)
    distances, _ = nbrs.kneighbors(centers)
    kth_distances = distances[:, -1]
    estimated_eps = np.percentile(kth_distances, 75) * eps_scale
    print(f"kth_distances: {kth_distances}\nestimated_eps: {estimated_eps}")

    # 聚类
    clustering = DBSCAN(eps=estimated_eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    print(f"labels: {labels}")

    # 提取聚类框和离群框
    clustered_boxes = [box for box, label in zip(boxes, labels) if label != -1]
    outlier_boxes = [box for box, label in zip(boxes, labels) if label == -1]

    # 计算每个聚类的标准宽高
    cluster_sizes = defaultdict(list)
    for box, label in zip(boxes, labels):
        if label == -1:
            continue
        w = np.linalg.norm(np.array(box[1]) - np.array(box[0]))
        h = np.linalg.norm(np.array(box[3]) - np.array(box[0]))
        cluster_sizes[label].append((w, h))

    cluster_standards = {}
    print("Standard width and height for each cluster:")
    for label, sizes in cluster_sizes.items():
        widths = [wh[0] for wh in sizes]
        heights = [wh[1] for wh in sizes]
        std_w = np.median(widths)
        std_h = np.median(heights)
        cluster_standards[label] = (std_w, std_h)
        print(f" - Cluster {label}: width = {std_w:.1f}, height = {std_h:.1f} ({len(sizes)} boxes)")

    return clustered_boxes, labels.tolist(), outlier_boxes, cluster_standards


def recover_boxes_by_size_match(outlier_boxes, cluster_standards, size_tolerance=0.1):
    """
    根据聚类标准长宽，从离群项中恢复尺寸匹配的 box。

    参数：
    - outlier_boxes: List[List[Tuple[int, int]]]，被过滤掉的 box
    - cluster_standards: Dict[label] = (width, height)，聚类标准尺寸
    - size_tolerance: float，长宽的允许误差比例（如 0.1 表示±10%）

    返回：
    - recovered_boxes: List[List[Tuple[int, int]]]，恢复的 box
    """
    recovered_boxes = []
    print("\n[Recover] Checking outliers for size-based recovery:")
    for box in outlier_boxes:
        w = np.linalg.norm(np.array(box[1]) - np.array(box[0]))
        h = np.linalg.norm(np.array(box[3]) - np.array(box[0]))
        #print (f"w,h:({w},{h})")
        for label, (std_w, std_h) in cluster_standards.items():
            if (
                abs(w - std_w) / std_w <= size_tolerance and
                abs(h - std_h) / std_h <= size_tolerance
            ):
                recovered_boxes.append(box)
                print(f"Recovered box: width = {w:.1f}, height = {h:.1f} -> matches cluster {label}")
                break

    if not recovered_boxes:
        print("No recoverable boxes found.")

    return recovered_boxes


def remove_overlapping_rect_simple(rectangles):
    print ("\n[Overlap] Removing overlapping boxes")
    def box_area(rect):
        return (rect[2] - rect[0]) * (rect[3] - rect[1])

    def boxes_overlap(a, b):
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    rectangles = sorted(rectangles, key=box_area, reverse=True)

    selected = []
    for rect in rectangles:
        if not any(boxes_overlap(rect, kept) for kept in selected):
            selected.append(rect)
    print ("Input: {}, Output: {}".format(len(rectangles), len(selected)))

    return selected

def remove_overlapping_boxes_simple(rectangles):
    def box_area(rect):
        x1, y1 = rect[0]
        x2, y2 = rect[2]
        return (x2 - x1) * (y2 - y1)

    def boxes_overlap(a, b):
        ax1, ay1 = a[0]
        ax2, ay2 = a[2]
        bx1, by1 = b[0]
        bx2, by2 = b[2]
        return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)

    rectangles = sorted(rectangles, key=box_area, reverse=True)

    selected = []
    for rect in rectangles:
        if not any(boxes_overlap(rect, kept) for kept in selected):
            selected.append(rect)

    return selected


def refine_boxes_by_size_consistency(boxes, cluster_standards, size_tolerance=0.1):
    """
    进一步过滤通过初步聚类保留下来的 box，仅保留尺寸接近标准的框。

    参数：
    - boxes: List[List[Tuple[int, int]]]，原始保留的 box
    - cluster_standards: Dict[label] = (width, height)，每个聚类的标准长宽
    - size_tolerance: float，相对误差容忍度（如 0.1 表示±10%）

    返回：
    - refined_boxes: List[List[Tuple[int, int]]]，尺寸合法的 box
    """
    refined_boxes = []
    print("\n[Refine] Re-checking kept boxes for size consistency:")

    for box in boxes:
        w = np.linalg.norm(np.array(box[1]) - np.array(box[0]))
        h = np.linalg.norm(np.array(box[3]) - np.array(box[0]))

        passed = False
        for label, (std_w, std_h) in cluster_standards.items():
            if (
                abs(w - std_w) / std_w <= size_tolerance and
                abs(h - std_h) / std_h <= size_tolerance
            ):
                passed = True
                print(f"Box OK: width = {w:.1f}, height = {h:.1f} matches cluster {label}")
                break

        if passed:
            refined_boxes.append(box)
        else:
            print(f"Box filtered out: width = {w:.1f}, height = {h:.1f} (no match)")

    print(f"Retained {len(refined_boxes)} of {len(boxes)} boxes after size check.")
    return refined_boxes


def process_image(image_path, output_image_path, output_txt_path, model_path,
                  legend_area_min_factor=0.001, legend_area_max_factor=0.1,
                  global_area_min_factor=0.0001, global_area_max_factor=0.01,
                  expand_pixel=30, cluster_eps_scale=2.0, cluster_min_samples=3,
                  cluster_recover_size_tolerance=0.1, default_bg_color=None,
                  color_test_initial_expand=1, color_test_border_thickness=1,
                  color_tolerance=25,
                  skip_legend=False, debug=True):
    print(f"\n##################\nProcessing image：{image_path}")
    # 获取 output_dir
    output_dir = os.path.dirname(output_image_path)
    
    # 从文件名中提取编号部分（例如，假设图像名是 "3.png"，我们提取 "3"）
    image_name = os.path.basename(output_image_path)
    image_base_name, _ = os.path.splitext(image_name)  # 获取文件名部分，不带扩展名

    main_img = cv2.imread(image_path)
    if main_img is None:
        raise FileNotFoundError(f"Unable to load: {image_path}")

    h_img, w_img = main_img.shape[:2]
    vis_img = main_img.copy()
    overlay = vis_img.copy()  # 用来画半透明区域

    predictions = predict_image_segmentation(main_img, model_path=model_path, smooth_contours=False,
                                              epsilon_factor=0.002,
                                              merge_same_class_boxes=False)
    all_boxes = []
    found_legend = False
    legend_counter = 0  # 用于编号
    filtered_out_boxes = []

    rec_id = 0

    if not skip_legend:
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
                    raise ValueError("No contours found in expanded mask")

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
                                                            binary_image_filename=binary_image_filename)

                selected = remove_overlapping_rect_simple(rectangles)

                kept_boxes, filtered_boxes = filter_boxes_by_uniform_color(selected,
                                                                            image=main_img,
                                                                            offset_xy=(x, y),  # legend 裁剪偏移
                                                                            initial_expand=color_test_initial_expand,
                                                                            border_thickness=color_test_border_thickness,
                                                                            default_bg_color=default_bg_color,
                                                                            color_tolerance=color_tolerance,
                                                                            debug=debug,
                                                                            debug_dir=output_dir,
                                                                            file_name=image_base_name,
                                                                            legend_counter=legend_counter,
                                                                            start_index=rec_id)
                all_boxes.extend(kept_boxes)
                filtered_out_boxes.extend(filtered_boxes)
                rec_id += len(selected)

                """
                for rect in selected:
                    rec_id += 1
                    x1, y1, x2, y2, score = rect
                    adjusted_rect = [x + x1, y + y1, x + x2, y + y2, score]
                    points = [
                        (x + x1, y + y1),
                        (x + x2, y + y1),
                        (x + x2, y + y2),
                        (x + x1, y + y2)
                    ]

                    if not is_box_surrounded_by_uniform_color(main_img, adjusted_rect, initial_expand=color_test_initial_expand, border_thickness=color_test_border_thickness,
                                                    default_bg_color=(255, 255, 255),
                                                    color_tolerance=15, debug_dir=output_dir if debug else None, file_name=f"{image_base_name}_{legend_counter}", index=rec_id):
                        filtered_out_boxes.append(points)
                        continue  # 跳过不合格的 box
                    
                    all_boxes.append(points)
                """

    if not found_legend or not all_boxes:
        if not found_legend:
            print(f"no legend found, searching the whole image: {image_path}")
        else:
            print(f"legend found, but no valid item box found in legend area, searching the whole image: {image_path}")

        legend_crop = main_img
        legend_area = w_img * h_img

        rectangles = obtain_legend_rectangle_bbox(legend_crop, legend_area,
                                                    area_min_factor=global_area_min_factor,
                                                    area_max_factor=global_area_max_factor,
                                                    binary_image_filename=os.path.join(output_dir, f"{image_base_name}_legend_binary.png")
                                                    )

        selected = remove_overlapping_rect_simple(rectangles)

        kept_boxes, filtered_boxes = filter_boxes_by_uniform_color(selected,
                                                                    image=main_img,
                                                                    offset_xy=(0, 0),  # legend 裁剪偏移
                                                                    initial_expand=color_test_initial_expand,
                                                                    border_thickness=color_test_border_thickness,
                                                                    default_bg_color=default_bg_color,
                                                                    color_tolerance=15,
                                                                    debug=debug,
                                                                    debug_dir=output_dir,
                                                                    file_name=image_base_name,
                                                                    legend_counter=legend_counter,
                                                                    start_index=rec_id)
        all_boxes.extend(kept_boxes)
        filtered_out_boxes.extend(filtered_boxes)
        rec_id += len(selected)

        """
        for rect in selected:
            rec_id += 1
            x1, y1, x2, y2, _ = rect
            points = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ]
            if not is_box_surrounded_by_uniform_color(main_img, rect, initial_expand=color_test_initial_expand, border_thickness=color_test_border_thickness, 
                                    default_bg_color=(255, 255, 255),
                                    color_tolerance=15, debug_dir=output_dir if debug else None, file_name=f"{image_base_name}_{legend_counter}", index=rec_id):
                filtered_out_boxes.append(points)
                continue  # 跳过不合格的 box
            
            all_boxes.append(points)
        """

    all_boxes = remove_overlapping_boxes_simple(all_boxes)

    filtered_out_boxes = remove_overlapping_boxes_simple(filtered_out_boxes)

    # --- 融合overlay到vis_img，产生半透明填充效果 ---
    clustered_boxes, labels, outlier_boxes, cluster_standards = filter_isolated_boxes_by_clustering_auto_eps(all_boxes, eps_scale=cluster_eps_scale, min_samples=cluster_min_samples)

    recovered_boxes = recover_boxes_by_size_match(outlier_boxes+filtered_out_boxes, cluster_standards, size_tolerance=cluster_recover_size_tolerance)

    filtered_boxes = refine_boxes_by_size_consistency(clustered_boxes, cluster_standards, size_tolerance=cluster_recover_size_tolerance)

    all_boxes = filtered_boxes + recovered_boxes
    #print (all_boxes)


    # Step 2：统一绘制 overlay 和 vis_img 中的框
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

    print(f"Saved {output_image_path} to {output_txt_path}")



def main():
    parser = argparse.ArgumentParser(description="Batch processing for extracting map legend regions and detecting item boxes")
    parser.add_argument('--input_dir', type=str, help="Path to the input root directory containing numbered subfolders")
    parser.add_argument('--output_dir', type=str, help="Path to the output directory where results will be saved")
    parser.add_argument('--model_path', type=str, help="Path to the YOLO model used for semantic segmentation")
    parser.add_argument('--legend_area_min_factor', type=float, default=0.001,
                        help="Minimum ratio of item box area to legend area when a legend is detected (default: 0.001)")
    parser.add_argument('--legend_area_max_factor', type=float, default=0.1,
                        help="Maximum ratio of item box area to legend area when a legend is detected (default: 0.1)")
    parser.add_argument('--global_area_min_factor', type=float, default=0.0005,
                        help="Minimum ratio of item box area to full image area when no legend is detected (default: 0.0001)")
    parser.add_argument('--global_area_max_factor', type=float, default=0.01,
                        help="Maximum ratio of item box area to full image area when no legend is detected (default: 0.01)")
    parser.add_argument('--expand_pixel', type=int, default=30,
                        help="Number of pixels to expand around the detected legend region for further processing (default: 30px)")
    parser.add_argument('--cluster_eps_scale', type=float, default=1.2,
                        help="Scale factor applied to estimated eps in DBSCAN clustering (default: 1.2)")
    parser.add_argument('--cluster_min_samples', type=int, default=3,
                        help="Minimum number of neighbors required to form a cluster in DBSCAN (default: 3)")
    parser.add_argument('--cluster_recover_size_tolerance', type=float, default=0.2,
                        help="Tolerance ratio when matching width/height to recover outlier boxes from clustering (default: 0.2)")
    parser.add_argument('--color_test_initial_expand', type=int, default=1,
                        help="Initial number of pixels to expand each box before checking border color uniformity (default: 1)")
    parser.add_argument('--color_test_border_thickness', type=int, default=1,
                        help="Thickness of the border (in pixels) to test for color consistency around the expanded box (default: 1)")
    parser.add_argument('--color_tolerance', type=float, default=25,
                        help="Tolerance for average border color difference (default: 25). ")
    parser.add_argument('--skip_legend', action='store_true',
                        help="If set, skip searching within legend box and directly detect items in the whole image.")


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    subdirs = [d for d in subdirs if d.isdigit()]

    #default_bg_color = None
    default_bg_color = (255, 255, 255)

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
              expand_pixel=args.expand_pixel,
              cluster_eps_scale=args.cluster_eps_scale, cluster_min_samples=args.cluster_min_samples,
              cluster_recover_size_tolerance=args.cluster_recover_size_tolerance, default_bg_color=default_bg_color,
              color_test_initial_expand=args.color_test_initial_expand, color_test_border_thickness=args.color_test_border_thickness,
              color_tolerance=args.color_tolerance,
              skip_legend=args.skip_legend)


if __name__ == "__main__":
    main()
