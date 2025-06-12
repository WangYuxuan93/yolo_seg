import cv2
import numpy as np
import os
import math
import random
import glob
import argparse
from collections import defaultdict, Counter
from itertools import combinations

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def is_rectangle_aligned(approx, angle_tolerance=15, alignment_tolerance=10):
    """
    判断轮廓是否是矩形，且矩形边与图像边缘（x/y轴）对齐。

    参数：
    - approx: cv2.approxPolyDP 输出的 4 点轮廓
    - angle_tolerance: 判断矩形角度偏差容忍度（默认 ±15°）
    - alignment_tolerance: 边方向与水平/垂直的容忍度（默认 ±10°）

    返回：
    - True 表示是对齐矩形，False 表示不是
    """
    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def classify_alignment(vec):
        angle = np.degrees(np.arctan2(vec[1], vec[0]))  # 与 x 轴夹角 [-180, 180]
        angle = abs(angle)  # 考虑角度对称性
        if angle <= alignment_tolerance or abs(angle - 180) <= alignment_tolerance:
            return 'horizontal'
        elif abs(angle - 90) <= alignment_tolerance:
            return 'vertical'
        else:
            return 'diagonal'

    pts = approx.reshape(4, 2)
    vectors = [pts[(i + 1) % 4] - pts[i] for i in range(4)]

    # 是否是矩形（角度近似90°）
    angles = [angle_between(vectors[i], vectors[(i + 1) % 4]) for i in range(4)]
    is_rectangular = all(abs(a - 90) <= angle_tolerance for a in angles)

    # 判断边方向（是否水平/垂直）
    alignments = [classify_alignment(v) for v in vectors]
    is_axis_aligned = alignments.count('horizontal') == 2 and alignments.count('vertical') == 2

    return is_rectangular and is_axis_aligned


def obtain_legend_rectangle_bbox(main_img, legend_area,
                                 area_min_factor=0.01, area_max_factor=0.5,
                                 binary_image_filename=None,
                                 contour_image_filename=None):
    print(f"\n[Search] Searching candidate item boxes")
    target_img = np.array(main_img)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # 自适应二值化
    blur = cv2.GaussianBlur(target_img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 5, 1
    )

    # 尝试闭运算以增强连通性
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if binary_image_filename is not None:
        cv2.imwrite(binary_image_filename, thresh)
        print(f"Binary image saved to: {binary_image_filename}")

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and is_rectangle_aligned(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            item_area = w * h
            if (0.9 < aspect_ratio < 3.5) and (legend_area * area_min_factor <= item_area <= legend_area * area_max_factor):
            #if (legend_area * area_min_factor <= item_area <= legend_area * area_max_factor):
                rectangles.append([x, y, x + w, y + h, 1.0])

    print("Found {} candidates".format(len(rectangles)))

    # 可视化未通过筛选的原始轮廓
    if contour_image_filename is not None:
        debug_img = main_img.copy()

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4 and is_rectangle_aligned(approx):
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h
                item_area = w * h
                #if (1.1 < aspect_ratio < 3.5) and (legend_area * area_min_factor <= item_area <= legend_area * area_max_factor):
                #    continue  # 通过筛选的跳过

                # 不通过的原始轮廓画红色线
                cv2.drawContours(debug_img, [contour], -1, (0, 0, 255), 1)

        cv2.imwrite(contour_image_filename, debug_img)
        print(f"Debug image with raw contours saved to: {contour_image_filename}")

    return rectangles




def remove_boxes_with_small_edge_distance(boxes, min_distance=1):
    """
    移除任意两个 box 之间边缘距离小于 min_distance 的所有 box。
    box 是 4 点形式的 list。
    """
    def box_edge_distance(box1, box2):
        """
        计算两个 box 之间的最小边缘距离，box 是 4 个点的 list。
        """
        pts1 = np.array(box1)
        pts2 = np.array(box2)

        # 获取两个 box 的边界框坐标 [x_min, y_min, x_max, y_max]
        x1_min, y1_min = np.min(pts1, axis=0)
        x1_max, y1_max = np.max(pts1, axis=0)

        x2_min, y2_min = np.min(pts2, axis=0)
        x2_max, y2_max = np.max(pts2, axis=0)

        # 计算水平和垂直方向的边缘距离（不重叠为正值）
        dx = max(x1_min - x2_max, x2_min - x1_max, 0)
        dy = max(y1_min - y2_max, y2_min - y1_max, 0)

        return np.hypot(dx, dy)

    to_remove = set()
    for i, j in combinations(range(len(boxes)), 2):
        dist = box_edge_distance(boxes[i], boxes[j])
        if dist < min_distance:
            to_remove.add(i)
            to_remove.add(j)
    
    filtered_boxes = [box for idx, box in enumerate(boxes) if idx not in to_remove]
    print (f"\n[Distance] Input: {len(boxes)}, Output: {len(filtered_boxes)}")
    return filtered_boxes


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

    edge_blocks = []
    label_list = []
    img_list = []

    if y1_ext > y1_outer:
        top = image[y1_outer:y1_ext, x1_outer:x2_outer]
        if top.size > 0:
            edge_blocks.append(top.reshape(-1, 3))
            label_list.append("TOP")
            img_list.append(top)
    if y2_outer > y2_ext:
        bottom = image[y2_ext:y2_outer, x1_outer:x2_outer]
        if bottom.size > 0:
            edge_blocks.append(bottom.reshape(-1, 3))
            label_list.append("BOTTOM")
            img_list.append(bottom)
    if x1_ext > x1_outer:
        left = image[y1_outer:y2_outer, x1_outer:x1_ext]
        if left.size > 0:
            edge_blocks.append(left.reshape(-1, 3))
            label_list.append("LEFT")
            img_list.append(left)
    if x2_outer > x2_ext:
        right = image[y1_outer:y2_outer, x2_ext:x2_outer]
        if right.size > 0:
            edge_blocks.append(right.reshape(-1, 3))
            label_list.append("RIGHT")
            img_list.append(right)

    if not edge_blocks:
        return False, None

    border_pixels = np.concatenate(edge_blocks, axis=0)

    # 判断颜色一致性
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

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        label_height = 20
        min_visual_width = 40

        img_list.append(image[y1:y2, x1:x2])
        label_list.append("ITEM")

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

    return result, ref_color


def filter_boxes_by_uniform_color(boxes, image, offset_xy=(0, 0),
                                   initial_expand=1,
                                   border_thickness=1,
                                   default_bg_color=None,
                                   color_tolerance=15,
                                   color_rounding=10,  # <-- 新增参数
                                   debug=False,
                                   debug_dir=None,
                                   file_name=None,
                                   legend_counter=None,
                                   start_index=0):
    """
    对 box 列表进行颜色一致性判断，过滤掉边缘颜色不符合的 box，
    并在通过初筛后统计 dominant color，只保留颜色接近主色的 box。

    参数：
    - boxes: List[List[int]]，每个为 [x1, y1, x2, y2, score]
    - image: 原图
    - offset_xy: (x_offset, y_offset)，box 坐标偏移量（如 legend 裁剪偏移）
    - color_tolerance: 与 dominant color 差值容忍度（欧几里得距离）
    - color_rounding: int，颜色量化桶大小（默认5，即四舍五入到5的倍数）
    - 返回：
        - kept_boxes: 最终保留 box（四点格式）
        - filtered_out_boxes: 被丢弃的 box（四点格式）
    """
    print("\n[Color] Filtering boxes by surrounded color")

    x_offset, y_offset = offset_xy
    temp_kept = []  # [(points, mean_color)]
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

        passed, mean_color = is_box_surrounded_by_uniform_color(
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
            temp_kept.append((points, mean_color))
        else:
            filtered_out_boxes.append(points)

    if not temp_kept:
        print("No box passed initial uniform color check.")
        return [], filtered_out_boxes

    print(f"Input: {len(boxes)}, Output: {len(temp_kept)}")

    return [points for points, color in temp_kept], filtered_out_boxes


def filter_by_dominant_edge_color(image, boxes, labels, color_tolerance=15, bucket_size=5,
                                   initial_expand=1, border_thickness=1, debug_dir=None, file_name="debug"):
    """
    对聚类后的 box 根据 dominant edge color 进行筛除（返回原始四点格式的 box）

    参数：
    - image: 原图 (BGR)
    - boxes: 每个 box 为 [(x1,y1), ...] 四个点
    - labels: 聚类标签
    - color_tolerance: 与聚类 dominant color 的容差
    - bucket_size: 颜色离散桶大小
    - initial_expand, border_thickness: 边缘扩展参数
    - debug_dir: 如果指定路径，则保存被过滤掉的 box 的调试图
    - file_name: 调试图片文件名前缀

    返回：
    - kept_boxes: 保留的原始四点 box
    - kept_labels: 对应 label
    - removed_boxes: 被剔除的原始四点 box
    """
   
    print(f"\n[Cluster Color Filter]")

    def round_color(color, bucket=5):
        return tuple((np.array(color) // bucket * bucket).astype(int))

    def to_rect_from_points(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), 1.0]

    os.makedirs(debug_dir, exist_ok=True) if debug_dir else None

    box_data = []
    for box, label in zip(boxes, labels):
        rect = to_rect_from_points(box)
        x1, y1, x2, y2, *_ = rect
        h, w = image.shape[:2]
        x1_ext = max(x1 - initial_expand, 0)
        y1_ext = max(y1 - initial_expand, 0)
        x2_ext = min(x2 + initial_expand, w - 1)
        y2_ext = min(y2 + initial_expand, h - 1)
        x1_outer = max(x1_ext - border_thickness, 0)
        y1_outer = max(y1_ext - border_thickness, 0)
        x2_outer = min(x2_ext + border_thickness, w - 1)
        y2_outer = min(y2_ext + border_thickness, h - 1)

        edge_blocks = []
        label_list = []
        img_list = []

        if y1_ext > y1_outer:
            top = image[y1_outer:y1_ext, x1_outer:x2_outer]
            if top.size > 0:
                edge_blocks.append(top.reshape(-1, 3))
                label_list.append("TOP")
                img_list.append(top)
        if y2_outer > y2_ext:
            bottom = image[y2_ext:y2_outer, x1_outer:x2_outer]
            if bottom.size > 0:
                edge_blocks.append(bottom.reshape(-1, 3))
                label_list.append("BOTTOM")
                img_list.append(bottom)
        if x1_ext > x1_outer:
            left = image[y1_outer:y2_outer, x1_outer:x1_ext]
            if left.size > 0:
                edge_blocks.append(left.reshape(-1, 3))
                label_list.append("LEFT")
                img_list.append(left)
        if x2_outer > x2_ext:
            right = image[y1_outer:y2_outer, x2_ext:x2_outer]
            if right.size > 0:
                edge_blocks.append(right.reshape(-1, 3))
                label_list.append("RIGHT")
                img_list.append(right)

        if not edge_blocks:
            box_data.append((box, label, None))
            continue

        border_pixels = np.concatenate(edge_blocks, axis=0)
        mean_color = np.mean(border_pixels, axis=0)

        rect_img = image[y1:y2, x1:x2]
        img_list.append(rect_img)
        label_list.append("ITEM")

        box_data.append((box, label, mean_color, label_list.copy(), img_list.copy(), np.mean(border_pixels, axis=0)))

    cluster_colors = defaultdict(list)
    cluster_centers = defaultdict(list)
    for box, label, *rest in box_data:
        color = rest[0] if rest else None
        if label != -1:
            if color is not None:
                cluster_colors[label].append(round_color(color, bucket_size))
            center = np.mean(np.array(box), axis=0)
            cluster_centers[label].append(center)

    cluster_dominant = {
        label: Counter(colors).most_common(1)[0][0]
        for label, colors in cluster_colors.items()
    }

    kept_boxes, kept_labels, removed_boxes = [], [], []
    per_cluster_counts = defaultdict(lambda: {"in": 0, "kept": 0})

    for i, (box, label, color, label_list, img_list, mean_color) in enumerate(box_data):
        center = np.mean(box, axis=0)

        if label == -1 or color is None:
            removed_boxes.append(box)
            reason = "label=-1 (unclustered)" if label == -1 else "mean_color=None (edge color extraction failed)"
            print(f" - Removed box at ({center[0]:.1f}, {center[1]:.1f}): {reason}")
            continue

        per_cluster_counts[label]["in"] += 1
        dom_color = np.array(cluster_dominant[label])
        dist = np.linalg.norm(color - dom_color)

        if dist <= color_tolerance:
            kept_boxes.append(box)
            kept_labels.append(label)
            per_cluster_counts[label]["kept"] += 1
        else:
            removed_boxes.append(box)
            print(f" - Removed box at ({center[0]:.1f}, {center[1]:.1f}), cluster {label}:")
            print(f"    > Dominant color for cluster {label}: {tuple(dom_color.astype(int))}")
            print(f"    > Box edge mean color: {tuple(color.astype(int))}")
            print(f"    > Color distance = {dist:.1f}, exceeds tolerance = {color_tolerance}")

            if debug_dir:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1
                label_height = 20
                min_visual_width = 40

                labeled_blocks = []
                max_height = 0

                for label_txt, img in zip(label_list, img_list):
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
                    text_size = cv2.getTextSize(label_txt, font, font_scale, font_thickness)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (label_height + text_size[1]) // 2
                    cv2.putText(label_img, label_txt, (text_x, text_y), font, font_scale,
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

                ref_r, ref_g, ref_b = [int(x) for x in dom_color]
                text_lines = [
                    f"Reference RGB: ({ref_r}, {ref_g}, {ref_b})",
                    f"Mean Diff: {dist:.2f}",
                    f"Tolerance: {color_tolerance}",
                    f"Pass: No"
                ]
                for j, line in enumerate(text_lines):
                    y = 15 + j * 18
                    cv2.putText(annotated, line, (10, y), font, font_scale,
                                (0, 0, 255), font_thickness, cv2.LINE_AA)

                debug_path = os.path.join(debug_dir, f"{file_name}_box_{i}_debug_cluster_color.png")
                cv2.imwrite(debug_path, annotated)

    valid_input_count = sum(1 for b, l in zip(boxes, labels) if l != -1)
    print(f"Input: {valid_input_count}, Output: {len(kept_boxes)}")
    for label in sorted(cluster_dominant.keys()):
        center_pts = np.array(cluster_centers[label])
        center_mean = np.mean(center_pts, axis=0)
        dom_color = cluster_dominant[label]
        count_in = per_cluster_counts[label]["in"]
        count_kept = per_cluster_counts[label]["kept"]
        print(f" - Cluster {label}: center = ({center_mean[0]:.1f}, {center_mean[1]:.1f}), "
              f"dominant RGB = {dom_color}, kept = {count_kept}/{count_in}")

    return kept_boxes, kept_labels, removed_boxes


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
    print ("Input: {}".format(len(boxes)))
    if len(boxes) == 0:
        return [], [], [], {}

    centers = np.array([
        [np.mean([p[0] for p in box]), np.mean([p[1] for p in box])]
        for box in boxes
    ])

    if len(centers) <= min_samples:
        print(f"Box count ({len(centers)}) <= min_samples ({min_samples}), skipping DBSCAN.")

        labels = [0] * len(boxes)
        cluster_sizes = defaultdict(list)
        for box in boxes:
            w = np.linalg.norm(np.array(box[1]) - np.array(box[0]))
            h = np.linalg.norm(np.array(box[3]) - np.array(box[0]))
            cluster_sizes[0].append((w, h))

        cluster_standards = {}
        print("Standard width and height:")
        widths = [wh[0] for wh in cluster_sizes[0]]
        heights = [wh[1] for wh in cluster_sizes[0]]
        std_w = np.median(widths)
        std_h = np.median(heights)
        cluster_standards[0] = (std_w, std_h)
        print(f" - Cluster 0: width = {std_w:.1f}, height = {std_h:.1f} ({len(widths)} boxes)")

        return boxes, labels, [], cluster_standards

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
    clustered_labels = [label for box, label in zip(boxes, labels) if label != -1]

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

    return clustered_boxes, clustered_labels, outlier_boxes, cluster_standards


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

def remove_overlapping_boxes_simple(rectangles, type="all"):
    print (f"\n[Overlap] Removing overlapping from {type} boxes")
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

    print ("Input: {}, Output: {}".format(len(rectangles), len(selected)))

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
    failed_boxes = []
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
            failed_boxes.append(box)
            print(f"Box filtered out: width = {w:.1f}, height = {h:.1f} (no match)")

    print(f"Retained {len(refined_boxes)} of {len(boxes)} boxes after size check.")
    return refined_boxes, failed_boxes


def load_predicted_text_boxes(pred_txt_path, angle_tolerance=15):
    """
    从 pred_image.txt 文件中读取文本框坐标，仅保留边缘平行或垂直的 box。

    参数：
    - pred_txt_path: str，文件路径，如 "bbox/pred_image.txt"
    - angle_tolerance: float，角度容忍度（单位：度，默认 ±15°）

    返回：
    - boxes: List[List[Tuple[int, int]]]，每个 box 为四个点的列表
    """
    def is_aligned(points, tolerance_deg=15):
        """检查 box 是否与图像边缘平行/垂直"""
        def angle(p1, p2):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return math.degrees(math.atan2(dy, dx))

        angles = []
        for i in range(4):
            a = points[i]
            b = points[(i + 1) % 4]
            theta = abs(angle(a, b)) % 180  # 只考虑正向角
            angles.append(theta)

        for theta in angles:
            if not (
                abs(theta - 0) <= tolerance_deg or
                abs(theta - 90) <= tolerance_deg or
                abs(theta - 180) <= tolerance_deg
            ):
                return False
        return True

    print(f"[Load] Reading predicted boxes from: {pred_txt_path}")
    boxes = []
    total = 0
    if not os.path.exists(pred_txt_path):
        print(f"[Warning] File not found: {pred_txt_path}")
        return boxes

    with open(pred_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(int, line.split(',')))
            if len(parts) != 8:
                print(f"[Warning] Skipped invalid line: {line}")
                continue
            total += 1
            points = [(parts[i], parts[i + 1]) for i in range(0, 8, 2)]
            if is_aligned(points, tolerance_deg=angle_tolerance):
                boxes.append(points)

    print(f"[Filter] Total boxes: {total}, Kept after angle filtering: {len(boxes)}")
    return boxes


def find_item_boxes_with_nearby_text(item_boxes, text_boxes, 
                                     h_thresh_ratio=1.0, w_thresh_ratio=2.0,
                                     offset_xy=(0, 0)):
    """
    判断 item box 是否存在右方或下方的非重叠 OCR 文本框（满足空间约束），并返回匹配对。

    参数：
    - item_boxes: List[List[int]]，每个为 [x1, y1, x2, y2, score]
    - text_boxes: List[List[Tuple[int, int]]]，OCR文本框，每个为4点
    - h_thresh_ratio: float，下方容忍度比例（默认 0.3 × 高度）
    - w_thresh_ratio: float，右方容忍度比例（默认 2.0 × 高度）
    - offset_xy: Tuple[int, int]，可选，对 item_boxes 添加的 (x, y) 偏移

    返回：
    - matched_pairs: List[Tuple[item_box, matched_text_box]]
        item_box 是加过 offset 的坐标，matched_text_box 是原始 4 点文本框
    """
    def box_to_bbox(box4pt):
        xs = [p[0] for p in box4pt]
        ys = [p[1] for p in box4pt]
        return [min(xs), min(ys), max(xs), max(ys)]

    x_offset, y_offset = offset_xy
    text_bboxes = [box_to_bbox(tb) for tb in text_boxes]

    matched_pairs = []

    # 应用偏移到所有 item box
    offset_item_boxes = []
    for box in item_boxes:
        x1, y1, x2, y2, score = box
        offset_item_boxes.append([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset, score])

    for ibox in offset_item_boxes:
        x1, y1, x2, y2, score = ibox
        h = y2 - y1
        w = x2 - x1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        for tbox_raw, tbox in zip(text_boxes, text_bboxes):
            tx1, ty1, tx2, ty2 = tbox

            # --- 跳过与其他 item box 重叠的文本框 ---
            overlap = False
            for obox in offset_item_boxes:
                if obox == ibox:
                    continue
                ox1, oy1, ox2, oy2, _ = obox
                if not (tx2 <= ox1 or tx1 >= ox2 or ty2 <= oy1 or ty1 >= oy2):
                    overlap = True
                    break
            if overlap:
                continue

            # --- 判断正右方 ---
            if tx1 >= x2:
                text_center_y = (ty1 + ty2) / 2
                if y1 <= text_center_y <= y2 and (tx1 - x2 <= w_thresh_ratio * h):
                    matched_pairs.append((ibox, tbox_raw))
                    break

            # --- 判断正下方 ---
            if ty1 >= y2:
                text_center_x = (tx1 + tx2) / 2
                if x1 <= text_center_x <= x2 and (ty1 - y2 <= h_thresh_ratio * h):
                    matched_pairs.append((ibox, tbox_raw))
                    break


    print(f"\n[Match] Found {len(matched_pairs)} item-text box pairs.")
    return matched_pairs


def draw_clusters_with_labels(image, boxes, labels, save_path, thickness=2):
    """
    使用预定义颜色绘制 cluster box。

    参数：
    - image: 原图
    - boxes: 每个 box 为 [(x1,y1), ...] 四点格式
    - labels: 与 boxes 对应的聚类编号
    - save_path: 图像保存路径
    """
    vis = image.copy()
    unique_labels = sorted(set(labels))

    # 颜色池（可自行扩展）
    PREDEFINED_COLORS = [
        (255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 165, 0),
        (128, 0, 128), (0, 255, 255), (255, 192, 203), (0, 0, 0),
        (139, 69, 19), (255, 255, 0), (0, 255, 0), (70, 130, 180),
        (255, 20, 147)
    ]

    label_colors = {}
    for i, label in enumerate(unique_labels):
        color = PREDEFINED_COLORS[i % len(PREDEFINED_COLORS)]
        label_colors[label] = color

    for box, label in zip(boxes, labels):
        color = label_colors[label]
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

    cv2.imwrite(save_path, vis)
    print(f"[Debug] Cluster visualization saved to: {save_path}")


def filter_duplicate_pure_color_boxes(
    boxes,
    image,
    duplicate_filter_size_tolerance=2,
    duplicate_filter_color_tolerance=10,
    shrink_pixels=2,
    color_std_max_threshold=20,
    debug_output_dir=None,
    file_name="image"
):
    """
    过滤重复纯色 box（尺寸一致 + 颜色一致 + 标准差低），并保存每个 box 的调试图像。

    图像特征：
    - 白色背景
    - 居中 box（缩放到统一高度）
    - 灰色边框标示 box 范围
    - 左上角竖排显示 Mean, Std, Size 信息

    参数：
    - boxes: List[List[(x, y)]]
    - image: 原图（BGR）
    - duplicate_filter_size_tolerance: 尺寸容差（像素）
    - duplicate_filter_color_tolerance: 颜色容差（欧氏距离）
    - shrink_pixels: 从 box 四边向内收缩像素数
    - debug_output_dir: 若不为 None，则保存所有 box 的 debug 图像
    - file_name: 图像名前缀（通常为原图文件名）

    返回：
    - retained_boxes: 保留的 box
    - removed_boxes: 被剔除的 box
    """

    if debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        target_height = 64  # 统一缩放高度

    def get_box_stats(box):
        x1, y1 = box[0]
        x3, y3 = box[2]

        # 尝试收缩 box
        x1s = min(x1 + shrink_pixels, x3 - 1)
        x3s = max(x3 - shrink_pixels, x1 + 1)
        y1s = min(y1 + shrink_pixels, y3 - 1)
        y3s = max(y3 - shrink_pixels, y1 + 1)

        roi = image[y1s:y3s, x1s:x3s]

        # 如果收缩后的区域为空，退回原 box 区域
        if roi.size == 0:
            roi = image[y1:y3, x1:x3]
            #return (0, 0), np.array([0, 0, 0]), np.array([255, 255, 255]), roi

        mean_color = np.mean(roi.reshape(-1, 3), axis=0)
        std_color = np.std(roi.reshape(-1, 3), axis=0)
        w, h = x3 - x1, y3 - y1
        return (w, h), mean_color, std_color, roi

    group_map = defaultdict(list)

    for idx, box in enumerate(boxes):
        size, mean_color, std_color, roi = get_box_stats(box)
        w, h = size
        mean_rgb = tuple((mean_color // duplicate_filter_color_tolerance *
                          duplicate_filter_color_tolerance).astype(int))
        std_max = np.max(std_color)

        # === 保存调试图像（所有 box） ===
        if debug_output_dir and roi.size > 0:
            mean_i = mean_color.astype(int)
            std_i = std_color.astype(int)
            lines = [
                f"Mean: {tuple(mean_i)}",
                f"Std:  {tuple(std_i)}",
                f"Size: ({w},{h})"
            ]

            # === 缩放 roi 到统一高度 ===
            scale = target_height / roi.shape[0]
            new_w = max(1, int(roi.shape[1] * scale))
            roi_resized = cv2.resize(roi, (new_w, target_height), interpolation=cv2.INTER_AREA)

            # === 创建白色背景并加边框 ===
            pad = 8
            bg_h, bg_w = roi_resized.shape[0] + 2 * pad, roi_resized.shape[1] + 2 * pad
            debug_img = np.ones((bg_h, bg_w, 3), dtype=np.uint8) * 255
            debug_img[pad:pad + roi_resized.shape[0], pad:pad + roi_resized.shape[1]] = roi_resized
            cv2.rectangle(debug_img, (pad, pad),
                          (pad + roi_resized.shape[1] - 1, pad + roi_resized.shape[0] - 1),
                          (180, 180, 180), 1)

            for i, line in enumerate(lines):
                y = 12 + i * 12
                cv2.putText(debug_img, line, (5, y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            debug_path = os.path.join(debug_output_dir, f"{file_name}_dup_box_{idx}_debug.png")
            cv2.imwrite(debug_path, debug_img)

        # === 满足纯色才分组 ===
        if std_max < color_std_max_threshold:
            key = (round(w / duplicate_filter_size_tolerance),
                   round(h / duplicate_filter_size_tolerance),
                   mean_rgb)
            group_map[key].append(idx)

    retained, removed = [], set()
    for group in group_map.values():
        if len(group) >= 2:
            for idx in group:
                removed.add(idx)

    for idx, box in enumerate(boxes):
        if idx not in removed:
            retained.append(box)

    print(f"\n[Duplicate Filter] Duplicate Pure-Color Box Filtering:")
    print(f"  Input  : {len(boxes)} boxes")
    print(f"  Output : {len(retained)} boxes")
    print(f"  Removed: {len(removed)} boxes (same size + pure color + same color)")
    if removed:
        sample_key = list(group_map.keys())[0]
        print(f"  Sample removed group key: {sample_key}")

    return retained, [boxes[i] for i in removed]


"""
def process_image(image_path, output_image_path, output_txt_path, model_path,
                  legend_area_min_factor=0.001, legend_area_max_factor=0.1,
                  global_area_min_factor=0.0001, global_area_max_factor=0.01,
                  expand_pixel=30, cluster_eps_scale=2.0, cluster_min_samples=3,
                  cluster_recover_size_tolerance=0.1, default_bg_color=None,
                  color_test_initial_expand=1, color_test_border_thickness=1,
                  color_tolerance=25, duplicate_filter_size_tolerance=5,
                  duplicate_filter_color_tolerance=2, duplicate_filter_shrink_pixels=4,
                  duplicate_filter_color_std_max_threshold=20,
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

    all_boxes = []
    found_legend = False
    legend_counter = 0  # 用于编号
    filtered_out_boxes = []
    rec_id = 0

    all_matched_pairs = []

    legend_crop = main_img
    legend_area = w_img * h_img

    rectangles = obtain_legend_rectangle_bbox(legend_crop, legend_area,
                                                area_min_factor=global_area_min_factor,
                                                area_max_factor=global_area_max_factor,
                                                binary_image_filename=os.path.join(output_dir, f"{image_base_name}_legend_binary.png") if debug else None,
                                                contour_image_filename=os.path.join(output_dir, f"{image_base_name}_contour.png")) if debug else None

    selected = remove_overlapping_rect_simple(rectangles)

    #matched_pairs = find_item_boxes_with_nearby_text(selected, predicted_text_boxes, offset_xy=(0, 0))
    #all_matched_pairs.extend(matched_pairs)

    kept_boxes, filtered_boxes = filter_boxes_by_uniform_color(selected,
                                                                image=main_img,
                                                                offset_xy=(0, 0),  # legend 裁剪偏移
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

    all_boxes = remove_overlapping_boxes_simple(all_boxes, type="all")

    filtered_out_boxes = remove_overlapping_boxes_simple(filtered_out_boxes, type="filtered out")

    #all_boxes = remove_boxes_with_small_edge_distance(all_boxes, min_distance=5)

    all_boxes, removed_boxes = filter_duplicate_pure_color_boxes(all_boxes, main_img,
                                                                duplicate_filter_size_tolerance=duplicate_filter_size_tolerance,
                                                                duplicate_filter_color_tolerance=duplicate_filter_color_tolerance,
                                                                shrink_pixels=duplicate_filter_shrink_pixels,
                                                                color_std_max_threshold=duplicate_filter_color_std_max_threshold,
                                                                debug_output_dir=output_dir if debug else None,
                                                                file_name=image_base_name)

    # --- 融合overlay到vis_img，产生半透明填充效果 ---
    clustered_boxes, labels, outlier_boxes, cluster_standards = filter_isolated_boxes_by_clustering_auto_eps(all_boxes, eps_scale=cluster_eps_scale, min_samples=cluster_min_samples)

    if debug:
        draw_clusters_with_labels(image=main_img,
                                boxes=clustered_boxes,
                                labels=labels,
                                save_path=os.path.join(output_dir,f"{image_base_name}_cluster_before_color_filter.png"))

    kept_boxes, kept_labels, removed_by_color = filter_by_dominant_edge_color(image=main_img,
                                                                                boxes=clustered_boxes,
                                                                                labels=labels,
                                                                                color_tolerance=color_tolerance,
                                                                                bucket_size=10,
                                                                                initial_expand=color_test_initial_expand,
                                                                                border_thickness=color_test_border_thickness,
                                                                                debug_dir=output_dir if debug else None,
                                                                                file_name=image_base_name)

    if debug:
        draw_clusters_with_labels(
            image=main_img,
            boxes=kept_boxes,
            labels=kept_labels,
            save_path=os.path.join(output_dir,f"{image_base_name}_cluster.png"))


    recovered_boxes = recover_boxes_by_size_match(outlier_boxes+filtered_out_boxes+removed_by_color, cluster_standards, size_tolerance=cluster_recover_size_tolerance)
    #recovered_boxes = recover_boxes_by_size_match(outlier_boxes, cluster_standards, size_tolerance=cluster_recover_size_tolerance)

    filtered_boxes = refine_boxes_by_size_consistency(kept_boxes, cluster_standards, size_tolerance=cluster_recover_size_tolerance)

    all_boxes = filtered_boxes + recovered_boxes
    #print (all_boxes)

    if not all_boxes:
        print ("\n[Fallback] No box remained, fall back to boxes filtered out")
        print ("Final box: {}".format(len(filtered_out_boxes)))
        all_boxes = filtered_out_boxes

    # Step 2：统一绘制 overlay 和 vis_img 中的框
    for box in all_boxes:
        # 每个 box 是四个点 [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        x1, y1 = box[0]
        x3, y3 = box[2]

        # 填充区域
        cv2.rectangle(overlay, (x1, y1), (x3, y3), (0, 255, 0), -1)
        # 边框
        cv2.rectangle(vis_img, (x1, y1), (x3, y3), (0, 255, 0), 2)

    # ✅ 额外画出红色的 predicted text boxes
    #for box in predicted_text_boxes:
    #    x1, y1 = box[0]
    #    x3, y3 = box[2]
    #    cv2.rectangle(vis_img, (x1, y1), (x3, y3), (0, 0, 255), 2)

    alpha = 0.2  # 半透明程度
    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

    cv2.imwrite(output_image_path, vis_img)

    with open(output_txt_path, 'w') as f:
        for points in all_boxes:
            line = ",".join([f"{px},{py}" for (px, py) in points])
            f.write(line + "\n")

    print(f"Saved {output_image_path} to {output_txt_path}")
"""


def process_image_from_array(image, image_name,
                             global_area_min_factor=0.0001, global_area_max_factor=0.01,
                             cluster_eps_scale=2.0, cluster_min_samples=3,
                             cluster_recover_size_tolerance=0.1, default_bg_color=None,
                             color_test_initial_expand=1, color_test_border_thickness=1,
                             color_tolerance=25, duplicate_filter_size_tolerance=5,
                             duplicate_filter_color_tolerance=2, duplicate_filter_shrink_pixels=4,
                             duplicate_filter_color_std_max_threshold=20,
                             debug=True, debug_dir="."):
    """
    从图像对象中提取图例项 box，返回通过筛选的 box 列表（四点坐标形式）
    """
    image_base_name, _ = os.path.splitext(image_name)
    h_img, w_img = image.shape[:2]
    all_boxes, filtered_out_boxes, rec_id = [], [], 0
    legend_area = w_img * h_img

    rectangles = obtain_legend_rectangle_bbox(
        image, legend_area,
        area_min_factor=global_area_min_factor,
        area_max_factor=global_area_max_factor,
        binary_image_filename=os.path.join(debug_dir, f"{image_base_name}_legend_binary.png") if debug else None,
        contour_image_filename=os.path.join(debug_dir, f"{image_base_name}_contour.png") if debug else None)

    selected = remove_overlapping_rect_simple(rectangles)

    kept_boxes, filtered_boxes = filter_boxes_by_uniform_color(
        selected, image=image, offset_xy=(0, 0),
        initial_expand=color_test_initial_expand,
        border_thickness=color_test_border_thickness,
        default_bg_color=default_bg_color,
        color_tolerance=color_tolerance,
        debug=debug,
        debug_dir=debug_dir,
        file_name=image_base_name,
        legend_counter=0,
        start_index=rec_id)

    all_boxes.extend(kept_boxes)
    filtered_out_boxes.extend(filtered_boxes)

    all_boxes = remove_overlapping_boxes_simple(all_boxes, type="all")
    filtered_out_boxes = remove_overlapping_boxes_simple(filtered_out_boxes, type="filtered out")

    all_boxes, _ = filter_duplicate_pure_color_boxes(
        all_boxes, image,
        duplicate_filter_size_tolerance=duplicate_filter_size_tolerance,
        duplicate_filter_color_tolerance=duplicate_filter_color_tolerance,
        shrink_pixels=duplicate_filter_shrink_pixels,
        color_std_max_threshold=duplicate_filter_color_std_max_threshold,
        debug_output_dir=debug_dir if debug else None,
        file_name=image_base_name)

    clustered_boxes, labels, outlier_boxes, cluster_standards = filter_isolated_boxes_by_clustering_auto_eps(
        all_boxes, eps_scale=cluster_eps_scale, min_samples=cluster_min_samples)

    if debug:
        draw_clusters_with_labels(image, clustered_boxes, labels,
                                  save_path=os.path.join(debug_dir, f"{image_base_name}_cluster_before_color_filter.png"))

    kept_boxes, kept_labels, removed_by_color = filter_by_dominant_edge_color(
        image=image, boxes=clustered_boxes, labels=labels,
        color_tolerance=color_tolerance, bucket_size=10,
        initial_expand=color_test_initial_expand,
        border_thickness=color_test_border_thickness,
        debug_dir=debug_dir if debug else None,
        file_name=image_base_name)

    if debug:
        draw_clusters_with_labels(image, kept_boxes, kept_labels,
                                  save_path=os.path.join(debug_dir, f"{image_base_name}_cluster.png"))

    recovered_boxes = recover_boxes_by_size_match(
        outlier_boxes + filtered_out_boxes + removed_by_color,
        cluster_standards,
        size_tolerance=cluster_recover_size_tolerance)

    filtered_boxes, failed_boxes = refine_boxes_by_size_consistency(
        kept_boxes, cluster_standards, size_tolerance=cluster_recover_size_tolerance)

    final_boxes = filtered_boxes + recovered_boxes
    if not final_boxes:
        print("\n[Fallback] No box remained, fall back to filtered out")
        final_boxes = filtered_out_boxes

    return final_boxes, filtered_out_boxes + failed_boxes


def extract_legend_box_from_image(image, image_name=None):
    """
    输入一张图像，手动指定所有参数，调用 process_image_from_array。
    
    返回：
    - final_boxes: 通过所有筛选的 box
    - filtered_out_boxes: 被中途筛掉的 box
    """

    # 手动设置参数（你可以在这里灵活调整）
    image_name = image_name if image_name is not None else "manual_input.png"
    global_area_min_factor = 0.0001
    global_area_max_factor = 0.01
    cluster_eps_scale = 2.0
    cluster_min_samples = 3
    cluster_recover_size_tolerance = 0.1
    default_bg_color = None
    color_test_initial_expand = 1
    color_test_border_thickness = 1
    color_tolerance = 25
    duplicate_filter_size_tolerance = 5
    duplicate_filter_color_tolerance = 2
    duplicate_filter_shrink_pixels = 4
    duplicate_filter_color_std_max_threshold = 20
    debug = False
    debug_dir = "./debug_output"  # 可选调试目录

    correct_flag = True
    try:
        # 调用主函数
        final_boxes, filtered_out_boxes = process_image_from_array(
            image=image,
            image_name=image_name,
            global_area_min_factor=global_area_min_factor,
            global_area_max_factor=global_area_max_factor,
            cluster_eps_scale=cluster_eps_scale,
            cluster_min_samples=cluster_min_samples,
            cluster_recover_size_tolerance=cluster_recover_size_tolerance,
            default_bg_color=default_bg_color,
            color_test_initial_expand=color_test_initial_expand,
            color_test_border_thickness=color_test_border_thickness,
            color_tolerance=color_tolerance,
            duplicate_filter_size_tolerance=duplicate_filter_size_tolerance,
            duplicate_filter_color_tolerance=duplicate_filter_color_tolerance,
            duplicate_filter_shrink_pixels=duplicate_filter_shrink_pixels,
            duplicate_filter_color_std_max_threshold=duplicate_filter_color_std_max_threshold,
            debug=debug,
            debug_dir=debug_dir)
    except:
        correct_flag = False
        return [], [], correct_flag

    return final_boxes, filtered_out_boxes, correct_flag


def extract_legend_box_from_path(image_path, save_path=None):
    """
    从图片路径中提取图例 box，并可视化：
    - final_boxes：绿色填充 + 绿色边框
    - filtered_out_boxes：红色填充 + 红色边框

    参数：
    - image_path: str，图片路径
    - save_path: 可选，可视化图像保存路径

    返回：
    - final_boxes: List[List[(x, y)]]
    - filtered_out_boxes: List[List[(x, y)]]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图片: {image_path}")

    image_name = os.path.basename(image_path)
    final_boxes, filtered_out_boxes = extract_legend_box_from_image(image, image_name=image_name)

    overlay = image.copy()
    vis_img = image.copy()

    # ✅ 绘制 final_boxes：绿色填充 + 边框
    for box in final_boxes:
        x1, y1 = box[0]
        x3, y3 = box[2]
        cv2.rectangle(overlay, (x1, y1), (x3, y3), (0, 255, 0), -1)  # 绿色填充
        cv2.rectangle(vis_img, (x1, y1), (x3, y3), (0, 255, 0), 2)   # 绿色边框

    # ✅ 绘制 filtered_out_boxes：红色填充 + 边框
    for box in filtered_out_boxes:
        x1, y1 = box[0]
        x3, y3 = box[2]
        cv2.rectangle(overlay, (x1, y1), (x3, y3), (0, 0, 255), -1)  # 红色填充
        cv2.rectangle(vis_img, (x1, y1), (x3, y3), (0, 0, 255), 2)   # 红色边框

    # 半透明融合
    vis_img = cv2.addWeighted(overlay, 0.2, vis_img, 0.8, 0)

    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"[Saved] 可视化图像保存至: {save_path}")

    return final_boxes, filtered_out_boxes


def process_image(image_path, output_image_path, output_txt_path, **kwargs):
    """
    图像路径接口函数：读取图像，调用 process_image_from_array，保存可视化和 box 坐标
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load: {image_path}")

    image_name = os.path.basename(image_path)
    boxes, _ = process_image_from_array(image, image_name, **kwargs)

    # 可视化绘图
    vis_img = image.copy()
    overlay = vis_img.copy()
    for box in boxes:
        x1, y1 = box[0]
        x3, y3 = box[2]
        cv2.rectangle(overlay, (x1, y1), (x3, y3), (0, 255, 0), -1)
        cv2.rectangle(vis_img, (x1, y1), (x3, y3), (0, 255, 0), 2)
    vis_img = cv2.addWeighted(overlay, 0.2, vis_img, 0.8, 0)

    # 保存输出
    cv2.imwrite(output_image_path, vis_img)
    with open(output_txt_path, 'w') as f:
        for points in boxes:
            f.write(",".join([f"{x},{y}" for (x, y) in points]) + "\n")
    print(f"Saved {output_image_path} to {output_txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch processing for extracting map legend regions and detecting item boxes")
    parser.add_argument('--input_dir', type=str, help="Path to the input root directory containing numbered subfolders")
    parser.add_argument('--output_dir', type=str, help="Path to the output directory where results will be saved")
    parser.add_argument('--global_area_min_factor', type=float, default=0.0001,
                        help="Minimum ratio of item box area to full image area when no legend is detected (default: 0.0001)")
    parser.add_argument('--global_area_max_factor', type=float, default=0.04,
                        help="Maximum ratio of item box area to full image area when no legend is detected (default: 0.01)")
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
    parser.add_argument('--color_tolerance', type=float, default=40,
                        help="Tolerance for average border color difference (default: 25). ")
    parser.add_argument('--duplicate_filter_size_tolerance', type=int, default=5,
                        help='Size tolerance (in pixels) for grouping boxes by dimensions (default: 5)')
    parser.add_argument('--duplicate_filter_color_tolerance', type=int, default=2,
                        help='Color rounding tolerance for grouping boxes by color (default: 2)')
    parser.add_argument('--duplicate_filter_shrink_pixels', type=int, default=5,
                        help='Number of pixels to shrink inward from each box edge before color analysis (default: 4)')
    parser.add_argument('--duplicate_filter_color_std_max_threshold', type=float, default=10,
                        help='Maximum allowed color standard deviation to consider a box as pure-color (default: 10)')
    parser.add_argument('--debug', action='store_true',
                        help="If set, save intermediate result.")


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.input_dir) and args.input_dir.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        print(f"[Single] Detected image file: {args.input_dir}")
        image = cv2.imread(args.input_dir)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {args.input_dir}")
        name = os.path.splitext(os.path.basename(args.input_dir))[0]
        save_path = os.path.join(args.output_dir, f"{name}_vis.png")
        txt_path = os.path.join(args.output_dir, f"{name}.txt")

        final_boxes, _ = extract_legend_box_from_path(args.input_dir, save_path=save_path)

        with open(txt_path, 'w') as f:
            for box in final_boxes:
                line = ",".join([f"{x},{y}" for (x, y) in box])
                f.write(line + "\n")
        print(f"[Saved] Box coords written to: {txt_path}")
    else:
        flag = True
        if flag:
            subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

            default_bg_color = None
            #default_bg_color = (255, 255, 255)

            for subdir in subdirs:
                image_folder = os.path.join(args.input_dir, subdir, 'image')
                tif_files = glob.glob(os.path.join(image_folder, '*.tif'))
                if not tif_files:
                    print(f"跳过 {subdir}，没有.tif文件")
                    continue
                image_path = tif_files[0]

                output_image_path = os.path.join(args.output_dir, f"{subdir}.png")
                output_txt_path = os.path.join(args.output_dir, f"{subdir}.txt")

                process_image(image_path, output_image_path, output_txt_path,
                    global_area_min_factor=args.global_area_min_factor,
                    global_area_max_factor=args.global_area_max_factor,
                    cluster_eps_scale=args.cluster_eps_scale, cluster_min_samples=args.cluster_min_samples,
                    cluster_recover_size_tolerance=args.cluster_recover_size_tolerance, default_bg_color=default_bg_color,
                    color_test_initial_expand=args.color_test_initial_expand, color_test_border_thickness=args.color_test_border_thickness,
                    color_tolerance=args.color_tolerance, duplicate_filter_size_tolerance=args.duplicate_filter_size_tolerance,
                    duplicate_filter_color_tolerance=args.duplicate_filter_color_tolerance, duplicate_filter_shrink_pixels=args.duplicate_filter_shrink_pixels,
                    duplicate_filter_color_std_max_threshold=args.duplicate_filter_color_std_max_threshold,
                    debug=args.debug)

        else:
            # 获取输入目录下的所有.png文件
            image_files = glob.glob(os.path.join(args.input_dir, '*.png'))

            default_bg_color = None  # or set to (255, 255, 255)

            for image_path in image_files:
                # 获取文件名（不含扩展名），作为输出文件名的前缀
                filename = os.path.splitext(os.path.basename(image_path))[0]

                output_image_path = os.path.join(args.output_dir, f"{filename}.png")
                output_txt_path = os.path.join(args.output_dir, f"{filename}.txt")

                process_image(
                    image_path, output_image_path, output_txt_path,
                    global_area_min_factor=args.global_area_min_factor,
                    global_area_max_factor=args.global_area_max_factor,
                    cluster_eps_scale=args.cluster_eps_scale,
                    cluster_min_samples=args.cluster_min_samples,
                    cluster_recover_size_tolerance=args.cluster_recover_size_tolerance,
                    default_bg_color=default_bg_color,
                    color_test_initial_expand=args.color_test_initial_expand,
                    color_test_border_thickness=args.color_test_border_thickness,
                    color_tolerance=args.color_tolerance,
                    debug=args.debug)


if __name__ == "__main__":
    main()
