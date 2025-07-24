import os
import argparse
import cv2
import numpy as np
from io import BytesIO
from paddleocr import PaddleOCR

from PIL import ImageFont, ImageDraw, Image

from paddleocr import TextRecognition, TextDetection

# 初始化 PaddleOCR 文本识别模型（v5新API）
text_rec_model = TextRecognition(model_name="PP-OCRv5_server_rec")

text_det_model = TextDetection(model_name="PP-OCRv5_server_det")

def match_legends_with_ocr(legend_results_ori, ocr_boxes):
    """
    根据 OCR 文字的结果，匹配每个图例框与文字，并返回包含 matched_indices 的原格式。

    Args:
        legend_results_ori (list of dict): 每个字典包含 'box' 键，值为 [x1, y1, x2, y2]
        ocr_boxes (list of tuple): 每个元素为 (x1, y1, x2, y2, text)

    Returns:
        list of dict: 每个字典新增 'matched_indices' 和 'text'
    """
    matched_legends = []

    for lgd in legend_results_ori:
        lx, ly, rx, ry = lgd['box']
        lw = rx - lx
        lh = ry - ly

        matched_indices = []
        inter_boxes = []

        for idx, (x1_o, y1_o, x2_o, y2_o, text) in enumerate(ocr_boxes):
            ox = (x1_o + x2_o) * 0.5
            oy = (y1_o + y2_o) * 0.5
            if lx + 0.5 * lw < x1_o < lx + 2 * lw and ly - 0.2 * lh < oy < ly + 1.2 * lh:
                inter_boxes.append((y1_o, text, idx))  # 保留索引

        if inter_boxes:
            inter_boxes.sort()  # 按 y1 排序
            matched_indices = [idx for _, _, idx in inter_boxes]
            text_merged = ' '.join([text for _, text, _ in inter_boxes])
        else:
            matched_indices = []
            text_merged = ''

        matched_legends.append({
            'box': lgd['box'],
            'matched_indices': matched_indices,
            'text': text_merged
        })

    return matched_legends


def draw_text_cn(cv2_img, text, pos, font_size=20, font_path="fonts/simhei.ttf", color=(0, 0, 0), bg_color=(255, 255, 255), draw_bg=True):
    img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[WARNING] Failed to load font: {font_path}. Error: {e}")
        font = ImageFont.load_default()

    # 文字尺寸
    bbox = draw.textbbox(pos, text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if draw_bg:
        # 背景框（带些 padding）
        padding = 2
        x, y = pos
        draw.rectangle([x - padding, y - padding, x + text_w + padding, y + text_h + padding], fill=bg_color)

    draw.text(pos, text, font=font, fill=color[::-1])  # BGR -> RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def save_text_results(output_dir, filename_no_ext, matched_legends, ocr_boxes, recognized_texts):
    os.makedirs(output_dir, exist_ok=True)
    output_txt_path = os.path.join(output_dir, f"{filename_no_ext}.txt")
    lines = []

    for lgd in matched_legends:
        collected_texts = []
        for idx in lgd['matched_indices']:
            text, score = recognized_texts.get(idx, ('', 0.0))
            if text.strip():
                collected_texts.append(text.strip())
        if not collected_texts:
            continue
        merged_text = ' '.join(collected_texts).strip()
        x1, y1, x2, y2 = lgd['box']
        lines.append(f"{merged_text} {x1} {y1} {x2} {y2}")

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[✓] 输出文本保存至: {output_txt_path}")


def filter_legends_with_ocr(legend_results_ori, ocr_boxes):
    matched_legends = []

    # ✅ OCR boxes 按 x 坐标升序排列，确保匹配顺序合理
    ocr_boxes_sorted = sorted(enumerate(ocr_boxes), key=lambda item: min(p[0] for p in item[1][:4]))

    raw_matches = []

    for lgd in legend_results_ori:
        lx1, ly1, lx2, ly2 = lgd['box']
        lw = lx2 - lx1
        lh = ly2 - ly1
        matched_indices = []

        # ✅ 找右侧 legend 邻居（最近且有垂直重叠）
        min_dx = float('inf')
        right_neighbor_x = None
        for other in legend_results_ori:
            if other is lgd:
                continue
            ox1, oy1, ox2, oy2 = other['box']
            if ox1 > lx2 and not (ly2 <= oy1 or ly1 >= oy2):
                dx = ox1 - lx2
                if dx < min_dx:
                    min_dx = dx
                    right_neighbor_x = ox1

        # ✅ 设置匹配横向范围
        primary_max_x = lx1 + 2.0 * lw
        extended_max_x = right_neighbor_x if right_neighbor_x is not None else lx1 + 10.0 * lw
        max_gap = 1.0 * lw
        current_right_bound = None

        # ✅ Step 1: 右侧主 + 扩展匹配
        for idx, occ in ocr_boxes_sorted:
            quad = occ[:4]
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            oy = (y1 + y2) * 0.5

            # 主区间直接匹配
            if lx1 + lw <= x1 <= primary_max_x and ly1 - 0.3 * lh < oy < ly2 + 0.3 * lh:
                matched_indices.append(idx)
                current_right_bound = x2 if current_right_bound is None else max(current_right_bound, x2)
                continue

            # 扩展区间匹配需要接近已有边界
            if primary_max_x < x1 <= extended_max_x and ly1 - 0.3 * lh < oy < ly2 + 0.3 * lh:
                if current_right_bound is not None and x1 - current_right_bound < max_gap:
                    matched_indices.append(idx)
                    current_right_bound = max(current_right_bound, x2)

        # ✅ Step 2: 若右侧无匹配，再尝试下方匹配
        if not matched_indices:
            for idx, occ in enumerate(ocr_boxes):
                quad = occ[:4]
                xs = [pt[0] for pt in quad]
                ys = [pt[1] for pt in quad]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                ox = (x1 + x2) * 0.5
                oy = (y1 + y2) * 0.5

                is_bottom = lx1 - 0.2 * lw < ox < lx2 + 0.2 * lw and ly2 <= oy <= ly2 + 1.5 * lh
                if is_bottom:
                    matched_indices.append(idx)

        if matched_indices:
            raw_matches.append({
                'box': [lx1, ly1, lx2, ly2],
                'matched_indices': matched_indices
            })

    # ✅ 后处理：每个 OCR 只能匹配一个 legend，保留距离最近的匹配
    ocr_to_best = {}
    for i, entry in enumerate(raw_matches):
        lx1, ly1, lx2, ly2 = entry['box']
        for idx in entry['matched_indices']:
            quad = ocr_boxes[idx][:4]
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            # ✅ 使用 OCR box 左边中点 和 legend box 中心点 的欧氏距离
            ocr_left_mid = ((x1 + x1) / 2, (y1 + y2) / 2)
            legend_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)
            dist = ((ocr_left_mid[0] - legend_center[0]) ** 2 + (ocr_left_mid[1] - legend_center[1]) ** 2) ** 0.5

            if idx not in ocr_to_best or dist < ocr_to_best[idx][1]:
                ocr_to_best[idx] = (i, dist)

    # ✅ 构造最终唯一匹配的结果
    legend_idx_to_ocr_indices = {}
    for idx, (legend_idx, _) in ocr_to_best.items():
        legend_idx_to_ocr_indices.setdefault(legend_idx, []).append(idx)

    for i, entry in enumerate(raw_matches):
        indices = legend_idx_to_ocr_indices.get(i, [])
        if indices:
            matched_legends.append({
                'box': entry['box'],
                'matched_indices': indices
            })

    return matched_legends


def sort_indices_by_reading_order(indices, ocr_boxes, line_tolerance_ratio=0.6):
    if not indices:
        return []

    # 提取中心点和高度信息
    centers = []
    heights = []
    for idx in indices:
        box = ocr_boxes[idx][:4]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h = y2 - y1
        centers.append((idx, cx, cy))
        heights.append(h)

    avg_height = np.median(heights)
    line_thresh = avg_height * line_tolerance_ratio

    # 按 y 坐标分行
    lines = []
    for idx, cx, cy in sorted(centers, key=lambda x: x[2]):
        placed = False
        for line in lines:
            if abs(cy - line[0][2]) < line_thresh:
                line.append((idx, cx, cy))
                placed = True
                break
        if not placed:
            lines.append([(idx, cx, cy)])

    # 每行内部按 x 排序，并拼接结果
    sorted_indices = []
    for line in lines:
        line_sorted = sorted(line, key=lambda x: x[1])  # 按 cx 排
        sorted_indices.extend([idx for idx, _, _ in line_sorted])

    return sorted_indices

def visualize_matches(image, legend_results_ori, matched_legends, ocr_boxes, recognized_texts, save_subdir=None):
    import os
    overlay = image.copy()
    raw_image = image.copy()  # ✅ 用于截图，避免画框干扰
    alpha = 0.4
    matched_ocr_set = set()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (0, 0, 0)  # 黑色文字
    bg_color = (200, 255, 200)  # 浅绿色背景框

    # ✅ 若提供保存子目录，则创建
    if save_subdir:
        os.makedirs(save_subdir, exist_ok=True)
        box_save_counter = 0  # 顺序编号截图

    # Draw all legend boxes (green)
    for lgd in legend_results_ori:
        x1, y1, x2, y2 = lgd['box']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Draw matched OCR boxes (red) and connect lines
    for lgd in matched_legends:
        lx1, ly1, lx2, ly2 = lgd['box']
        legend_center = ((lx1 + lx2) // 2, (ly1 + ly2) // 2)

        collected_texts = []

        for idx in lgd['matched_indices']:
            matched_ocr_set.add(idx)
            quad = ocr_boxes[idx][:4]
            text, score = recognized_texts.get(idx, ('', 0.0))
            if text.strip():
                collected_texts.append(text.strip())

            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            ox, oy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 180), 2)
            cv2.line(image, legend_center, (ox, oy), (0, 0, 200), 2)

            # ✅ 保存截图 + 添加上方文字（使用 raw_image）
            if save_subdir:
                cropped = crop_quad(raw_image, quad)
                h, w = cropped.shape[:2]
                space = 25  # 留出空白区域高度
                canvas = np.ones((h + space, w, 3), dtype=np.uint8) * 255  # 白底
                canvas[space:, :] = cropped
                label = text.strip()
                #cv2.putText(canvas, label, (2, space - 5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                #canvas = draw_text_cn(canvas, label, (2, 2), font_size=18, font_path='fonts/simhei.ttf', color=(0, 0, 0))
                canvas = draw_text_cn(
                    canvas, label, (2, 2),
                    font_size=14,
                    font_path='fonts/simhei.ttf',
                    color=(0, 0, 0),
                    bg_color=(255, 255, 255),
                    draw_bg=True
                )
                
                save_path = os.path.join(save_subdir, f"{box_save_counter:03d}.jpg")
                cv2.imwrite(save_path, canvas)
                box_save_counter += 1

        # 显示合并后的识别文本
        if collected_texts:
            text_to_show = ' '.join(collected_texts)
            tx, ty = lx1, ly1 - 5
            #image = draw_text_cn(image, text_to_show, (tx, ty - 20), font_size=20, font_path='fonts/simhei.ttf', color=(0, 0, 0))
            image = draw_text_cn(
                        image, text_to_show, (tx, ty - 20),
                        font_size=14,
                        font_path='fonts/simhei.ttf',
                        color=(0, 0, 0),
                        bg_color=(255, 255, 200),
                        draw_bg=True
                    )

    # Draw unmatched OCR boxes (blue)
    for idx, occ in enumerate(ocr_boxes):
        if idx in matched_ocr_set:
            continue
        quad = occ[:4]
        xs = [pt[0] for pt in quad]
        ys = [pt[1] for pt in quad]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    combined = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return combined


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_quad(image, quad):
    quad_np = np.array(quad, dtype=np.float32).reshape((4, 2))
    rect = order_points(quad_np)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def recognize_text_from_indices(image, ocr_boxes, indices, batch_size=32):
    result_dict = {}

    crops = []
    idx_list = []

    for idx in indices:
        box = ocr_boxes[idx]
        quad = box[:4]
        cropped = crop_quad(image, quad)
        crops.append(cropped)
        idx_list.append(idx)

    # ⏩ 批量识别，显式传入 batch_size
    batch_results = text_rec_model.predict(crops, batch_size=batch_size)

    for i, result in enumerate(batch_results):
        idx = idx_list[i]
        if result and isinstance(result, dict):
            rec_text = result.get('rec_text', '')
            rec_score = result.get('rec_score', 0.0)
        else:
            rec_text, rec_score = '', 0.0
        result_dict[idx] = (rec_text, rec_score)

    return result_dict

def load_image_as_opencv_matrix(local_filepath):
    with open(local_filepath, 'rb') as f:
        file_bytes = f.read()
    image_cache = BytesIO(file_bytes)
    image_data = np.asarray(bytearray(image_cache.read()), dtype=np.uint8)
    return cv2.imdecode(image_data, cv2.IMREAD_COLOR)

def draw_and_save(image, save_path):
    cv2.imwrite(save_path, image)
    print(f"Saved visualization to {save_path}")


def process_folder(image_folder, predictor_func, output_folder, label, save_ocr=True):
    image_list = sorted([
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
    ])
    print(f"\n[{label}] Found {len(image_list)} images in {image_folder}")

    for image_path in image_list:
        image = load_image_as_opencv_matrix(image_path)
        filename = os.path.splitext(os.path.basename(image_path))[0]

        raw_ocr_boxes = predictor_func(image)

        legend_results_ori = []
        txt_path = os.path.join(image_folder, filename + ".txt")
        if not os.path.isfile(txt_path):
            print(f"[WARNING] No legend box file found for {filename}")
            continue
        with open(txt_path, 'r') as f:
            for line in f:
                coords = list(map(int, line.strip().split(',')))
                if len(coords) != 8:
                    continue
                xs = coords[::2]
                ys = coords[1::2]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                legend_results_ori.append({
                    'box': [x1, y1, x2, y2],
                    'bgr': [0, 0, 0],
                    'mask': None,
                    'color': '',
                    'polygons': [[[]]],
                    'mappingArea': ''
                })

        filtered_ocr_boxes = raw_ocr_boxes

        # Step 1: 识别全部 OCR box 的文字
        all_indices = list(range(len(filtered_ocr_boxes)))
        recognized_texts = recognize_text_from_indices(image, filtered_ocr_boxes, all_indices)

        # Step 2: 构造 OCR box + 文本的结构供匹配使用
        ocr_boxes_with_text = []
        for idx, box in enumerate(filtered_ocr_boxes):
            quad = box[:4]
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            text, _ = recognized_texts.get(idx, ('', 0.0))
            ocr_boxes_with_text.append((x1, y1, x2, y2, text))

        # Step 3: 使用你自定义的匹配函数（返回的 matched_legends 是原格式）
        matched_legends = match_legends_with_ocr(legend_results_ori, ocr_boxes_with_text)

        # Step 4: 排序（可选）
        for lgd in matched_legends:
            lgd['matched_indices'] = sort_indices_by_reading_order(lgd['matched_indices'], filtered_ocr_boxes)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            out_path = os.path.join(output_folder, f"{filename}_matched.jpg")
            ocr_subfolder = os.path.join(output_folder, f"{filename}_ocr")
            vis = visualize_matches(
                image.copy(),
                legend_results_ori,
                matched_legends,
                filtered_ocr_boxes,
                recognized_texts,
                save_subdir=ocr_subfolder if save_ocr else None
            )
            draw_and_save(vis, out_path)

            # ✅ 保存提取文本输出
            text_output_dir = os.path.join(output_folder, "output_text")
            save_text_results(text_output_dir, filename, matched_legends, filtered_ocr_boxes, recognized_texts)


def main(args):
    print("cuda:", args.cuda)
    print("Using PaddleOCR for OCR box + text detection...")
    """
    def paddleocr_detector(image):
        results = ocr_model.ocr(image)
        
        # results 是一个列表（可能是多页），我们只处理第一页
        if not results or not isinstance(results[0], dict):
            return [], {}

        res_dict = results[0]
        polys = res_dict['dt_polys']           # (N, 4, 2)
        #texts = res_dict['rec_texts']
        #scores = res_dict['rec_scores']

        boxes = []
        #recognized_texts = {}

        for idx, poly in enumerate(polys):
            # 转换为整数格式 + dummy text
            quad = [[int(pt[0]), int(pt[1])] for pt in poly]
            #quad.append(text)  # 第五个元素是文本，用于兼容旧格式
            boxes.append(quad)
            #recognized_texts[idx] = (text, float(score))

        return boxes#, recognized_texts
    """
    
    def paddleocr_detector(image):
        results = text_det_model.predict(image, batch_size=1)

        if not results or not isinstance(results[0], dict):
            return []

        polys = results[0].get("dt_polys", [])

        boxes = []
        for poly in polys:
            quad = [[int(pt[0]), int(pt[1])] for pt in poly]
            boxes.append(quad)

        return boxes

    predictor_func = paddleocr_detector

    process_folder(
        image_folder=args.image_folder,
        predictor_func=predictor_func,
        output_folder=args.output_dir,
        label='mainmap',
        save_ocr=args.save_ocr
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--save_ocr', action='store_true')
    args = parser.parse_args()
    main(args)