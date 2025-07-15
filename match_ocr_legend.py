import os
import argparse
import cv2
import numpy as np
from io import BytesIO
from paddleocr import PaddleOCR
from PIL import ImageFont, ImageDraw, Image
from paddleocr import TextRecognition, TextDetection

# 初始化 PaddleOCR 文本识别模型
text_rec_model = TextRecognition(model_name="PP-OCRv5_server_rec")
text_det_model = TextDetection(model_name="PP-OCRv5_server_det")


def draw_text_cn(cv2_img, text, pos, font_size=20, font_path="fonts/simhei.ttf", color=(0, 0, 0), bg_color=(255, 255, 255), draw_bg=True):
    img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[WARNING] Failed to load font: {font_path}. Error: {e}")
        font = ImageFont.load_default()

    bbox = draw.textbbox(pos, text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if draw_bg:
        padding = 2
        x, y = pos
        draw.rectangle([x - padding, y - padding, x + text_w + padding, y + text_h + padding], fill=bg_color)

    draw.text(pos, text, font=font, fill=color[::-1])
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

    print(f"[\u2713] 输出文本保存至: {output_txt_path}")


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
        split_ocr_boxes = raw_ocr_boxes  # 此处简化，可按需替换为拆分函数

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
                legend_results_ori.append({'box': [x1, y1, x2, y2]})

        filtered_ocr_boxes = split_ocr_boxes  # 此处简化
        matched_legends = []
        for i, lgd in enumerate(legend_results_ori):
            matched_indices = list(range(len(filtered_ocr_boxes)))  # 简化，全部匹配
            matched_legends.append({'box': lgd['box'], 'matched_indices': matched_indices})

        matched_indices = set()
        for lgd in matched_legends:
            matched_indices.update(lgd['matched_indices'])

        recognized_texts = {}
        for idx in matched_indices:
            recognized_texts[idx] = (f"text{idx}", 0.99)  # mock text

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            out_path = os.path.join(output_folder, f"{filename}_matched.jpg")
            draw_and_save(image, out_path)

            # ✅ 保存提取文本输出
            text_output_dir = os.path.join(output_folder, "output_text")
            save_text_results(text_output_dir, filename, matched_legends, filtered_ocr_boxes, recognized_texts)


def main(args):
    print("Using PaddleOCR for OCR box + text detection...")

    def paddleocr_detector(image):
        results = text_det_model.predict(image, batch_size=1)
        if not results or not isinstance(results[0], dict):
            return []
        polys = results[0].get("dt_polys", [])
        boxes = [[[int(pt[0]), int(pt[1])] for pt in poly] for poly in polys]
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
