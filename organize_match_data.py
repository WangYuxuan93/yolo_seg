import os
import json
import cv2
import shutil
from PIL import Image, ImageDraw, ImageFont

def draw_legend_boxes(root_dir, output_dir="output", font_path="simhei.ttf"):
    visual_dir = os.path.join(output_dir, "visual")
    gold_dir = os.path.join(output_dir, "gold")
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    # 加载中文字体（确保有 simhei.ttf 或替换为你的中文字体）
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        raise RuntimeError(f"无法加载字体：{font_path}，请确保该ttf文件存在")

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        meta_path = os.path.join(folder_path, "meta.json")
        image_path = os.path.join(folder_path, "labeled", "main.tif")

        if not os.path.exists(meta_path) or not os.path.exists(image_path):
            print(f"缺失文件，跳过: {folder_name}")
            continue

        # 读取meta.json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 读取主图（OpenCV用于画框，Pillow用于写字）
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 拷贝原始图到 gold 目录
        shutil.copyfile(image_path, os.path.join(gold_dir, f"{folder_name}.tif"))

        # 写 gold 的文本文件
        txt_path = os.path.join(gold_dir, f"{folder_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as txt_file:

            legend_items = meta.get("legend_items", {})
            for label, info in legend_items.items():
                box = info.get("bounding_box", {})
                x1, y1 = box.get("x1"), box.get("y1")
                x2, y2 = box.get("x2"), box.get("y2")
                if None in [x1, y1, x2, y2]:
                    continue

                # 写文本文件
                txt_file.write(f"{label} {x1} {y1} {x2} {y2}\n")

                # 画框（OpenCV）
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 使用Pillow写中文
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        for label, info in legend_items.items():
            box = info.get("bounding_box", {})
            x1, y1 = box.get("x1"), box.get("y1")
            if None in [x1, y1]:
                continue

            # 字体位置
            text_pos = (x1, y1 - 22 if y1 > 22 else y1 + 5)
            draw.text(text_pos, label, fill=(255, 0, 0), font=font)

        # 保存可视化图像
        final_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        vis_path = os.path.join(visual_dir, f"{folder_name}_legend.jpg")
        cv2.imwrite(vis_path, final_image)
        print(f"完成处理: {folder_name}")

if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="在主图上画出legend box并标注中文")
    parser.add_argument("input_dir", help="包含多个子文件夹的输入根目录")
    parser.add_argument("--output_dir", default="output", help="保存结果的输出目录")
    parser.add_argument("--font_path", default="fonts/simhei.ttf", help="中文字体路径，如 simhei.ttf 或其他 .ttf")
    args = parser.parse_args()

    draw_legend_boxes(args.input_dir, args.output_dir, args.font_path)
