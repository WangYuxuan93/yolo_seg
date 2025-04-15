import os
import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO segmentation results with class name labels.")
    parser.add_argument('--input_root', type=str, required=True, help="Root directory with numbered subfolders")
    return parser.parse_args()

def read_yolo_segmentation_file_to_pixel_coords(txt_path, image_width, image_height):
    objects = []

    if not os.path.exists(txt_path):
        return []

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) % 2 != 0:
                print(f"Warning: Invalid line in {txt_path}: {line}")
                continue

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

def visualize_on_image(image_path, label_path, class_colors, class_names):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    h, w = img.shape[:2]
    objects = read_yolo_segmentation_file_to_pixel_coords(label_path, w, h)

    for obj in objects:
        points = np.array(obj['points'], dtype=np.int32).reshape((-1, 1, 2))
        class_id = obj['class_id']
        class_name = class_names.get(class_id, f"class_{class_id}")
        color = class_colors.get(class_name, (0, 255, 0))  # default: green

        # 半透明填充
        overlay = img.copy()
        cv2.drawContours(overlay, [points], -1, color, -1)
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)

        # 轮廓边界
        cv2.drawContours(img, [points], -1, color, 2)

        # 类别标签（左上角）
        x_text, y_text = points[0][0][0], points[0][0][1]
        (w_text, h_text), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x_text, y_text - h_text), (x_text + w_text, y_text), color, -1)
        cv2.putText(img, class_name, (x_text, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Segmentation Preview", img)
    key = cv2.waitKey(0)
    if key == ord('q') or key == 27:
        return False
    return True

def main():
    args = parse_args()

    # 类别映射：class_id → class_name
    class_names = {
        0: 'main map',
        1: 'legend',
        2: 'item',
        3: 'compass',
        4: 'scale',
        5: 'title'
        # 如果有更多类别，在这里加
    }

    # 类别颜色：class_name → BGR 颜色
    class_colors = {
        'main map': (255, 0, 0),   # 红色
        'legend': (0, 0, 255),     # 蓝色
        'item': (0, 255, 0),
        'compass': (0, 255, 255),
        'scale': (255, 255, 255),
        'title': (255, 255, 0)
        # 其他类别颜色也可加
    }

    folder_path = args.input_root
    image_dir = os.path.join(folder_path, "image")
    layout_dir = os.path.join(folder_path, "layout")

    for filename in sorted(os.listdir(image_dir)):
        print (filename)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue

        image_path = os.path.join(image_dir, filename)
        txt_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(layout_dir, txt_name)

        print(f"Previewing: {image_path}")
        continue_display = visualize_on_image(image_path, label_path, class_colors, class_names)
        if not continue_display:
            print("Stopped.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()