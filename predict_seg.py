import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Process images with YOLO model and visualize masks.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save processed images")
    return parser.parse_args()

# 加载YOLO模型
def load_model():
    return YOLO("outputs/layout-bs256-gpu8-v0/train2/weights/best.pt")  # 加载自定义模型

# 处理每个图片
def process_images(input_dir, output_dir, model):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中的所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    # 定义颜色：为每个类别分配不同的颜色
    colors = {
        'main map': [255, 0, 0],  # 红色
        'legend': [0, 0, 255],  # 蓝色
        # 你可以根据需要添加其他类别及颜色
    }

    # 处理每个图片
    for filename in image_files:
        # 对图片进行预测
        results = model(os.path.join(input_dir, filename))  # 使用图像路径进行预测

        # 读取图片
        img = cv2.imread(os.path.join(input_dir, filename))

        # 存储所有检测到的类别
        detected_classes = set()

        # 可视化掩码和添加边框
        for result in results:
            for i, mask in enumerate(result.masks.data):
                # 获取每个掩码的类别名称
                class_id = int(result.boxes.cls[i])  # 获取当前掩码的类别 ID
                class_name = result.names[class_id]  # 获取类别名称

                # 如果类别名没有对应颜色，随机分配一个颜色
                if class_name not in colors:
                    color = np.random.randint(0, 255, 3).tolist()  # 随机生成颜色
                else:
                    color = colors[class_name]

                # 将掩码调整为与原图相同的大小
                mask_resized = cv2.resize(mask.cpu().numpy(), (img.shape[1], img.shape[0]))  # 调整掩码大小与原图一致
                mask_resized = np.uint8(mask_resized * 255)  # 将掩码值归一化到 0-255 范围

                # 创建一个与原图大小相同的图像，初始化为全黑
                mask_overlay = np.zeros_like(img, dtype=np.uint8)

                # 填充掩码区域
                mask_overlay[mask_resized > 0] = color  # 将该区域填充为指定颜色

                # 将原图和掩码区域进行加权叠加，设置透明度
                img = cv2.addWeighted(img, 0.7, mask_overlay, 0.3, 0)

                # 绘制掩码的边框（边框颜色与类别一致）
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) > 1:
                        cv2.drawContours(img, [contour], -1, color, 2)  # 使用类别颜色绘制边框

                # 获取每个物体的边界框（左上角坐标和右下角坐标）
                x_min, y_min, x_max, y_max = result.boxes.xyxy[i].cpu().numpy()

                # 在每个物体的边界框内绘制类别名称并添加底色
                text_color = (255, 255, 255)  # 白色文本
                bg_color = color  # 使用类别颜色作为背景色

                # 获取文本大小
                (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                # 绘制背景矩形
                cv2.rectangle(img, (int(x_min), int(y_min) - h - 10), (int(x_min) + w, int(y_min)), bg_color, -1)
                # 在矩形内绘制类别名称
                cv2.putText(img, class_name, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

                # 将类别添加到检测到的类别集合中
                detected_classes.add(class_name)

        # 保存处理后的图像到文件
        output_path = os.path.join(output_dir, f"masked_{filename}")
        cv2.imwrite(output_path, img)  # 保存图像到指定路径

        # 如果你想确认保存了图片，可以输出提示信息
        print(f"Processed and saved image: {output_path}")

def main():
    # 解析命令行参数
    args = parse_args()

    # 加载YOLO模型
    model = load_model()

    # 处理图像
    process_images(args.input_dir, args.output_dir, model)

if __name__ == "__main__":
    main()
