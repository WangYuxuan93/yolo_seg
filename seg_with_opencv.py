import cv2
import numpy as np
import argparse

def obtain_legend_rectangle_bbox(main_img):
    target_img = np.array(main_img)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

    # 二值化（自适应阈值效果更好）
    blur = cv2.GaussianBlur(target_img, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        
        # 计算轮廓近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 筛选矩形（4个顶点 + 面积/宽高比限制）
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 1.2 <= aspect_ratio <= 2: #and 10000 > w*h > 450:
                rectangles.append([x, y, x + w, y + h, 1.0])
    return rectangles

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="从地图图像中提取矩形框")
    parser.add_argument('image_path', type=str, help="输入地图图像文件路径")
    parser.add_argument('output_image_path', type=str, help="输出图像路径，用于保存标记后的图像")

    args = parser.parse_args()

    # 读取图像
    main_img = cv2.imread(args.image_path)

    # 调用函数获取矩形框
    rectangles = obtain_legend_rectangle_bbox(main_img)

    # 在原图上绘制矩形框
    for rect in rectangles:
        x1, y1, x2, y2, _ = rect
        cv2.rectangle(main_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形，线宽2

    # 保存图像
    cv2.imwrite(args.output_image_path, main_img)

    print(f"处理完成，结果已保存到 {args.output_image_path}")

if __name__ == "__main__":
    main()
