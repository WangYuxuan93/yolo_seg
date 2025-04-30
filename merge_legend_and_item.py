import os
import cv2
import numpy as np

def draw_boxes_on_image(image_path, txt_path, output_path, smooth_contours=False, epsilon_factor=0.02):
    # 读取图片
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]
    
    # 创建一个与原图相同大小的蒙版（黑色背景）
    green_overlay = np.zeros_like(image, dtype=np.uint8)

    # 读取txt文件并解析
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # 解析YOLO格式的框信息
    for line in lines:
        parts = line.strip().split()
        label = "legend" if int(parts[0]) == 0 else "main map"  # 根据数字标签选择文字标签
        points = list(map(float, parts[1:]))  # 获取多边形的所有点
        
        # 将归一化的坐标转换为像素坐标
        points = np.array(points).reshape(-1, 2)
        points[:, 0] *= img_width  # x坐标乘以图像宽度
        points[:, 1] *= img_height  # y坐标乘以图像高度
        
        # 如果启用平滑功能
        if smooth_contours:
            points = points.astype(np.float32)  # 转换为 float32
            hull = cv2.convexHull(points)  # 计算凸包
            arc_length = cv2.arcLength(hull, closed=True)  # 计算弧长
            epsilon = epsilon_factor * arc_length  # epsilon控制平滑程度
            approx = cv2.approxPolyDP(hull, epsilon=epsilon, closed=True)  # 获取平滑后的多边形
            points = np.array([pt[0] for pt in approx], dtype=np.int32)  # 扁平化并转换为整数坐标
            polygon_draw = points  # 保存平滑后的多边形
        
        else:
            polygon_draw = points  # 如果不平滑，则直接使用原始点
        
        # 计算左上角和右上角
        left_top = polygon_draw[np.argmin(polygon_draw[:, 0] + polygon_draw[:, 1])]  # x最小且y最小的点为左上角
        right_top = polygon_draw[np.argmax(polygon_draw[:, 0] - polygon_draw[:, 1])]  # x最大且y最小的点为右上角

        # 检查点的数量是否大于等于3（至少有三个点才能形成多边形）
        if len(polygon_draw) >= 3:
            # 设置颜色（根据标签区分不同颜色）
            color = (0, 255, 0) if label == "main map" else (255, 0, 0)  # 绿色为main map，蓝色为legend
            
            # 绘制多边形框
            cv2.polylines(image, [np.int32(polygon_draw)], isClosed=True, color=color, thickness=2)
            
            # 文字底纹
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            padding = 5  # 底纹的内边距
            
            # 如果是主图（main map），叠加透明绿色
            if label == "main map":
                # 创建半透明绿色叠加（Alpha=0.3表示30%的透明度）
                cv2.fillPoly(green_overlay, [np.int32(polygon_draw)], (0, 255, 0))  # 填充绿色

                # 使用掩膜蒙版叠加透明绿色
                alpha = 0.3
                cv2.addWeighted(green_overlay, alpha, image, 1, 0, image)  # 直接叠加到原图
            
                
                text_x = int(left_top[0])  # 使用左上角的x坐标
                text_y = int(left_top[1] + 10)  # 使用左上角的y坐标，并略微偏移
                
                # 在文字上方绘制白色底纹
                cv2.rectangle(image, (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (255, 255, 255), -1)  # 白色底纹
                
                # 绘制文字
                cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness)
            
            # 在框的右上角写上legend标签
            if label == "legend":
                text_x = int(right_top[0]) - text_size[0]  # 使用右上角的x坐标，并考虑文字宽度
                text_y = int(right_top[1]) - text_size[1]  # 使用右上角的y坐标，略微偏移
                
                # 绘制文字底纹
                cv2.rectangle(image, (text_x - padding, text_y - text_size[1] - padding),
                              (text_x + text_size[0] + padding, text_y + padding),
                              (255, 255, 255), -1)  # 白色底纹
                
                # 绘制文字
                cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness)
        
        else:
            print(f"Warning: Invalid points found in {txt_path}. Skipping this line.")
    
    # 保存图片
    cv2.imwrite(output_path, image)

def process_directories(item_dir, legend_dir, output_dir, smooth_contours=True, epsilon_factor=0.02):
    # 获取item图像列表
    item_images = [f for f in os.listdir(item_dir) if f.endswith('.jpg')]
    
    for item_image in item_images:
        # 获取对应的txt文件名，去掉 "masked_" 前缀
        txt_file = item_image.replace('.jpg', '.txt').replace('masked_', '')  # 去掉 masked_ 前缀
        legend_txt_path = os.path.join(legend_dir, txt_file)  # 拼接legend路径
        #print(legend_txt_path)
        
        # 检查txt文件是否存在
        if os.path.exists(legend_txt_path):
            item_image_path = os.path.join(item_dir, item_image)
            output_image_path = os.path.join(output_dir, item_image)
            
            # 在图像上绘制框，并启用平滑功能
            draw_boxes_on_image(item_image_path, legend_txt_path, output_image_path, smooth_contours=smooth_contours, epsilon_factor=epsilon_factor)
            print(f"Processed: {item_image}")
        else:
            print(f"Warning: {txt_file} not found in legend directory")

# 输入文件夹路径
item_dir = 'data/0326_real_map/0326-refine-expand5-ratio05/set1'  # 请替换为实际的路径
legend_dir = 'data/legend/0326-v2/set1'  # 请替换为实际的路径
output_dir = 'data/legend-item/0326/set1'  # 请替换为实际的路径

os.makedirs(output_dir, exist_ok=True)

# 调用函数
process_directories(item_dir, legend_dir, output_dir, smooth_contours=True, epsilon_factor=0.02)
