from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

filename = "00010.jpg"
# 加载YOLO模型
#model = YOLO("yolo11n-seg.pt")  # 你可以替换为你的自定义模型路径
model = YOLO("seg_train_layout_v0/train4/weights/best.pt")  # 加载自定义模型

# 对图片进行预测
results = model("../layout/images/train/" + filename)  # 使用图像URL进行预测

# 获取预测结果
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    #print ("xy:{}".format(xy[0].shape))
    #print ("xyn:{}".format(xyn[0].shape))
    print ("masks:{}".format(masks.shape))
    class_ids = result.boxes.cls  # 获取每个物体的类别 ID
    names = result.names  # 获取类别名称
    print ("names:{}".format(names))

# 读取图片
img = cv2.imread("bus.jpg")

# 定义颜色：为每个类别分配不同的颜色
colors = {
    'main map': [255, 0, 0],
    'legend': [0, 255, 0],
    'stop sign': [0, 0, 255],  # 蓝色
    # 你可以根据需要添加其他类别及颜色
}

# 存储所有检测到的类别
detected_classes = set()

# 可视化掩码
for result in results:
    for i, mask in enumerate(result.masks.data):
        # 获取每个掩码的类别名称
        class_id = int(result.boxes.cls[i])  # 获取当前掩码的类别 ID
        class_name = names[class_id]  # 获取类别名称
        
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

        # 添加当前类别到检测到的类别集合中
        detected_classes.add(class_name)

# 绘制类别名称和底色
y_offset = 30  # 初始位置，用于放置类别名称
for class_name in detected_classes:
    color = colors.get(class_name, [255, 255, 255])  # 获取该类别对应的颜色
    text_color = (255, 255, 255)  # 白色文本，确保文本清晰
    bg_color = color  # 使用类别对应的颜色作为背景色

    # 获取文本大小
    (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)  # 获取文本大小
    # 绘制背景矩形
    cv2.rectangle(img, (img.shape[1] - w - 10, y_offset - 10), (img.shape[1] - 10, y_offset + h + 10), bg_color, -1)  # 颜色背景
    # 在矩形内绘制类别名称
    cv2.putText(img, class_name, (img.shape[1] - w - 5, y_offset + h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)  # 白色文本
    y_offset += h + 20  # 调整位置以放置下一个类别



# 保存处理后的图像到文件
output_path = "masked_"+filename
cv2.imwrite(output_path, img)  # 保存图像到指定路径

# 如果你想确认保存了图片，可以输出提示信息
print(f"Image saved as {output_path}")