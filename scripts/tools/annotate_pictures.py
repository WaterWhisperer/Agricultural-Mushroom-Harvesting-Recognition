import cv2
import numpy as np
import os
import hashlib

def generate_deterministic_color(class_name):
    """通过哈希算法生成确定性颜色"""
    hash_obj = hashlib.md5(class_name.encode())
    hash_num = int(hash_obj.hexdigest(), 16)
    
    r = (hash_num & 0xFF0000) >> 16
    g = (hash_num & 0x00FF00) >> 8
    b = hash_num & 0x0000FF
    
    return (b, g, r)  # OpenCV使用BGR格式

def draw_yolo_bounding_boxes(image_path, annotations_path, output_path, class_colors=None):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 获取图片的宽度和高度
    image_height, image_width, _ = image.shape

    # 读取标注文件
    with open(annotations_path, 'r') as file:
        annotations = file.readlines()

    # 绘制边界框
    for annotation in annotations:
        annotation = annotation.strip().split()
        if len(annotation) < 5:
            print(f"标注格式错误: {annotation}")
            continue

        class_index = int(annotation[0])
        x_center = float(annotation[1]) * image_width
        y_center = float(annotation[2]) * image_height
        width = float(annotation[3]) * image_width
        height = float(annotation[4]) * image_height

        # 计算边界框的左上角和右下角坐标
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # 获取颜色（默认红色作为后备）
        color = class_colors.get(class_index, (0, 0, 255)) if class_colors else (0, 0, 255)

        # 计算自适应文字大小
        font_scale = max(0.5, min(2.0, (x_max - x_min) / 200))
        font_thickness = max(1, int(font_scale * 1.5))

        # 绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # 绘制类别标签
        if class_colors:
            label = f"Class_{class_index}"
            # 计算文字尺寸
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            # 绘制文字背景
            cv2.rectangle(image, 
                        (x_min, y_min - text_height - 5),
                        (x_min + text_width + 5, y_min - 5),
                        color, -1)
            # 绘制文字
            cv2.putText(image, label, 
                      (x_min, y_min - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale,
                      (255, 255, 255),
                      font_thickness)

    # 保存图片
    cv2.imwrite(output_path, image)
    print(f"标注后的图片已保存为: {output_path}")

def process_images_and_labels(image_folder, labels_folder, output_folder, class_names=None):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 生成全局颜色映射表
    class_colors = {}
    if class_names:
        class_colors = {i: generate_deterministic_color(name) for i, name in enumerate(class_names)}

    # 遍历图片文件夹
    for image_filename in os.listdir(image_folder):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(image_folder, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(labels_folder, label_filename)

            if not os.path.exists(label_path):
                print(f"标注文件缺失: {label_path}")
                continue

            output_path = os.path.join(output_folder, image_filename)
            draw_yolo_bounding_boxes(image_path, label_path, output_path, class_colors)

if __name__ == '__main__':
    image_folder = 'data/raw/images'
    labels_folder = 'data/raw/labels'
    output_folder = 'data/raw/annotated_pictures'
    class_names = ['mushroom']  # 支持多类别扩展
    
    process_images_and_labels(image_folder, labels_folder, output_folder, class_names)
