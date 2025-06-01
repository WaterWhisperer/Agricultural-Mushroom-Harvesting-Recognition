import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import random

def load_images_and_annotations(image_dir, label_dir):
    """加载图像和对应的YOLO格式标注"""
    images = []
    annotations = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        if image is None:
            continue
            
        # 加载对应的标注文件
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    # 转换为pascal_voc格式: [x_min, y_min, x_max, y_max]
                    x_min = (x_center - width/2) * image.shape[1]
                    y_min = (y_center - height/2) * image.shape[0]
                    x_max = (x_center + width/2) * image.shape[1]
                    y_max = (y_center + height/2) * image.shape[0]
                    
                    bboxes.append([x_min, y_min, x_max, y_max])
                    class_labels.append(class_id)
        
        images.append(image)
        annotations.append({
            'image': image,
            'bboxes': bboxes,
            'labels': class_labels
        })
    
    return images, annotations

def save_augmented_data(image, bboxes, labels, output_img_dir, output_label_dir, base_name, idx):
    """保存增强后的图像和标注"""
    # 保存图像
    img_filename = f"{base_name}_aug{idx}.jpg"
    img_path = os.path.join(output_img_dir, img_filename)
    cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # 保存YOLO格式标注
    label_filename = f"{base_name}_aug{idx}.txt"
    label_path = os.path.join(output_label_dir, label_filename)
    
    with open(label_path, 'w') as f:
        for i, bbox in enumerate(bboxes):
            # 转换回YOLO格式 (归一化坐标)
            x_center = ((bbox[0] + bbox[2]) / 2) / image.shape[1]
            y_center = ((bbox[1] + bbox[3]) / 2) / image.shape[0]
            width = (bbox[2] - bbox[0]) / image.shape[1]
            height = (bbox[3] - bbox[1]) / image.shape[0]
            
            f.write(f"{int(labels[i])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def augment_dataset(input_img_dir, input_label_dir, output_img_dir, output_label_dir, augmentations_per_image=20):
    """执行数据增强"""
    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 加载原始数据
    _, annotations = load_images_and_annotations(input_img_dir, input_label_dir)
    
    # 定义增强序列 (支持边界框增强)
    transform = A.Compose([
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.8),
        A.RandomScale(scale_limit=(0.8, 1.2), p=0.8),
        A.RandomSizedCrop(min_max_height=(int(0.9*512), 512), height=512, width=512, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        # 颜色变换
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.8),
        A.ToGray(p=0.2),
        
        # 噪声和模糊
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        
        # 天气效果
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.1),
        
        # 高级增强
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.GridDistortion(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # 执行增强
    for data in tqdm(annotations, desc="Augmenting images"):
        base_name = os.path.splitext(os.path.basename(data.get('image_path', 'image')))[0]
        
        for aug_idx in range(augmentations_per_image):
            try:
                # 应用增强
                transformed = transform(
                    image=data['image'],
                    bboxes=data['bboxes'],
                    labels=data['labels']
                )
                
                # 保存增强结果
                save_augmented_data(
                    transformed['image'],
                    transformed['bboxes'],
                    transformed['labels'],
                    output_img_dir,
                    output_label_dir,
                    base_name,
                    aug_idx
                )
            except Exception as e:
                print(f"Error augmenting image {base_name}: {str(e)}")
                continue

if __name__ == "__main__":
    # 配置路径
    input_image_dir = "data/raw/images"       # 原始图像目录
    input_label_dir = "data/raw/labels"       # 原始标注目录
    output_image_dir = "data/augmented/images"  # 增强图像输出目录
    output_label_dir = "data/augmented/labels"  # 增强标注输出目录
    
    # 执行增强（每张原始图像生成20个增强版本）
    augment_dataset(
        input_image_dir,
        input_label_dir,
        output_image_dir,
        output_label_dir,
        augmentations_per_image=20
    )
