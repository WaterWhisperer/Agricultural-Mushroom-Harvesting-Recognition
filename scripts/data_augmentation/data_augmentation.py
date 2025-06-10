import cv2
import numpy as np
import os
import albumentations as A
from tqdm import tqdm
import random
import shutil
import argparse

def load_images_and_annotations(image_dir, label_dir):
    """加载图像和对应的YOLO格式标注"""
    annotations = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # 加载对应的标注文件
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                        
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError:
                        continue
                    
                    # 转换为pascal_voc格式: [x_min, y_min, x_max, y_max]
                    x_min = (x_center - width/2) * img_width
                    y_min = (y_center - height/2) * img_height
                    x_max = (x_center + width/2) * img_width
                    y_max = (y_center + height/2) * img_height
                    
                    # 确保边界框有效
                    if x_min < x_max and y_min < y_max:
                        bboxes.append([x_min, y_min, x_max, y_max])
                        class_labels.append(class_id)
        
        if bboxes:
            annotations.append({
                'image': image,
                'bboxes': bboxes,
                'labels': class_labels,
                'file_name': img_file  # 保存原始文件名
            })
        else:
            print(f"警告: {img_file} 没有有效的边界框")
    
    print(f"成功加载 {len(annotations)} 个有效图像和标注")
    return annotations

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
            img_height, img_width = image.shape[:2]
            x_center = ((bbox[0] + bbox[2]) / 2) / img_width
            y_center = ((bbox[1] + bbox[3]) / 2) / img_height
            width = (bbox[2] - bbox[0]) / img_width
            height = (bbox[3] - bbox[1]) / img_height
            
            # 确保坐标在0-1范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            f.write(f"{int(labels[i])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def get_augmentation_pipeline(mode='train'):
    """根据模式获取增强管道"""
    if mode == 'train':
        # 训练集使用更强的增强
        return A.Compose([
            # 几何变换 - 更温和
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(0.8, 1.2), p=0.5),
            
            # 颜色变换
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            A.ToGray(p=0.2),
            
            # 噪声和模糊
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MedianBlur(blur_limit=3),
            ], p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        # 验证集使用较温和的增强
        return A.Compose([
            # 几何变换
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.RandomResizedCrop(height=512, width=512, scale=(0.7, 1.0), ratio=(0.7, 1.3), p=0.5),
            
            # 颜色变换
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            A.ToGray(p=0.2),
            
            # 噪声和模糊
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def augment_dataset(input_img_dir, input_label_dir, output_img_dir, output_label_dir, augmentations_per_image=20, mode='train'):
    """执行数据增强"""
    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 加载原始数据
    annotations = load_images_and_annotations(input_img_dir, input_label_dir)
    
    if not annotations:
        print("错误: 没有找到有效的图像和标注数据")
        return
    
    # 获取增强管道
    transform = get_augmentation_pipeline(mode)
    
    # 执行增强
    success_count = 0
    for data in tqdm(annotations, desc=f"增强图像 ({mode}模式)"):
        # 使用原始文件名作为基础名称
        base_name = os.path.splitext(data['file_name'])[0]
        
        for aug_idx in range(augmentations_per_image):
            try:
                # 应用增强
                transformed = transform(
                    image=data['image'],
                    bboxes=data['bboxes'],
                    labels=data['labels']
                )
                
                # 检查增强后的边界框
                valid_bboxes = []
                valid_labels = []
                img_height, img_width = transformed['image'].shape[:2]
                
                for bbox, label in zip(transformed['bboxes'], transformed['labels']):
                    # 确保边界框有效
                    if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                        valid_bboxes.append(bbox)
                        valid_labels.append(label)
                
                if not valid_bboxes:
                    continue
                
                # 保存增强结果
                save_augmented_data(
                    transformed['image'],
                    valid_bboxes,
                    valid_labels,
                    output_img_dir,
                    output_label_dir,
                    base_name,
                    aug_idx
                )
                success_count += 1
                
            except Exception as e:
                continue
    
    print(f"完成! 成功生成 {success_count} 个增强样本")
    print(f"原始图像: {len(annotations)} 张, 增强倍数: {augmentations_per_image}, 总样本数: {len(annotations)*augmentations_per_image}")
    print(f"有效增强率: {success_count/(len(annotations)*augmentations_per_image)*100:.2f}%")
    return success_count

def prepare_test_set(input_img_dir, input_label_dir, output_img_dir, output_label_dir):
    """准备测试集（原始数据）"""
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 复制原始图像
    for img_file in os.listdir(input_img_dir):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            src_img = os.path.join(input_img_dir, img_file)
            dst_img = os.path.join(output_img_dir, img_file)
            if not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)
            
            # 复制对应标注
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(input_label_dir, label_file)
            dst_label = os.path.join(output_label_dir, label_file)
            if os.path.exists(src_label) and not os.path.exists(dst_label):
                shutil.copy(src_label, dst_label)
    
    print(f"测试集准备完成: {len(os.listdir(output_img_dir))} 张图像")

if __name__ == "__main__":
    # 配置路径
    parser = argparse.ArgumentParser(description="蘑菇图片数据增强工具")
    parser.add_argument('--input_images_dir', required=True, help='原始图像目录')
    parser.add_argument('--input_labels_dir', required=True, help='原始标注目录') 
    parser.add_argument('--output_dir', required=True, help='增强后数据集目录')
    parser.add_argument('--augmentations_per_train_image', required=True, help='每张图片增强次数')
    parser.add_argument('--augmentations_per_val_image', required=True, help='每张图片增强次数')
    args = parser.parse_args()
    
    # 打印标题
    print("="*50)
    print("蘑菇识别数据增强脚本")
    print("="*50)
    
    # 创建训练集 
    print(f"\n生成训练集({args.augmentations_per_train_image}轮增强):")
    train_count = augment_dataset(
        args.input_images_dir,
        args.input_labels_dir,
        args.output_dir+"/train/images",
        args.output_dir+"/train/labels",
        augmentations_per_image=args.augmentations_per_train_image,
        mode='train'
    )
    
    # 创建验证集 
    print(f"\n生成验证集 ({args.augmentations_per_val_image}轮增强):")
    val_count = augment_dataset(
        args.input_images_dir,
        args.input_labels_dir,
        args.output_dir+"/val/images",
        args.output_dir+"/val/labels",
        augmentations_per_image=args.augmentations_per_val_image,
        mode='val'
    )
    
    # 准备测试集 (原始数据)
    print("\n准备测试集 (原始数据):")
    prepare_test_set(
        args.input_images_dir,
        args.input_labels_dir,
        args.output_dir+"/test/images",
        args.output_dir+"/test/labels"
    )
    
    # 最终统计
    print("\n="*50)
    print(f"训练集: {train_count} 张增强图像")
    print(f"验证集: {val_count} 张增强图像")
    print(f"测试集: {len(os.listdir('data/test/images'))} 张原始图像")
    print("="*50)
