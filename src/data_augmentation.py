import cv2
import numpy as np
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import random
from tqdm import tqdm

def load_images_and_annotations(image_dir, label_dir):
    """加载图像和对应的YOLO格式标注"""
    images = []
    annotations = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # 加载对应的标注文件
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    bboxes.append(BoundingBox(
                        x1=(x_center - width/2) * image.shape[1],
                        y1=(y_center - height/2) * image.shape[0],
                        x2=(x_center + width/2) * image.shape[1],
                        y2=(y_center + height/2) * image.shape[0],
                        label=class_id
                    ))
        
        images.append(image)
        annotations.append(BoundingBoxesOnImage(bboxes, shape=image.shape))
    
    return images, annotations

def save_augmented_data(image, bboxes, output_img_dir, output_label_dir, base_name, idx):
    """保存增强后的图像和标注"""
    # 保存图像
    img_filename = f"{base_name}_aug{idx}.jpg"
    img_path = os.path.join(output_img_dir, img_filename)
    cv2.imwrite(img_path, image)
    
    # 保存YOLO格式标注
    label_filename = f"{base_name}_aug{idx}.txt"
    label_path = os.path.join(output_label_dir, label_filename)
    
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            # 转换回YOLO格式 (归一化坐标)
            x_center = ((bbox.x1 + bbox.x2) / 2) / image.shape[1]
            y_center = ((bbox.y1 + bbox.y2) / 2) / image.shape[0]
            width = (bbox.x2 - bbox.x1) / image.shape[1]
            height = (bbox.y2 - bbox.y1) / image.shape[0]
            
            f.write(f"{int(bbox.label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def augment_dataset(input_img_dir, input_label_dir, output_img_dir, output_label_dir, augmentations_per_image=20):
    """执行数据增强"""
    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 加载原始数据
    images, annotations = load_images_and_annotations(input_img_dir, input_label_dir)
    
    # 定义增强序列
    seq = iaa.Sequential([
        # 几何变换
        iaa.Sometimes(0.7, iaa.Fliplr(0.5)),  # 50%概率水平翻转
        iaa.Sometimes(0.7, iaa.Flipud(0.3)),  # 30%概率垂直翻转
        iaa.Sometimes(0.8, iaa.Affine(
            rotate=(-30, 30),  # 旋转
            shear=(-15, 15),   # 剪切
            scale=(0.8, 1.2)   # 缩放
        )),
        iaa.Sometimes(0.5, iaa.CropAndPad(
            percent=(-0.1, 0.1),  # 随机裁剪和填充
            pad_mode='constant'
        )),
        iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.01, 0.15))),  # 透视变换
        
        # 颜色变换
        iaa.Sometimes(0.8, iaa.OneOf([
            iaa.Multiply((0.8, 1.2)),  # 亮度调整
            iaa.LinearContrast((0.8, 1.2)),  # 对比度调整
            iaa.AddToHueAndSaturation((-20, 20)),  # 色调和饱和度
            iaa.Grayscale(alpha=(0.0, 1.0))  # 灰度化
        ])),
        
        # 噪声和模糊
        iaa.Sometimes(0.5, iaa.OneOf([
            iaa.GaussianBlur(sigma=(0, 1.5)),  # 高斯模糊
            iaa.MotionBlur(k=7, angle=[-45, 45]),  # 运动模糊
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # 高斯噪声
            iaa.ImpulseNoise(0.05)  # 脉冲噪声
        ])),
        
        # 天气效果
        iaa.Sometimes(0.2, iaa.Clouds()),
        iaa.Sometimes(0.2, iaa.Fog()),
        
        # 高级增强
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)),
        iaa.Sometimes(0.2, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
    ], random_order=True)

    # 执行增强
    for img_idx, (image, bboxes) in tqdm(enumerate(zip(images, annotations)), 
                                       total=len(images),
                                       desc="Augmenting images"):
        base_name = os.path.splitext(os.listdir(input_img_dir)[img_idx])[0]
        
        for aug_idx in range(augmentations_per_image):
            # 应用增强
            seq_det = seq.to_deterministic()
            augmented_img = seq_det.augment_image(image)
            augmented_bboxes = seq_det.augment_bounding_boxes([bboxes])[0]
            
            # 保存增强结果
            save_augmented_data(
                augmented_img, 
                augmented_bboxes.bounding_boxes,
                output_img_dir,
                output_label_dir,
                base_name,
                aug_idx
            )

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
