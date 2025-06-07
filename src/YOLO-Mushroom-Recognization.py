import os
import json
import sys  # 导入sys模块用于退出程序
from process import process_img
from ultralytics import YOLO

import argparse

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='蘑菇识别系统')
    parser.add_argument('--model_path', type=str, default='weights/mushroom_v8n.pt', 
                        help='模型文件路径 (默认: weights/mushroom_v8n.pt)')
    parser.add_argument('--source', type=str, default='dir', 
                        choices=['dir', 'camera'], help='输入源: dir(目录图片) 或 camera(摄像头) (默认: dir)')
    parser.add_argument('--input_dir', type=str, default='data/images/', 
                        help='输入图片目录 (默认: data/images/)，当 source=dir 时有效')
    parser.add_argument('--output_file', type=str, default='output.txt', 
                        help='输出文件名 (默认: output.txt)，当 source=dir 时有效')
    parser.add_argument('--camera_id', type=int, default=0, 
                        help='摄像头设备ID (默认: 0)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='实时显示检测画面 (默认: False)')
    parser.add_argument('--use_cpu', action='store_true', default=True,
                        help='使用CPU推理 (默认: True)')
    args = parser.parse_args()
    
    # 加载模型（只加载一次）
    model = YOLO(args.model_path)
    
    # 设置设备类型
    device = "cpu" if args.use_cpu else None
    if device:
        model.to(device)
    
    results = {}
    
    if args.source == 'dir':
        # 处理目录图片模式
        if not os.path.exists(args.input_dir):
            print(f"错误: 输入目录不存在 - {args.input_dir}")
            sys.exit(1)
        
        # 获取所有图片路径
        img_paths = [os.path.join(args.input_dir, img) 
                     for img in os.listdir(args.input_dir) 
                     if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not img_paths:
            print(f"警告: {args.input_dir} 目录中没有图片文件")
            sys.exit(0)
        
        # 处理每张图片
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            mushroom_list = process_img(img_path)
            results[img_name] = mushroom_list
            print(f"处理完成: {img_name} -> 检测到 {len(mushroom_list)} 个目标")
    
    elif args.source == 'camera':
        # 处理摄像头视频流模式
        import cv2
        print(f"正在打开摄像头 (ID: {args.camera_id})...")
        cap = cv2.VideoCapture(args.camera_id)
        
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 (ID: {args.camera_id})")
            sys.exit(1)
        
        print("摄像头已打开，按 'q' 键退出...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头帧")
                break
            
            frame_count += 1
            img_name = f"frame_{frame_count:04d}.jpg"
            
            # 处理当前帧
            mushroom_list = process_img(frame)
            results[img_name] = mushroom_list
            
            # 在画面上绘制检测结果
            for detection in mushroom_list:
                x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 显示检测结果
            if args.show:
                cv2.imshow('Mushroom Detection', frame)
            
            # 打印检测信息
            print(f"帧 {frame_count}: 检测到 {len(mushroom_list)} 个蘑菇")
            
            # 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用官方要求的格式保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write('{\n')
        items = []
        for filename, detections in results.items():
            # 使用默认JSON序列化（冒号后和逗号后有空格）
            detections_str = json.dumps(detections, ensure_ascii=False)
            items.append(f'\t"{filename}": {detections_str}')
        f.write(',\n'.join(items))
        f.write('\n}')
    
    print(f"检测完成，结果已保存至: {args.output_file}")

if __name__ == '__main__':
    main()
