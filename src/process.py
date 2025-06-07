import os
import time
import sys
from ultralytics import YOLO

#
#参数:
#   img_path: 要识别的图片的路径
#
#返回:
#   返回结果为各赛题中要求的识别结果，具体格式可参考提供压缩包中的 “图片对应输出结果.txt” 中一张图片对应的结果
#
def process_img(img_path):
    """
    处理单张图片的蘑菇检测
    :param img_path: 图片路径
    :return: 检测结果列表
    """
    # 模型路径
    model_path = 'weights/mushroom_v8n.pt'

    # 只在第一次调用时加载模型
    if not hasattr(process_img, "model"):
        model_path = 'weights/mushroom_v8n.pt'
        process_img.model = YOLO(model_path)

    # 执行检测
    results = process_img.model(img_path, device="cpu")
    
    # 提取检测框信息
    boxes = results[0].boxes.xywh.cpu().numpy()
    mushroom_list = [
        {"x": int(box[0]-box[2]/2), "y": int(box[1]-box[3]/2), "w": int(box[2]), "h": int(box[3])}
        for box in boxes
    ]
    
    return mushroom_list

import argparse

# 性能测试主函数
def performance_test(input_dir):
    # 确保图片目录存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 - {input_dir}")
        sys.exit(1)
    
    img_paths = [os.path.join(input_dir, img) 
                 for img in os.listdir(input_dir) 
                 if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not img_paths:
        print(f"警告: {input_dir} 目录中没有图片文件")
        sys.exit(0)
    
    def now():
        return int(time.time()*1000)
    
    last_time = 0
    count_time = 0
    max_time = 0
    min_time = now()
    
    
    for img_path in img_paths:
        # 构建完整图片路径
        full_img_path = img_path
        print(f"处理图片: {full_img_path}")
        
        last_time = now()
        result = process_img(full_img_path)
        run_time = now() - last_time
        
        print(f"检测到 {len(result)} 个蘑菇, 耗时: {run_time} ms\n")
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    
    print('\n性能统计:')
    print(f"平均处理时间: {int(count_time/len(img_paths))} ms")
    print(f"最大处理时间: {max_time} ms")
    print(f"最小处理时间: {min_time} ms")
    print(f"总处理时间: {count_time} ms")
    print(f"处理图片数量: {len(img_paths)}")

if __name__=='__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='蘑菇识别系统性能测试')
    parser.add_argument('--input_dir', type=str, default='data/images/', 
                        help='输入图片目录 (默认: data/images/)')
    args = parser.parse_args()
    
    performance_test(args.input_dir)
