import os
import json
from process import process_img

def main():
    # 配置参数
    imgs_folder = 'data/input'  # 输入图片目录
    output_file = 'output.txt'  # 输出文件名
    model_path = 'weights/mushroom_v8n.pt'  # 模型文件路径
    use_cpu = True  # 使用CPU推理
    
    # 获取所有图片路径
    img_paths = [os.path.join(imgs_folder, img) 
                 for img in os.listdir(imgs_folder) 
                 if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # 处理每张图片
    results = {}
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        mushroom_list = process_img(img_path, model_path, use_cpu)
        results[img_name] = mushroom_list
        print(f"处理完成: {img_name} -> 检测到 {len(mushroom_list)} 个目标")
    
    # 使用官方要求的格式保存结果（带空格）
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('{\n')
        items = []
        for filename, detections in results.items():
            # 使用默认JSON序列化（冒号后和逗号后有空格）
            detections_str = json.dumps(detections, ensure_ascii=False)
            items.append(f'\t"{filename}": {detections_str}')
        f.write(',\n'.join(items))
        f.write('\n}')
    
    print(f"检测完成，结果已保存至: {output_file}")

if __name__ == '__main__':
    main()