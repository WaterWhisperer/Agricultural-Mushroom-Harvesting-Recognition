import json
import os

def convert_to_yolo(json_data, output_dir, image_width=640, image_height=480):
    """
    将JSON格式的标注数据转换为YOLO格式
    
    参数:
        json_data: 包含标注信息的JSON数据
        image_width: 图像宽度(默认640)
        image_height: 图像高度(默认480)
        output_dir: 输出目录(默认'yolo_labels')
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每张图像的标注
    for image_name, bboxes in json_data.items():
        # 创建对应的YOLO标签文件(与图像同名，扩展名为.txt)
        label_filename = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                # 提取边界框信息
                x_center = bbox['x'] + bbox['w'] / 2
                y_center = bbox['y'] + bbox['h'] / 2
                width = bbox['w']
                height = bbox['h']
                
                # 归一化到[0,1]范围
                x_center /= image_width
                y_center /= image_height
                width /= image_width
                height /= image_height
                
                # YOLO格式: class_id x_center y_center width height
                # 这里假设所有对象都是同一类(class_id=0)
                line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                f.write(line)
    
    print(f"转换完成，结果保存在 {output_dir} 目录中")

if __name__ == "__main__":
    # JSON数据文件路径和输出目录
    json_file_path = "data/raw/图片对应输出结果.txt"
    output_dir = "data/raw/labels"

    with open(json_file_path, "r", encoding="utf-8") as f:
        json_str = f.read()   

    # 解析JSON
    json_data = json.loads(json_str)
    
    # 转换为YOLO格式
    # 注意: 需要根据实际情况设置正确的图像宽度和高度
    convert_to_yolo(json_data, output_dir=output_dir, image_width=640, image_height=480)