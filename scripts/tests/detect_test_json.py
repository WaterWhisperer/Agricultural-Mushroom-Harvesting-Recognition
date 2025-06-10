from ultralytics import YOLO
import cv2
import os
import json

def detect_pictures(model_path, input_folder, output_image_folder, output_json_path):
    model = YOLO(model_path)
    # 创建输出目录
    os.makedirs(output_image_folder, exist_ok=True)
    
    # 初始化结果字典
    detection_results = {}
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(input_folder, filename)
            frame = cv2.imread(image_path)
            
            # 进行检测
            results = model.predict(source=frame, conf=0.5)
            
            # 绘制检测框并保存图像
            annotated_frame = results[0].plot()
            output_image_path = os.path.join(output_image_folder, f"detected_{filename}")
            cv2.imwrite(output_image_path, annotated_frame)
            
            # 提取检测框信息
            boxes = results[0].boxes.xywh.cpu().numpy()
            detection_list = []
            for box in boxes:
                x_center, y_center, w, h = box
                # 转换为左上角坐标 (x, y) 和宽高 (w, h)
                x = int(x_center - w/2)
                y = int(y_center - h/2)
                detection_list.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                })
            
            # 添加到结果字典
            detection_results[filename] = detection_list
            print(f"处理完成: {filename} -> 检测到 {len(detection_list)} 个目标")

    # 保存JSON结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, ensure_ascii=False, indent=4)
    print(f"检测结果已保存至: {output_json_path}")

if __name__ == "__main__":
    model_path = "weights/mushroom2.0_yolov8n.pt"  # 可替换为具体模型路径
    input_folder = "test_picture"
    output_image_folder = "test_detected"
    output_json_path = "detections.json"  # JSON输出路径
    
    detect_pictures(
        model_path, 
        input_folder, 
        output_image_folder,
        output_json_path
    )
