from ultralytics import YOLO
import cv2
import os
import json

def detect_pictures(model_path, input_folder, output_image_folder, output_txt_path):
    model = YOLO(model_path)
    os.makedirs(output_image_folder, exist_ok=True)
    
    detection_results = {}
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(input_folder, filename)
            frame = cv2.imread(image_path)
            
            results = model.predict(source=frame, conf=0.5)
            
            annotated_frame = results[0].plot()
            output_image_path = os.path.join(output_image_folder, f"detected_{filename}")
            cv2.imwrite(output_image_path, annotated_frame)
            
            boxes = results[0].boxes.xywh.cpu().numpy()
            detection_list = []
            for box in boxes:
                x_center, y_center, w, h = box
                x = int(x_center - w/2)
                y = int(y_center - h/2)
                detection_list.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                })
            
            detection_results[filename] = detection_list
            print(f"处理完成: {filename} -> 检测到 {len(detection_list)} 个目标")

    # 使用官方的格式：带制表符缩进，每行一个键值对
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        items = []
        for filename, detections in detection_results.items():
            # 将检测列表转换为紧凑JSON格式（无空格）
            detections_str = json.dumps(detections, ensure_ascii=False, separators=(',', ':'))
            items.append(f'\t"{filename}": {detections_str}')
        f.write(',\n'.join(items))
        f.write('\n}')
    print(f"检测结果已保存至: {output_txt_path}")

if __name__ == "__main__":
    model_path = "weights/best.pt"
    input_folder = "data/test/images"
    output_image_folder = "data/test/detected"
    output_txt_path = "data/test/detections.txt"
    
    detect_pictures(
        model_path, 
        input_folder, 
        output_image_folder,
        output_txt_path
    )
