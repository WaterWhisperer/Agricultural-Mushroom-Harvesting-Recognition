from ultralytics import YOLO
import cv2
import os

def detect_pictures(model_path, input_folder, output_folder):
    model = YOLO(model_path)
    #创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(input_folder, filename)
            frame = cv2.imread(image_path)

            results = model.predict(source=frame ,conf = 0.5)
            annotated_frame = results[0].plot()

            output_path = os.path.join(output_folder, f"detected_{filename}")
            cv2.imwrite(output_path, annotated_frame)
            print(f"处理完成: {filename} -> 保存至 {output_path}")

if __name__ == "__main__":
    model_path = "weights/best.pt"
    input_folder = "data/test/images"
    output_folder = "data/test/detected"
    detect_pictures(model_path, input_folder, output_folder)