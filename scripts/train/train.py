# train.py
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='YOLO Model Training')
    parser.add_argument('--model', type=str, required=True, help='Base model name (e.g. yolov8n, yolov11n)')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--weights', type=str, default='', help='Custom weights path')
    parser.add_argument('--name', type=str, default='', help='Custom run name')

    args = parser.parse_args()

    # 加载模型（优先使用自定义权重）
    model_path = args.weights if args.weights else f"{args.model}.pt"
    model = YOLO(model_path)

    # 训练配置
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=f"{args.name}_custom_train"
    )

if __name__ == "__main__":
    main()