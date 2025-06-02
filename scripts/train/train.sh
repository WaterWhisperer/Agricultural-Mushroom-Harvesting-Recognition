#!/bin/bash

# 定义模型列表和参数
MODELS=("yolov8n" "yolov8s" "yolo11n" "yolo11s")  # 根据实际模型名称修改
DATA_PATH="data/data.yaml"
BASE_WEIGHTS_DIR="weights/"
NAME="mushroom"

# 循环训练不同模型
for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Training $model..."
    echo "========================================"
    
    # 设置参数（可根据不同模型调整参数）
    case $model in
        "yolov8n")
            epochs=120
            batch=16
            imgsz=640
            weights_type=".pt"
            ;;
        "yolo11n")
            epochs=120
            batch=16
            imgsz=640
            weights_type=".pt"
            ;;
        "yolov8s")
            epochs=120
            batch=16
            imgsz=640
            weights_type=".pt"
            ;;
        "yolo11s")
            epochs=120
            batch=16
            imgsz=640
            weights_type=".pt"
            ;;
        *)
            epochs=200
            batch=8
            imgsz=640 
            weights_type=".pt"
            ;;
    esac

    # 执行训练命令
    python train.py \
        --model "$model" \
        --data "$DATA_PATH" \
        --epochs $epochs \
        --batch $batch \
        --weights "$BASE_WEIGHTS_DIR$model$weights_type" \
        --imgsz $imgsz \
        --name "${NAME}_${model}"

    # 等待10秒防止资源冲突
    sleep 10
done

echo "All models trained!"