#!/bin/bash

# 默认参数值
DEFAULT_MODELS=("yolov8n" "yolov8s" "yolo11n" "yolo11s")
DEFAULT_DATA_PATH="data/data.yaml"
DEFAULT_WEIGHTS_DIR="weights/"
DEFAULT_NAME="mushroom"
DEFAULT_EPOCHS=100
DEFAULT_BATCH=16
DEFAULT_IMGSZ=640

# 显示帮助信息
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --models     Comma-separated list of models (default: ${DEFAULT_MODELS[*]})"
    echo "  -d, --data       Path to data.yaml (default: $DEFAULT_DATA_PATH)"
    echo "  -w, --weights    Base weights directory (default: $DEFAULT_WEIGHTS_DIR)"
    echo "  -n, --name       Base name for runs (default: $DEFAULT_NAME)"
    echo "  -e, --epochs     Number of epochs (default: $DEFAULT_EPOCHS)"
    echo "  -b, --batch      Batch size (default: $DEFAULT_BATCH)"
    echo "  -i, --imgsz      Image size (default: $DEFAULT_IMGSZ)"
    echo "  -h, --help       Show this help message"
    exit 0
}

# 验证模型格式
validate_models() {
    local models_str="$1"
    # 检查是否包含空格
    if [[ "$models_str" == *" "* ]]; then
        echo "Error: Models should be comma-separated without spaces"
        echo "Example: yolov8n,yolov8s"
        exit 1
    fi
    # 检查是否为空
    if [[ -z "$models_str" ]]; then
        echo "Error: No models specified"
        exit 1
    fi
    # 检查是否只包含逗号
    if [[ "$models_str" == *,* && ! "$models_str" =~ [a-zA-Z0-9] ]]; then
        echo "Error: Invalid model format"
        echo "Example: yolov8n,yolov8s"
        exit 1
    fi
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            validate_models "$2"
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -w|--weights)
            BASE_WEIGHTS_DIR="$2"
            shift 2
            ;;
        -n|--name)
            NAME="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH="$2"
            shift 2
            ;;
        -i|--imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# 使用默认值（如果未指定）
MODELS=${MODELS:-${DEFAULT_MODELS[@]}}
DATA_PATH=${DATA_PATH:-$DEFAULT_DATA_PATH}
BASE_WEIGHTS_DIR=${BASE_WEIGHTS_DIR:-$DEFAULT_WEIGHTS_DIR}
NAME=${NAME:-$DEFAULT_NAME}

# 打印当前配置
echo "========================================"
echo "Training Configuration:"
echo "Models: ${MODELS[*]}"
echo "Data Path: $DATA_PATH"
echo "Weights Directory: $BASE_WEIGHTS_DIR"
echo "Run Name: $NAME"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH"
echo "Image Size: $IMGSZ"
echo "========================================"

# 循环训练不同模型
for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Training $model..."
    echo "========================================"
    
    # 设置参数（可根据不同模型调整参数）
    case $model in
        "yolov8n")
            epochs=$EPOCHS
            batch=$BATCH
            imgsz=$IMGSZ
            weights_type=".pt"
            ;;
        "yolo11n")
            epochs=$EPOCHS
            batch=$BATCH
            imgsz=$IMGSZ
            weights_type=".pt"
            ;;
        "yolov8s")
            epochs=$EPOCHS
            batch=$BATCH
            imgsz=$IMGSZ
            weights_type=".pt"
            ;;
        "yolo11s")
            epochs=$EPOCHS
            batch=$BATCH
            imgsz=$IMGSZ
            weights_type=".pt"
            ;;
        *)
            epochs=$EPOCHS
            batch=$BATCH
            imgsz=$IMGSZ
            weights_type=".pt"
            ;;
    esac

    # 执行训练命令
    python scripts/train/train.py \
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
