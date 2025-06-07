# 蘑菇识别系统使用指南

## 主程序参数详解

### 参数列表

| 参数名            | 类型 | 默认值                      | 描述                                                |
| ----------------- | ---- | --------------------------- | --------------------------------------------------- |
| `--model_path`  | str  | `weights/mushroom_v8n.pt` | 模型文件路径                                        |
| `--source`      | str  | `dir`                     | 输入源类型：`dir`(目录图片) 或 `camera`(摄像头) |
| `--input_dir`   | str  | `data/images/`            | 输入图片目录 (当source=dir时有效)                   |
| `--output_file` | str  | `output.txt`              | 输出文件名 (当source=dir时有效)                     |
| `--camera_id`   | int  | `0`                       | 摄像头设备ID                                        |
| `--show`        | bool | `False`                   | 实时显示检测画面                                    |
| `--use_cpu`     | bool | `True`                    | 使用CPU推理                                         |

### 使用示例

```bash

# 基本图片检测
python src/YOLO-Mushroom-Recognization.py

# 使用v8s模型检测自定义目录
python src/YOLO-Mushroom-Recognization.py \
  --model_path weights/mushroom_v8s.pt \
  --input_dir custom_imgs \
  --output_file custom_output.json

# 摄像头检测（显示画面）
python src/YOLO-Mushroom-Recognization.py \
  --source camera \
  --camera_id 0 \
  --show
```

## 辅助脚本使用指南

### 数据增强

```bash
# 创建数据增强环境
conda env create -f scripts/data_augmentation/environment.yaml

# 运行数据增强脚本
python scripts/data_augmentation/data_augmentation.py \
  --input_dir data/raw/images \
  --output_dir data/augmented \
  --augmentations rotate flip color
```

支持的数据增强操作：

- `rotate`：随机旋转（-30°至30°）
- `flip`：水平/垂直翻转
- `color`：颜色调整（亮度、对比度、饱和度）
- `noise`：添加高斯噪声
- `blur`：高斯模糊

### 模型训练

```bash
# 启动训练
bash scripts/train/train.sh \
  --model yolov8n.pt \
  --data data/data.yaml \
  --epochs 100 \
  --imgsz 640
```

训练参数：

- `--model`：基础模型架构
- `--data`：数据配置文件路径
- `--epochs`：训练轮数
- `--imgsz`：输入图片尺寸
- `--batch`：批次大小（根据显存调整）

### 模型评估工具

```bash
python scripts/tools/evaluate_models.py \
  --gt data/raw/图片对应输出结果.txt \
  --models data/test/detections_v8n.txt data/test/detections_v8s.txt \
  --names v8n v8s \
```

评估指标：

- 精确率(Precision)
- 召回率(Recall)
- F1分数
- mAP@0.5
- 推理速度(FPS)

### 标注转换工具

```bash
# JSON转YOLO格式
python scripts/tools/json2txt.py
```

### 模型转换工具

```bash
# 导出为ONNX格式
python scripts/tools/export_onnx.py \
  --model weights/mushroom_v8n.pt \
  --output weights/mushroom_v8n.onnx 
```

## 性能优化

### 内存优化

- 使用 `--half`参数启用半精度推理
- 减小 `--imgsz`参数值（如从640降至416）
- 使用更小的模型架构（如yolov8n）

## 常见问题解决

### 摄像头无法打开

**症状**：`无法打开摄像头 (ID: 0)`**解决方案**：

1. 检查摄像头连接
2. 尝试不同摄像头ID（0,1,2等）
3. 在Linux系统添加用户到video组：

   ```bash
   sudo usermod -aG video $USER
   ```

### 性能问题

**症状**：处理速度慢**优化方案**：

1. 使用量化模型：`--model_path weights/mushroom_v8n_quant.pt`
2. 启用半精度：添加 `--half`参数
3. 减小图片尺寸：修改代码中的 `imgsz`参数
