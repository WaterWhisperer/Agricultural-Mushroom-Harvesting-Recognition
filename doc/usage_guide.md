# 蘑菇识别系统使用指南

## 基本使用

### 1. 快速启动

```bash
bash run.sh
```

这将使用默认配置运行系统：

- 使用`weights/mushroom_v8n.pt`模型
- 自动检测`data/images/`目录中的图片
- 结果保存到`output.txt`

### 2. 指定模型

```bash
python src/YOLO-Mushroom-Recognization.py --model_path weights/mushroom_11s.pt
```

可用的模型文件：

- `mushroom_v8n.pt` (默认)
- `mushroom_v8s.pt`
- `mushroom_11n.pt`
- `mushroom_11s.pt`

### 3. 自定义输入/输出

```bash
python src/YOLO-Mushroom-Recognization.py \
  --input_dir custom_imgs \
  --output_file custom_output.json
```

## 高级功能

### 性能测试

```bash
python src/process.py
```

测试所有图片的处理时间并输出统计信息：

- 平均处理时间
- 最大处理时间
- 最小处理时间

### 模型评估

```bash
python scripts/tools/evaluate_models.py \
  --gt data/raw/图片对应输出结果.txt \
  --models data/test/detections_v8n.txt data/test/detections_v8s.txt \
  --names v8n v8s
```

评估不同模型在测试集上的性能：

- 精确率(Precision)
- 召回率(Recall)
- F1分数

### 模型导出(ONNX格式)

```bash
python scripts/tools/export_onnx.py \
  --model weights/mushroom_v8n.pt \
  --output weights/mushroom_v8n.onnx
```

导出模型为ONNX格式，便于在边缘设备部署。

## 常见问题解决

### 图片处理失败

如果出现`FileNotFoundError`错误：

1. 确认图片路径正确
2. 检查文件权限
3. 验证图片格式（支持JPG、PNG等常见格式）

### 性能优化

在低性能设备上：

1. 使用更小的模型（如`mushroom_v8n.pt`）
2. 降低图片分辨率（修改代码中的`imgsz`参数）
3. 使用`--half`参数启用半精度推理（如果设备支持）
