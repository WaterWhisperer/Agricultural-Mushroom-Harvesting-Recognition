<h1 align="center">农业蘑菇收割识别系统实验报告</h1>

## 1. 实验环境

### 1.1 硬件环境

- 运行平台: ARM架构(香橙派或类似设备)
- 内存要求: ≥2GB RAM
- 操作系统: Linux(推荐)

### 1.2 软件环境

- Python 3.8.20+
- PyTorch (CPU版本)
- ultralytics
- OpenCV
- albumentations(用于数据增强)

## 2. 实验方法

### 2.1 数据集处理

1. 原始数据预处理

   - 使用[`scripts/tools/json2label.py`](scripts/tools/json2label.py)将JSON标注转换为YOLO的标签格式
   - 生成标准化的边界框坐标(归一化到0-1范围)

2. 数据增强([`scripts/data_augmentation/data_augmentation.py`](scripts/data_augmentation/data_augmentation.py)):

   - 随机旋转(-30°~30°)
   - 水平/垂直翻转
   - 颜色调整(亮度、对比度、饱和度)
   - 高斯噪声
   - 高斯模糊

<p align="center">
  <img src="/home/water/桌面/Agricultural-Mushroom-Harvesting-Recognition/slide/data_augmentation1.png" alt="Data Augmentation 1"/>
  <img src="/home/water/桌面/Agricultural-Mushroom-Harvesting-Recognition/slide/data_augmentation2.png" alt="Data Augmentation 2"/>
</p>

### 2.2 模型训练

- 基础模型: YOLOv8系列(v8n, v8s)和YOLO11系列(11n, 11s)
- 训练策略:
  - 批次大小: 根据模型配置自适应
  - 图像尺寸: 可配置(默认640x480)
  - 训练轮次: 根据模型类型调整

<p align="center">
  <img src="/home/water/桌面/Agricultural-Mushroom-Harvesting-Recognition/slide/train.png" alt="Training"/>
</p>

## 3. 算法实现

### 3.1 目标检测实现

核心检测逻辑([`src/process.py`](src/process.py)):

```python
def process_img(img_path):
    model_path = 'weights/mushroom2.0_v8n.pt'
    if not hasattr(process_img, "model"):
        process_img.model = YOLO(model_path)
    results = process_img.model(img_path, device="cpu")
    boxes = results[0].boxes.xywh.cpu().numpy()
    # 转换为所需格式...
```

### 3.2 性能优化

1. 模型优化:

   - 图像尺寸优化(416-640可选)
   - 轻量级模型架构

2. 推理优化:

   - 批处理推理
   - CPU优化部署

## 4. 实验结果

### 4.1 检测结果

<p align="center">
  <img src="/home/water/桌面/Agricultural-Mushroom-Harvesting-Recognition/slide/detection1.png" alt="detection1"/>
  <img src="/home/water/桌面/Agricultural-Mushroom-Harvesting-Recognition/slide/detection2.png" alt="detection2"/>
</p>

### 4.2 检测性能

使用[`scripts/tools/evaluate_models.py`](scripts/tools/evaluate_models.py)进行评估:

- 评估指标:
  - 精确率(Precision)
  - 召回率(Recall)
  - F1分数
  - 平均IoU
  - 综合评分(IoU 60% + F1 40%)

<p align="center">
  <img src="/home/water/桌面/Agricultural-Mushroom-Harvesting-Recognition/slide/evaluate_models.png" alt="Evaluate"/>
</p>

### 4.3 运行效率

[`src/process.py`](src/process.py)中的性能测试结果:

- 平均处理时间
- 最大/最小处理时间
- 总体吞吐量

<p align="center">
  <img src="/home/water/桌面/Agricultural-Mushroom-Harvesting-Recognition/slide/performance_test1.png" alt="Performance1"/>
  <img src="/home/water/桌面/Agricultural-Mushroom-Harvesting-Recognition/slide/performance_test2.png" alt="Performance2"/>
</p>

## 5. 结果分析

### 5.1 模型对比分析

- YOLOv8n vs YOLOv8s:

  - v8n: 更快速度，较轻量级
  - v8s: 更高精度，较大模型

- YOLO11n vs YOLO11s:

  - v11n: 更快速度，较轻量级
  - v11s: 更高精度，较大模型

- YOLOV8 vs YOLO11:

  - YOLOv8: 更快速度，精度略低
  - YOLOv11: 更高精度，速度略慢

无论是YOLOv8n、YOLOv8s还是YOLO11n、YOLO11s，在测试集上的精准率在98%左右，但是速度方面，YOLOv8n、YOLO11n的速度更快，且更轻量级。

### 5.2 算法优缺点

优点:

1. 实时性能好
2. 部署简单
3. 精度可靠
4. 资源占用低

缺点:

1. 受相似目标影响较大
2. 对光照敏感
3. CPU推理速度受限

## 6. 改进方向

1. 模型优化:

   - 探索更多轻量级架构
   - 模型量化与压缩
   - 针对性能瓶颈优化

2. 数据增强:

   - 增加更多场景数据
   - 优化数据增强策略
   - 引入更多真实场景数据

3. 针对硬件平台优化:

   - 针对ARM架构进行优化
   - 针对香橙派等含NPU芯片的设备进行优化，如NPU推理加速

## 7. 结论

本项目成功实现了面向ARM设备的蘑菇检测系统，在保证检测精度的同时实现了较好的实时性。通过数据增强、模型优化等手段，系统展现出良好的实用性和可扩展性。未来可进一步优化模型性能和系统适应性。
