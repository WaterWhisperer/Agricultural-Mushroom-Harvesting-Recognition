# 农业蘑菇收割识别项目

本项目专注于蘑菇识别，专为CPU设备（如香橙派）优化，使用YOLO模型实现高效识别。

## 环境要求

- Python 3.8+
- PyTorch CPU版本
- ultralytics
- OpenCV

## 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 单独安装PyTorch CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 项目结构

```plaintext
.
├── data/                 # 数据集目录
├── doc/                  # 文档
├── scripts/              # 辅助脚本
├── slide/                # PPT 
├── src/                  # 项目代码
├── video/                # 演示视频
├── weights/              # 模型权重文件
├── README.md             # 项目说明文档
└── run.sh                # 启动脚本 
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/WaterWhisperer/Agricultural-Mushroom-Harvesting-Recognition.git
cd Agricultural-Mushroom-Harvesting-Recognition
```

### 2. 配置环境与安装依赖（两种方式）

#### 方式一

```bash
# 创建并激活环境
conda env create -f environment.yaml
conda activate mushroom
```

#### 方式二

```bash
# 创建conda环境
conda create -n mushroom python=3.11

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 单独安装PyTorch CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. 准备测试图片

将测试图片放入`data/input/`目录

### 4. 运行系统

```bash
# 使用默认配置运行（自动使用CPU模式）
bash run.sh

# 或手动运行主程序
python src/YOLO-Mushroom-Recognization.py
```

### 5. 查看结果

识别结果将保存在 `output.txt` 文件中，格式符合官方要求。

## 高级选项

### 指定模型

```bash
python src/YOLO-Mushroom-Recognization.py --model_path weights/mushroom_v8n.pt
```

### 性能测试

```bash
python src/process.py
```

## 模型评估

```bash
python scripts/evaluate_models.py \
  --gt data/raw/图片对应输出结果.txt \
  --models data/test/detections_v8n.txt data/test/detections_v8s.txt \
  --names v8n v8s
```

## 硬件要求

- 香橙派或类似ARM设备
- 至少2GB RAM
- 推荐使用Linux系统
