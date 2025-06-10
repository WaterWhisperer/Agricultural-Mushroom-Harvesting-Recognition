# 农业蘑菇收割识别项目

本项目专注于蘑菇识别，使用YOLO模型实现高效识别。

## 环境要求

- Python 3.8.20+
- PyTorch CPU版本
- ultralytics
- OpenCV

## 项目结构

```plaintext
.
├── data/                     # 数据集目录
├── doc/                      # 文档
├── scripts/                  # 辅助脚本
├── slide/                    # PPT 
├── src/                      # 项目代码
├── video/                    # 演示视频
├── weights/                  # 模型权重文件
├── experimental_report.pdf   # 实验报告
├── README.md                 # 项目说明文档
└── run.sh                    # 启动脚本 
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/WaterWhisperer/Agricultural-Mushroom-Harvesting-Recognition.git
cd Agricultural-Mushroom-Harvesting-Recognition
```

### 2. 配置环境与安装依赖

```bash
# 创建conda环境并激活
conda create -n mushroom python=3.11
conda activate mushroom

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备测试图片

将测试图片放入 `data/images/`目录

### 4. 运行系统

```bash
# 使用默认配置运行（自动使用CPU模式）
bash run.sh data/images
```

### 5. 查看结果

识别结果将保存在 `output.txt` 文件中。

## 高级选项

详见[doc/usage_guide.md](doc/usage_guide.md)

### 性能测试

```bash
python src/process.py --input_dir your/images/dir
```

### 模型评估

```bash
#第一步
#先解压raw.zip获取“图片对应输出结果.txt”

#第二步
#执行程序获取模型评估结果（请按照实际路径更新参数）
python scripts/tools/evaluate_models.py \
  --gt data/raw/图片对应输出结果.txt \
  --models data/test/detections_v8n.txt data/test/detections_v8s.txt \
  --names v8n v8s
```

## 硬件要求

- 香橙派或类似ARM设备
- 至少2GB RAM
- 推荐使用Linux系统
