# 蘑菇识别系统启动指南

## 系统要求

- Linux操作系统（推荐Ubuntu 20.04+）
- Python 3.8.20+
- 至少2GB内存
- 香橙派或类似ARM设备

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/WaterWhisperer/Agricultural-Mushroom-Harvesting-Recognition.git
cd Agricultural-Mushroom-Harvesting-Recognition
```

### 2. 配置环境与安装依赖

```bash
# 创建conda环境
conda create -n mushroom python=3.11

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备测试图片

将测试图片放入 `data/images/`目录

### 4. 运行系统

```bash
# 使用默认配置运行（自动使用CPU模式）
bash run.sh

# 或手动运行主程序
python src/YOLO-Mushroom-Recognization.py
```

## 性能测试

```bash
# 使用默认配置运行性能测试
python src/process.py

# 自定义输入图像目录参数
python src/process.py \
  --input_dir custom_imgs
```

成功运行后将显示每张图片的处理时间和性能统计。
