# 蘑菇识别系统启动指南

## 系统要求

- Linux操作系统（推荐Ubuntu 20.04+）
- Python 3.8+
- 至少2GB内存
- 香橙派或类似ARM设备（仅支持CPU模式）

## 安装步骤

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

将测试图片放入data/input/目录

### 4. 运行系统

```bash
# 使用默认配置运行（自动使用CPU模式）
bash run.sh

# 或手动运行主程序
python src/YOLO-Mushroom-Recognization.py
```

## 性能测试

```bash
python src/process.py
```

成功运行后将显示每张图片的处理时间和性能统计。
