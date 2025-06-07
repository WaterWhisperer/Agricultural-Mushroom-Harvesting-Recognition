#!/bin/bash

# 蘑菇识别系统一键运行脚本
# 使用默认配置运行主程序

echo "正在运行蘑菇识别系统（CPU模式）..."
echo "---------------------------------"

# 执行主识别程序（使用默认参数）
python src/YOLO-Mushroom-Recognization.py

echo "---------------------------------"
echo "识别完成！结果保存在 output.txt 文件中"
