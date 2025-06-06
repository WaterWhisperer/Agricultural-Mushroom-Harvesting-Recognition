#!/bin/bash

# 蘑菇识别系统运行脚本
# 默认使用CPU模式运行

echo "正在运行蘑菇识别系统（CPU模式）..."
echo "---------------------------------"

# 执行识别程序
python src/YOLO-Mushroom-Recognization.py

echo "---------------------------------"
echo "识别完成！结果保存在 output.txt 文件中"
