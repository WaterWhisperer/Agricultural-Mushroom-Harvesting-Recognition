#!/bin/bash

# 蘑菇识别系统一键运行脚本
# 使用默认配置运行主程序

# 检查传递的参数个数
if [ $# -gt 1 ]; then
  echo "错误: 请正确输入测试图片所在目录"
  exit 1
fi

echo "正在运行蘑菇识别系统（CPU模式）..."
echo "---------------------------------"

# 执行主识别程序（使用默认参数）
python src/YOLO-Mushroom-Recognization.py --input_dir $1

echo "---------------------------------"
echo "识别完成！结果保存在 output.txt 文件中"
