'''
脚本名：convert_format.py
功能：转换蘑菇识别结果格式
使用方法：
1. 运行脚本：python convert_format.py json2txt --input input.json --output output.txt
2. 运行脚本：python convert_format.py txt2json --input input.txt --output output.json
'''
import json
import argparse

def convert_json_to_txt(input_path, output_path):
    """将JSON格式转换为官方TXT格式"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用官方的格式：带制表符缩进，每行一个键值对
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        items = []
        for key, value in data.items():
            # 将值转换为紧凑JSON格式（无空格）
            value_str = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
            items.append(f'\t"{key}": {value_str}')
        f.write(',\n'.join(items))
        f.write('\n}')

def convert_txt_to_json(input_path, output_path):
    """将官方紧凑TXT格式转换为标准JSON格式"""
    # 官方格式是有效的JSON，可以直接加载
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 保存为标准JSON（带缩进）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="蘑菇识别结果格式转换工具")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # JSON转TXT命令
    json2txt_parser = subparsers.add_parser('json2txt', help='将JSON格式转换为官方TXT格式')
    json2txt_parser.add_argument('--input', required=True, help='输入JSON文件路径')
    json2txt_parser.add_argument('--output', required=True, help='输出TXT文件路径')
    
    # TXT转JSON命令
    txt2json_parser = subparsers.add_parser('txt2json', help='将官方TXT格式转换为JSON格式')
    txt2json_parser.add_argument('--input', required=True, help='输入TXT文件路径')
    txt2json_parser.add_argument('--output', required=True, help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'json2txt':
        convert_json_to_txt(args.input, args.output)
        print(f"转换完成: {args.input} -> {args.output}")
    elif args.command == 'txt2json':
        convert_txt_to_json(args.input, args.output)
        print(f"转换完成: {args.input} -> {args.output}")
