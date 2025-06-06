import os
import time
import sys
from ultralytics import YOLO

#
#参数:
#   img_path: 要识别的图片的路径
#
#返回:
#   返回结果为各赛题中要求的识别结果，具体格式可参考提供压缩包中的 “图片对应输出结果.txt” 中一张图片对应的结果
#
def process_img(img_path, model_path, use_cpu=True):
    """
    处理单张图片的蘑菇检测
    :param img_path: 图片路径
    :param model_path: 模型文件路径
    :param use_cpu: 是否使用CPU
    :return: 检测结果列表
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 设置设备
    device = "cpu" if use_cpu else None
    
    # 执行检测
    results = model(img_path, device=device)
    
    # 提取检测框信息
    boxes = results[0].boxes.xywh.cpu().numpy()
    mushroom_list = [
        {"x": int(box[0]-box[2]/2), "y": int(box[1]-box[3]/2), "w": int(box[2]), "h": int(box[3])}
        for box in boxes
    ]
    
    return mushroom_list

#
#以下代码仅作为选手测试代码时使用，仅供参考，可以随意修改
#但是最终提交代码后，process.py文件是作为模块进行调用，而非作为主程序运行
#因此提交时请根据情况删除不必要的额外代码
#
if __name__=='__main__':
    imgs_folder = 'data/input'
    # 确保图片目录存在
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)
        print(f"创建图片目录: {imgs_folder}")
    
    img_paths = os.listdir(imgs_folder)
    if not img_paths:
        print(f"警告: {imgs_folder} 目录中没有图片文件")
        sys.exit(0)  # 正常退出程序
    
    def now():
        return int(time.time()*1000)
    last_time = 0
    count_time = 0
    max_time = 0
    min_time = now()
    for img_path in img_paths:
        # 构建完整图片路径
        full_img_path = os.path.join(imgs_folder, img_path)
        print(full_img_path,':')
        
        # 检查文件是否存在
        if not os.path.exists(full_img_path):
            print(f"错误: 文件不存在 - {full_img_path}")
            continue
            
        last_time = now()
        result = process_img(full_img_path, "weights/mushroom_v8n.pt")
        run_time = now() - last_time
        print('result:\n',result)
        print('run time: ', run_time, 'ms')
        print()
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    print('\n')
    print('avg time: ',int(count_time/len(img_paths)),'ms')
    print('max time: ',max_time,'ms')
    print('min time: ',min_time,'ms')
