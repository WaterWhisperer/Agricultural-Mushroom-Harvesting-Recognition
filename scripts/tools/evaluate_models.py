import json
import os
import argparse
import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个矩形框的IoU（交并比）
    box格式: [x, y, w, h] (左上角坐标+宽高)
    """
    # 计算两个框的坐标
    box1_x1, box1_y1 = box1['x'], box1['y']
    box1_x2, box1_y2 = box1['x'] + box1['w'], box1['y'] + box1['h']
    
    box2_x1, box2_y1 = box2['x'], box2['y']
    box2_x2, box2_y2 = box2['x'] + box2['w'], box2['y'] + box2['h']
    
    # 计算交集区域
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    # 计算交集面积
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 计算并集面积
    box1_area = box1['w'] * box1['h']
    box2_area = box2['w'] * box2['h']
    union_area = box1_area + box2_area - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def evaluate_model(gt_path, pred_path):
    """
    评估单个模型的表现
    :param gt_path: 官方标注TXT路径
    :param pred_path: 模型输出TXT路径
    :return: 评估结果字典
    """
    # 加载标注数据
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    # 加载预测数据
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # 初始化评估指标
    total_iou = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    correct_detections = 0
    image_scores = []
    
    # 遍历每张图片
    for img_name, gt_boxes in gt_data.items():
        # 获取该图片的预测结果
        pred_boxes = pred_data.get(img_name, [])
        
        # 统计目标数量
        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)
        
        # 计算该图片的IoU总和
        img_iou_sum = 0
        
        # 匹配预测框和真实框
        for gt_box in gt_boxes:
            best_iou = 0
            best_match = None
            
            # 为每个真实框找到最佳匹配的预测框
            for pred_box in pred_boxes:
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_box
            
            # 如果找到匹配且IoU>0.5，视为正确检测
            if best_iou > 0.5:
                correct_detections += 1
                img_iou_sum += best_iou
                
                # 移除已匹配的预测框
                if best_match in pred_boxes:
                    pred_boxes.remove(best_match)
            else:
                img_iou_sum += 0  # 未检测到，IoU为0
        
        # 计算该图片的平均IoU
        if len(gt_boxes) > 0:
            img_avg_iou = img_iou_sum / len(gt_boxes)
            image_scores.append(img_avg_iou)
    
    # 计算整体指标
    precision = correct_detections / total_pred_boxes if total_pred_boxes > 0 else 0
    recall = correct_detections / total_gt_boxes if total_gt_boxes > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(image_scores) if image_scores else 0
    
    # 综合评分（IoU占60%，F1占40%）
    final_score = 0.6 * mean_iou + 0.4 * f1_score
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_iou": mean_iou,
        "final_score": final_score,
        "total_images": len(gt_data),
        "correct_detections": correct_detections,
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes
    }

def main():
    parser = argparse.ArgumentParser(description="蘑菇识别模型评估工具")
    parser.add_argument('--gt', required=True, help='官方标注TXT文件路径')
    parser.add_argument('--models', nargs='+', required=True, help='模型输出TXT文件路径列表')
    parser.add_argument('--names', nargs='+', required=True, help='模型名称列表，与--models顺序对应')
    
    args = parser.parse_args()
    
    # 验证输入
    if len(args.models) != len(args.names):
        print("错误：模型文件数量与模型名称数量不一致")
        return
    
    # 评估每个模型
    results = {}
    for model_path, model_name in zip(args.models, args.names):
        print(f"正在评估模型: {model_name}...")
        results[model_name] = evaluate_model(args.gt, model_path)
    
    # 打印评估报告
    print("\n" + "="*50)
    print("蘑菇识别模型评估报告")
    print("="*50)
    print(f"官方标注文件: {args.gt}")
    print(f"评估模型数量: {len(args.models)}")
    print("-"*90)
    
    # 打印表头
    print(f"{'模型名称':<20}{'综合评分':<12}{'平均IoU':<12}{'精确率':<12}{'召回率':<12}{'F1分数':<12}")
    print("-"*90)
    
    # 打印每个模型的结果
    for name, metrics in results.items():
        print(f"{name:<25}{metrics['final_score']:<16.4f}{metrics['mean_iou']:<13.4f}  "
              f"{metrics['precision']:<15.4f}  {metrics['recall']:<15.4f}  {metrics['f1_score']:<15.4f}")

    print("\n详细指标说明:")
    print("- 综合评分: 平均IoU(60%) + F1分数(40%)")
    print("- 平均IoU: 所有图片检测框的平均交并比")
    print("- 精确率: 正确检测框数 / 总检测框数")
    print("- 召回率: 正确检测框数 / 总标注框数")
    print("- F1分数: 精确率和召回率的调和平均数")

if __name__ == "__main__":
    main()
