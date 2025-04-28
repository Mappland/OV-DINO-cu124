import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def calculate_category_statistics(ground_truth: Dict[str, int], predictions: Dict[str, int]) -> Dict[str, Dict]:
    """
    计算每个类别的识别统计信息
    
    Args:
        ground_truth: 真实标签统计，格式为 {类别名: 数量}
        predictions: 预测标签统计，格式为 {类别名: 数量}
    
    Returns:
        包含每个类别统计信息的字典
    """
    # 合并所有类别
    all_categories = set(ground_truth.keys()) | set(predictions.keys())
    
    category_stats = {}
    for category in all_categories:
        gt_count = ground_truth.get(category, 0)
        pred_count = predictions.get(category, 0)
        
        if gt_count > 0:
            recall = pred_count / gt_count  # 召回率
        else:
            recall = 0.0
            
        if pred_count > 0:
            precision = min(gt_count, pred_count) / pred_count  # 精确率
        else:
            precision = 0.0
            
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        category_stats[category] = {
            "ground_truth": gt_count,
            "predictions": pred_count,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }
    
    return category_stats


def generate_category_chart(category_stats: Dict, output_path: str) -> None:
    """
    生成类别统计图表
    
    Args:
        category_stats: 类别统计信息
        output_path: 图表输出路径
    """
    categories = list(category_stats.keys())
    recalls = [stats["recall"] * 100 for stats in category_stats.values()]
    precisions = [stats["precision"] * 100 for stats in category_stats.values()]
    
    # 按召回率排序显示
    sorted_indices = sorted(range(len(categories)), key=lambda i: recalls[i], reverse=True)
    categories = [categories[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    
    # 图表设置
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, recalls, width, label='召回率')
    plt.bar(x + width/2, precisions, width, label='精确率')
    
    plt.xlabel('类别')
    plt.ylabel('百分比 (%)')
    plt.title('各类别识别性能')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_evaluation_report(
    images_data: List[Dict],
    ground_truth: Dict[str, int],
    predictions: Dict[str, int],
    output_dir: str
) -> None:
    """
    生成Markdown格式的评估报告
    
    Args:
        images_data: 每张图像的评估数据
        ground_truth: 真实标签统计
        predictions: 预测标签统计
        output_dir: 输出目录
    """
    # 计算类别统计信息
    category_stats = calculate_category_statistics(ground_truth, predictions)
    
    # 生成类别图表
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    chart_path = os.path.join(charts_dir, "category_stats.png")
    generate_category_chart(category_stats, chart_path)
    
    report_path = os.path.join(output_dir, "evaluation_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        # 标题
        f.write("# OV-DINO 模型评估报告\n\n")
        
        # 总体统计
        f.write("## 总体统计\n\n")
        total_gt = sum(ground_truth.values())
        total_pred = sum(predictions.values())
        f.write(f"- 评估图像总数: {len(images_data)}\n")
        f.write(f"- 真实标注总数: {total_gt}\n")
        f.write(f"- 模型预测总数: {total_pred}\n")
        f.write(f"- 总体检出率: {(total_pred/total_gt)*100:.2f}%\n\n")
        
        # 类别统计表格
        f.write("## 类别识别统计\n\n")
        f.write("| 类别 | 真实数量 | 预测数量 | 召回率 | 精确率 | F1分数 |\n")
        f.write("| ---- | -------- | -------- | ------ | ------ | ------ |\n")
        
        for category, stats in sorted(category_stats.items(), key=lambda x: x[1]["f1"], reverse=True):
            f.write(f"| {category} | {stats['ground_truth']} | {stats['predictions']} | ")
            f.write(f"{stats['recall']*100:.2f}% | {stats['precision']*100:.2f}% | {stats['f1']*100:.2f}% |\n")
        
        f.write("\n")
        
        # 类别统计图表
        f.write("## 类别检测可视化\n\n")
        rel_chart_path = os.path.relpath(chart_path, output_dir)
        f.write(f"![类别统计图]({rel_chart_path})\n\n")
        
        # 样本图像展示
        f.write("## 样本图像评估\n\n")
        
        # 最多展示10个样本
        sample_images = images_data[:10]
        for idx, image_data in enumerate(sample_images):
            f.write(f"### 样本 {idx+1}: {image_data['file_name']}\n\n")
            
            # 创建两列布局的表格
            f.write("| 原始图像 | 推理结果 |\n")
            f.write("| ------- | ------- |\n")
            
            # 使用相对路径引用图片
            origin_rel_path = os.path.relpath(image_data['origin_path'], output_dir)
            infer_rel_path = os.path.relpath(image_data['infer_path'], output_dir)
            
            f.write(f"| ![原始图像]({origin_rel_path}) | ![推理结果]({infer_rel_path}) |\n\n")
            
            # 标注对比
            f.write("**真实标注:** ")
            f.write(", ".join(image_data["ground_truth"]) if image_data["ground_truth"] else "无")
            f.write("\n\n")
            
            f.write("**模型预测:** ")
            f.write(", ".join(image_data["predictions"]) if image_data["predictions"] else "无")
            f.write("\n\n")
            
            # 分割线
            if idx < len(sample_images) - 1:
                f.write("---\n\n")
    
    print(f"评估报告已生成: {report_path}")


def save_evaluation_data(evaluation_data: Dict, output_dir: str) -> None:
    """
    保存评估数据为JSON文件
    
    Args:
        evaluation_data: 评估数据
        output_dir: 输出目录
    """
    data_path = os.path.join(output_dir, "evaluation_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
