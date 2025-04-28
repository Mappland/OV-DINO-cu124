import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

# 添加项目路径
sys.path.insert(0, "./")  # 确保可以导入项目模块

from demo.predictors import OVDINODemo
from detrex.data.datasets import clean_words_or_phrase

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.evaluation.coco_evaluation import instances_to_coco_json


# 修复torch.load的问题
def patch_torch_load():
    import types

    def patched_torch_load(self, f):
        return torch.load(f, map_location=torch.device("cuda"), weights_only=False)

    DetectionCheckpointer._torch_load = types.MethodType(
        patched_torch_load, DetectionCheckpointer
    )


# 加载模型
def build_model(
    config_file: str, checkpoint_file: str, device: str = "cuda"
) -> Tuple[Any, OVDINODemo]:
    """构建并加载OV-DINO模型"""

    # 默认参数
    min_size_test = 800
    max_size_test = 1333
    img_format = "RGB"
    metadata_dataset = "coco_2017_val"

    # 加载配置和实例化模型
    cfg = LazyConfig.load(config_file)
    model = instantiate(cfg.model)
    model.to(torch.device(device))
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_file)
    model.eval()

    demo = OVDINODemo(
        model=model,
        sam_predictor=None,
        min_size_test=min_size_test,
        max_size_test=max_size_test,
        img_format=img_format,
        metadata_dataset=metadata_dataset,
    )

    return model, demo


# 可视化原始图片
def visualize_origin_image(
    coco: COCO,
    image_info: Dict[str, Any],
    image_path: str,
    output_path: str,
) -> List[Dict]:
    # 绘制原始图像的标注
    image_annotations = coco.imgToAnns[image_info["id"]]
    origin_image = cv2.imread(image_path)
    for annotation in image_annotations:
        # {
        #     "id": 37,
        #     "image_id": 9,
        #     "category_id": 70,
        #     "bbox": [
        #         21.358490566037688,
        #         336.9150943396226,
        #         132.54716981132077,
        #         73.58490566037739,
        #     ],
        #     "area": 9753.47098611606,
        #     "iscrowd": 0,
        # }
        bbox = annotation["bbox"]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(origin_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(output_path, origin_image)
    # 保存图像
    cv2.imwrite(output_path, origin_image)
    return image_annotations


# 推单张图
def infer_single_image(
    demo: OVDINODemo,  # 模型
    image_id: int,  # 图片ID
    image_path: str,  # 图片路径
    categories_text: str,  # 类别文本
    output_path: Optional[str] = None,  # 输出路径
    confidence_threshold: float = 0.5,  # 置信度阈值
    with_segmentation: bool = False,  # 是否使用分割
) -> Tuple[List[Dict], np.ndarray]:
    """对单张图片进行推理"""

    # 处理类别
    category_names = [
        clean_words_or_phrase(cat_name.strip())
        for cat_name in categories_text.split(",")
    ]

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return [], None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 执行推理
    predictions, visualized_output = demo.run_on_image(
        image, category_names, confidence_threshold, with_segmentation
    )

    # 获取JSON结果
    json_results = instances_to_coco_json(
        predictions["instances"].to(demo.cpu_device), image_id
    )
    for json_result in json_results:
        json_result["category_name"] = category_names[json_result["category_id"]]

    # 保存结果图像
    output_image = visualized_output.get_image()
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    return json_results, output_image


# 生成单行markdown报告
def generate_md_report(
    coco: COCO,  # COCO对象
    image_info: Dict[str, Any],  # 图像信息
    infer_output_path: str,  # 推理输出路径
    origin_output_path: str,  # 原始输出路径
    json_results: List[Dict],  # 推理结果
) -> str:
    """生成单张图片的评估报告"""
    # 获取图片原始标注
    image_id = image_info["id"]
    original_annotations = coco.imgToAnns[image_id]

    # 计算IoU函数
    def calculate_iou(box1, box2):
        """计算两个边界框的IoU"""
        # 格式: [x, y, width, height]
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算框的坐标范围
        x1_end, y1_end = x1 + w1, y1 + h1
        x2_end, y2_end = x2 + w2, y2 + h2

        # 计算交集坐标
        x_intersection = max(0, min(x1_end, x2_end) - max(x1, x2))
        y_intersection = max(0, min(y1_end, y2_end) - max(y1, y2))

        # 计算交集面积
        intersection_area = x_intersection * y_intersection

        # 计算并集面积
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        # 计算IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    # 获取预测框列表
    predicted_boxes = []
    for result in json_results:
        # 从模型预测结果中提取边界框和类别
        if "bbox" in result:
            # 直接使用result中的bbox
            box = result["bbox"]
        else:
            # 从模型预测结果中的坐标转换为[x, y, w, h]格式
            box = [
                result.get("x", 0),
                result.get("y", 0),
                result.get("width", 0),
                result.get("height", 0),
            ]

        category_id = result.get("category_id", 0)
        predicted_boxes.append(
            {"bbox": box, "category_id": category_id, "score": result.get("score", 1.0)}
        )

    # 匹配预测框和原始框
    matches = []
    used_predictions = set()

    # IoU阈值，大于该阈值认为是正确匹配
    iou_threshold = 0.5

    for orig in original_annotations:
        best_iou = iou_threshold
        best_pred_idx = -1

        orig_bbox = orig["bbox"]
        orig_category_id = orig["category_id"]

        for pred_idx, pred in enumerate(predicted_boxes):
            if pred_idx in used_predictions:
                continue

            pred_bbox = pred["bbox"]
            pred_category_id = pred["category_id"]

            iou = calculate_iou(orig_bbox, pred_bbox)

            # 判断IoU和类别是否匹配
            if iou > best_iou and orig_category_id == pred_category_id:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx != -1:
            matches.append({"iou": best_iou, "pred_idx": best_pred_idx})
            used_predictions.add(best_pred_idx)

    # 计算正确率：正确匹配的数量 / 原始框的总数
    total_annotations = len(original_annotations)
    total_correct = len(matches)
    accuracy = total_correct / total_annotations if total_annotations > 0 else 0

    # 计算正确检测的类别的平均IoU作为综合相似度
    avg_similarity = (
        sum(match["iou"] for match in matches) / len(matches) if matches else 0
    )

    # 提取相对路径，用于在markdown中显示
    rel_origin_path = os.path.relpath(
        origin_output_path, os.path.dirname(os.path.dirname(origin_output_path))
    )
    rel_infer_path = os.path.relpath(
        infer_output_path, os.path.dirname(os.path.dirname(infer_output_path))
    )

    # 生成markdown表格行
    return f"| ![原始图像]({rel_origin_path}) | ![推理图像]({rel_infer_path}) | {accuracy:.2f} ({total_correct}/{total_annotations}) | {avg_similarity:.2f} |\n"
