import json
import os
import shutil  # 添加 shutil 模块用于文件复制
from datetime import datetime

from PIL import Image  # 导入PIL库用于图片处理


def clean_label(label):
    """
    清洗标签：
    1. 移除标点符号和特殊字符
    2. 处理重复词 (如"杯杯底"→"杯底")
    3. 过滤无意义标签
    """
    # 移除标点符号和特殊字符
    import re

    label = re.sub(r"[^\w\s]", "", label)

    # 处理重复词，如"杯杯底"变为"杯底"
    common_prefixes = ["杯", "盆", "桶", "瓶"]
    for prefix in common_prefixes:
        if label.startswith(prefix + prefix):
            label = prefix + label[len(prefix) * 2 :]

    # 过滤无意义标签
    if label in [
        "SS",
        "d",
        "dd",
        "ff",
        "个",
        "的",
        "疯",
        "点点点",
        "啊",
        "发",
        "瓶口",
        "瓶身",
        "瓶底",
        "包装纸",
        "瓶盖",
        "杯底嘴口",
    ]:
        return ""
    return label


def collect_unique_labels(input_dir):
    """
    遍历目录中的所有 JSON 文件，收集所有唯一的 label。
    :param input_dir: 包含图片和 JSON 文件的目录路径。
    :return: 包含唯一 label 的集合。
    """
    unique_labels = set()
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".json"):
            continue

        json_path = os.path.join(input_dir, file_name)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for shape in data.get("shapes", []):
                original_label = shape.get("label")
                if original_label:
                    # 应用标签清洗
                    cleaned_label = clean_label(original_label)
                    if cleaned_label:  # 只添加有效标签
                        unique_labels.add(cleaned_label)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    return unique_labels


def translate_coco_labels(coco_data, translate_map):
    """
    将 COCO 数据中的["categories"]翻译为中文，并处理重复类别。
    :param coco_data: COCO 数据结构。
    :param translate_map: 类别翻译映射字典。
    :return: 翻译后的 COCO 数据结构。
    """
    # 第一步：翻译所有类别
    for category in coco_data["categories"]:
        category["name"] = translate_map.get(category["name"], category["name"]).lower()

    # 第二步：检测重复类别并创建映射
    translated_names = {}  # 翻译后名称 -> [原始类别ID列表]
    for category in coco_data["categories"]:
        name = category["name"]
        if name not in translated_names:
            translated_names[name] = []
        translated_names[name].append(category["id"])

    # 有重复的类别名称
    has_duplicates = any(len(ids) > 1 for ids in translated_names.values())

    if has_duplicates:
        print(f"检测到重复类别名称，开始整合...")

        # 第三步：创建旧ID到新ID的映射
        old_to_new_id = {}
        new_categories = []

        for new_id, (name, old_ids) in enumerate(translated_names.items()):
            # 为每个唯一的翻译名称分配一个新ID
            for old_id in old_ids:
                old_to_new_id[old_id] = new_id

            # 创建新的类别条目
            new_categories.append(
                {
                    "id": new_id,
                    "name": name,
                    "supercategory": "object",  # 使用默认supercategory
                }
            )

            if len(old_ids) > 1:
                print(f"  合并类别: '{name}' - 原始ID {old_ids} -> 新ID {new_id}")

        # 第四步：更新annotations中的category_id
        for annotation in coco_data["annotations"]:
            old_category_id = annotation["category_id"]
            annotation["category_id"] = old_to_new_id[old_category_id]

        # 第五步：替换类别列表
        coco_data["categories"] = new_categories

        print(
            f"类别整合完成: 原有{len(old_to_new_id)}个类别 -> 现有{len(new_categories)}个类别"
        )

    return coco_data


def process_folder_to_coco(
    folder_path,
    global_coco_data,
    categories,
    image_id_start,
    annotation_id_start,
    image_output_dir=None,
    input_root_dir=None,
):
    """
    处理单个文件夹，并将数据添加到全局 COCO 数据结构中。

    :param folder_path: 文件夹路径。
    :param global_coco_data: 全局 COCO 数据结构。
    :param categories: 类别映射字典。
    :param image_id_start: 图片 ID 起始值。
    :param annotation_id_start: 标注 ID 起始值。
    :param image_output_dir: 图片输出目录，如果提供则复制图片到此目录。
    :param input_root_dir: 输入的根目录，用于计算相对路径。
    :return: 更新后的 image_id 和 annotation_id。
    """
    image_id = image_id_start
    annotation_id = annotation_id_start

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".json"):
            continue

        # 读取 JSON 文件
        json_path = os.path.join(folder_path, file_name)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            continue

        # 获取图片路径
        base_name = file_name.rsplit(".", 1)[0]  # 去掉扩展名
        image_jpg_path = base_name + ".jpg"
        image_png_path = base_name + ".png"

        if os.path.exists(os.path.join(folder_path, image_jpg_path)):
            image_path = image_jpg_path
        elif os.path.exists(os.path.join(folder_path, image_png_path)):
            image_path = image_png_path
        else:
            print(f"No matching .jpg or .png file found for {file_name}")
            continue

        image_full_path = os.path.join(folder_path, image_path)

        # 确保图片文件确实存在
        assert os.path.exists(
            image_full_path
        ), f"Image file does not exist: {image_full_path}"

        # 从图片文件中直接获取尺寸信息
        try:
            with Image.open(image_full_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            print(f"Error reading image {image_full_path}: {e}")
            # 如果图片读取失败，尝试使用JSON中的尺寸作为备选
            image_height = data.get("imageHeight")
            image_width = data.get("imageWidth")
            if not image_height or not image_width:
                print(f"Cannot determine image dimensions for {image_full_path}")
                continue

        # 获取相对路径，用于COCO数据
        relative_path = os.path.relpath(
            image_full_path, start=os.path.dirname(folder_path)
        )

        # 如果提供了输出目录和输入根目录，复制图片到该目录并保留文件夹结构
        if image_output_dir and input_root_dir:
            # 计算图片相对于输入根目录的路径
            rel_path = os.path.relpath(image_full_path, input_root_dir)

            # 构建目标路径 (保留文件夹结构)
            dest_image_path = os.path.join(image_output_dir, rel_path)

            # 确保目标目录存在
            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)

            # 复制图片文件
            shutil.copy2(image_full_path, dest_image_path)

            # 更新相对路径用于COCO数据
            relative_path = rel_path

        # 添加图片信息到 COCO 数据
        global_coco_data["images"].append(
            {
                "id": image_id,
                "file_name": relative_path,  # 使用相对路径
                "height": image_height,
                "width": image_width,
            }
        )

        # 处理标注信息
        for shape in data.get("shapes", []):
            original_label = shape.get("label")
            points = shape.get("points")
            if not original_label or not points or len(points) != 2:
                continue

            # 应用标签清洗
            label = clean_label(original_label)
            if not label:  # 跳过无效标签
                continue

            # 获取类别 ID
            category_id = categories.get(label)
            if category_id is None:  # 如果是新标签，添加到类别中
                category_id = len(categories)
                categories[label] = category_id
                global_coco_data["categories"].append(
                    {"id": category_id, "name": label, "supercategory": "object"}
                )

            # 计算边界框坐标
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            width = abs(points[1][0] - points[0][0])
            height = abs(points[1][1] - points[0][1])

            # 添加标注信息到 COCO 数据
            global_coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        # 更新计数器
        image_id += 1

    return image_id, annotation_id


if __name__ == "__main__":
    # 替换为您的输入目录
    input_directory = "/home/ghc/GLIP/DATASET/coco/train2017/data13"
    # 输出的 COCO JSON 文件名
    output_json_file = "/home/mappland/project/OV-DINO/datas/all_data/coco.json"
    # 图片输出目录
    image_output_directory = "/home/mappland/project/OV-DINO/datas/all_data/img"
    # 需要筛选的文件夹前缀
    folder_prefixes = [
        "常",
        "砂",
        "蒸",
        "刀",
        "板",
        "叉",
        "盒",
        "杯",
        "饮",
        "裤",
        "宠",
        "化",
        "冰",
        "盆",
        "胶",
        "剪",
        "铲",
        "方",
        "洗",
        "厨",
        "玻",
        "袋",
        "笔",
        "圆",
        "瓶",
        "桶",
        "架",
        "游",
        "被",
        "聚",
        "插",
        "移",
        "门",
        "健",
        "包",
        "压",
        "电",
        "尺",
        "落",
        "锅",
        "灯",
        "触",
        "挂",
        "开",
        "空",
        "指",
        "抽",
        "不",
        "水",
        "手",
        "个",
        "V",
        "罐",
    ]
    # 筛选文件夹
    folder_list = [
        folder_name
        for folder_name in os.listdir(input_directory)
        if any(folder_name.startswith(prefix) for prefix in folder_prefixes)
    ]

    print(folder_list)

    # 翻译映射文件路径 dict[str:str]
    # 例如：{"label": "标签"}
    translate_map_fille = "/home/mappland/project/OV-DINO/translate_map.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    os.makedirs(image_output_directory, exist_ok=True)

    # 初始化全局 COCO 数据结构
    global_coco_data = {
        "info": {
            "description": "COCO Format Dataset with All Labels",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Generated by Python Script",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # 动态收集唯一的 label
    all_unique_labels = set()
    for folder_name in folder_list:
        folder_path = os.path.join(input_directory, folder_name)
        all_unique_labels.update(collect_unique_labels(folder_path))

    # 定义类别映射（从 0 开始编号）
    categories = {}
    for category_id, label in enumerate(sorted(all_unique_labels)):
        categories[label] = category_id
        global_coco_data["categories"].append(
            {"id": category_id, "name": label, "supercategory": "object"}
        )

    # 初始化计数器
    image_id_start = 0
    annotation_id_start = 0

    # 遍历筛选出的文件夹并处理
    for folder_name in folder_list:
        folder_path = os.path.join(input_directory, folder_name)
        print(f"Processing folder: {folder_path}")
        image_id_start, annotation_id_start = process_folder_to_coco(
            folder_path=folder_path,
            global_coco_data=global_coco_data,
            categories=categories,
            image_id_start=image_id_start,
            annotation_id_start=annotation_id_start,
            image_output_dir=image_output_directory,  # 图片输出目录
            input_root_dir=input_directory,  # 添加输入根目录
        )

    with open(translate_map_fille, "r", encoding="UTF-8") as f:
        translate_map = json.load(f)

    # 翻译 COCO 数据中的标签
    global_coco_data = translate_coco_labels(global_coco_data, translate_map)

    # 写入 COCO 格式 JSON 文件
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(global_coco_data, f, ensure_ascii=False, indent=4)

    print(f"COCO format annotations with all labels saved to {output_json_file}")
    print(f"Images copied to {image_output_directory}")
    print(f"Total unique labels: {len(all_unique_labels)}")
    print(f"Labels: {sorted(all_unique_labels)}")
