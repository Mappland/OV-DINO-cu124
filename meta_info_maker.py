import json

def generate_meta_info(coco_json_path):
    """
    读取 COCO 格式的 JSON 文件，并生成 meta_info 配置。
    
    Args:
        coco_json_path (str): COCO 格式数据集的 JSON 文件路径。
    
    Returns:
        dict: 包含 thing_dataset_id_to_contiguous_id 和 thing_classes 的 meta_info 字典。
    """
    # 读取 COCO 格式的 JSON 文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 提取类别信息
    categories = coco_data.get("categories", [])
    if not categories:
        raise ValueError("JSON 文件中未找到 'categories' 字段，请检查文件格式是否正确。")
    
    # 初始化 meta_info
    meta_info = {
        "thing_dataset_id_to_contiguous_id": {},
        "thing_classes": []
    }
    
    # 遍历类别信息，填充 meta_info
    for idx, category in enumerate(categories):
        dataset_id = category["id"]  # 数据集中的原始类别 ID
        category_name = category["name"]  # 类别名称
        
        # 填充映射关系：dataset_id -> contiguous_id
        meta_info["thing_dataset_id_to_contiguous_id"][dataset_id] = idx
        
        # 添加类别名称到 thing_classes 列表
        meta_info["thing_classes"].append(category_name)
    
    return meta_info


def register_custom_datasets(meta_info, train_annotations, train_images, val_annotations, val_images):
    """
    注册自定义的训练和验证数据集。
    
    Args:
        meta_info (dict): 包含 thing_dataset_id_to_contiguous_id 和 thing_classes 的元信息。
        train_annotations (str): 训练集标注文件路径。
        train_images (str): 训练集图像根目录路径。
        val_annotations (str): 验证集标注文件路径。
        val_images (str): 验证集图像根目录路径。
    
    Returns:
        str: 动态生成的注册代码。
    """
    # 获取类别数量
    num_classes = len(meta_info["thing_classes"])
    
    # 动态生成注册代码
    code = f"""
# Case 1 (Recommend): If you follow the coco format, you need uncomment and change the following code.
# 1. Define custom_meta_info, just a example, you need to change it to your own.
meta_info = {{
    "thing_dataset_id_to_contiguous_id": {meta_info["thing_dataset_id_to_contiguous_id"]},
    "thing_classes": {meta_info["thing_classes"]}
}}

# 2. Register custom train dataset.
register_custom_ovd_instances(
    "custom_train_ovd_unipro",  # dataset_name
    meta_info,
    "{train_annotations}",  # annotations_json_file
    "{train_images}",  # image_root
    {num_classes},  # number_of_classes You also need to change model.num_classes in the ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_custom_24ep.py#L37.
    "full",  # template, default: full
)

# 3. Register custom val dataset.
register_custom_ovd_instances(
    "custom_val_ovd_unipro",
    meta_info,
    "{val_annotations}",  # annotations_json_file
    "{val_images}",  # image_root
    {num_classes},
    "full",
)
"""
    return code


if __name__ == "__main__":
    # 替换为你的 COCO JSON 文件路径
    coco_json_path = "/home/mappland/project/OV-DINO/datas/all_data/coco.json"
    
    # 训练和验证数据集的路径
    train_annotations = "/home/mappland/project/OV-DINO/datas/all_data/coco.json"
    train_images = "/home/mappland/project/OV-DINO/datas/all_data/img"
    val_annotations = "/home/mappland/project/OV-DINO/datas/all_data/coco.json"
    val_images = "/home/mappland/project/OV-DINO/datas/all_data/img"
    
    try:
        # 生成 meta_info
        meta_info = generate_meta_info(coco_json_path)
        
        # 动态生成注册代码
        generated_code = register_custom_datasets(
            meta_info,
            train_annotations,
            train_images,
            val_annotations,
            val_images
        )
        
        # 打印生成的代码
        print("生成的完整代码如下：")
        print(generated_code)
    except Exception as e:
        print(f"发生错误: {e}")