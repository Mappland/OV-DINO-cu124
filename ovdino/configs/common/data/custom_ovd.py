import itertools

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detrex.data import DetrDatasetMapper
from detrex.data.datasets import register_custom_ovd_instances
from omegaconf import OmegaConf

dataloader = OmegaConf.create()

# Case 0: If you want to define it by yourself, you can change it on ovdino/detrex/data/datasets/custom_ovd.py, you need to uncomment the code (ovdino/detrex/data/datasets/__init__.py L21) first.
# Case 1 (Recommend): If you follow the coco format, you need uncomment and change the following code.
# 1. Define custom_meta_info, just a example, you need to change it to your own.
meta_info = {
    "thing_dataset_id_to_contiguous_id": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60, 61: 61, 62: 62, 63: 63, 64: 64, 65: 65, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70, 71: 71, 72: 72, 73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80, 81: 81, 82: 82, 83: 83, 84: 84, 85: 85, 86: 86, 87: 87, 88: 88, 89: 89, 90: 90, 91: 91, 92: 92, 93: 93, 94: 94, 95: 95, 96: 96, 97: 97, 98: 98, 99: 99, 100: 100, 101: 101, 102: 102, 103: 103, 104: 104, 105: 105, 106: 106, 107: 107, 108: 108, 109: 109, 110: 110, 111: 111, 112: 112, 113: 113, 114: 114, 115: 115, 116: 116, 117: 117, 118: 118, 119: 119, 120: 120, 121: 121, 122: 122, 123: 123, 124: 124, 125: 125, 126: 126, 127: 127, 128: 128, 129: 129, 130: 130, 131: 131, 132: 132, 133: 133, 134: 134, 135: 135, 136: 136, 137: 137, 138: 138, 139: 139, 140: 140},
    "thing_classes": ['bottom_plate', 'upper_cover', 'upper_tooth_plate', 'lower_tooth_plate', 'middle_pillar', 'main_body', 'side_panel', 'side_door', 'side_view', 'refrigerator_side', 'refrigerator_front', 'blade_body', 'blade', 'knife_sheath', 'knife_tip', 'knife_cover', 'knife_end', 'knife_handle', 'knife_back', 'front_panel', 'secondary_lock', 'function_button', 'bag_top_surface', 'd_pad', 'semi_automatic_ice_maker', 'hinge', 'hanging_rod', 'hanging_chain', 'rear_panel', 'ceiling_plate', 'coffee_maker', 'trademark', 'cup_spout', 'floor_beam', 'wall_mount', 'kettle_spout', 'shell', 'head_support', 'base_plate', 'bottom_side', 'switch', 'carrying_opening', 'trigger_button', 'hand_grip', 'drawer', 'pull_handle', 'hanging_hole', 'ring', 'hooks', 'finger_ring', 'nail_file', 'indicator_light', 'press_head', 'press_plate', 'keys', 'interface', 'control_panel', 'carrying_handle', 'carrying_ring', 'cup_lid', 'plug', 'storage_shelf', 'camera', 'joystick', 'data_cable', 'knob', 'monitor', 'machine_body', 'barcode_label', 'cup_body', 'cup_bottom', 'cup_opening', 'bucket_opening', 'bucket_bottom', 'bucket_lid', 'bucket_body', 'bucket_surface', 'horizontal_board', 'horizontal_layout', 'front_side', 'crystal_lamp_decoration', 'silencer', 'sliding_button', 'slide_rail', 'sliding_track', 'sliding_knob', 'lamp_socket', 'lamp_shell', 'lamp_base', 'lamp_holder', 'lamp_pole', 'lamp_frame', 'lamp_column', 'incandescent_light_bulb', 'lampshade', 'lamp_arm', 'peephole', 'power_button', 'electric_wire', 'basin_bottom', 'basin_opening', 'basin_exterior', 'box_lid', 'cover', 'eye_mask', 'hard_drive_body', 'can_body', 'can_opening', 'can_neck', 'flip_lid', 'headphones', 'backplate', 'back_side', 'steamer', 'bag_side', 'bag_opening', 'touching_buildplate', 'adjustable_feet', 'pivot', 'edge_corner_point', 'connecting_cable', 'connecting_shaft', 'lock', 'lock_catch', 'pot_side', 'pot_opening', 'pot_bottom', 'pot_handle', 'pot_lid', 'pot_lid_handle', 'pot_ear', 'keyboard', 'lens', 'the_door', 'door_side', 'door_stopper', 'door_panel', 'door_frame', 'door_front', 'house_lock', 'top_view']
}

# 2. Register custom train dataset.
register_custom_ovd_instances(
    "custom_train_ovd_unipro",  # dataset_name
    meta_info,
    "/home/mappland/project/OV-DINO/datas/all_data/coco.json",  # annotations_json_file
    "/home/mappland/project/OV-DINO/datas/all_data/img",  # image_root
    141,  # number_of_classes You also need to change model.num_classes in the ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_custom_24ep.py#L37.
    "full",  # template, default: full
)

# 3. Register custom val dataset.
register_custom_ovd_instances(
    "custom_val_ovd_unipro",
    meta_info,
    "/home/mappland/project/OV-DINO/datas/all_data/coco.json",  # annotations_json_file
    "/home/mappland/project/OV-DINO/datas/all_data/img",  # image_root
    141,
    "full",
)
# # 4. Optional, register custom test dataset.
# register_custom_ovd_instances(
#     "custom_test_ovd",
#     meta_info,
#     "/path/to/test.json",
#     "/path/to/test/images",
#     2,
#     "full",  # choices: ["identity", "simple", "full"]
# )

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="custom_train_ovd_unipro"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
                ),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
                ),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="custom_val_ovd_unipro", filter_empty=False
    ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
