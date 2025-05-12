#!/bin/bash
cd /home/mappland/project/ov-dino-github/OV-DINO-cu124/ovdino

CUDA_VISIBLE_DEVICES=0,1,2,3,6,7 https_proxy=http://127.0.0.1:8890 http_proxy=http://127.0.0.1:8890 all_proxy=http://127.0.0.1:8890 bash scripts/finetune_debug.sh \
 projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_custom_24ep.py \
 /home/mappland/project/ov-dino-github/OV-DINO-cu124/inits/ovdino/ovdino_swint_ogc-coco50.2_lvismv40.1_lvis32.9.pth