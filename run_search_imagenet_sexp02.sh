#!/bin/bash
# sexp02 返现上一把加了帕累托计算，这次删了
# 重新跑

# Output parameters
OUTPUT_PATH="./output_arch/imagenet_arch_20blk_sexp02.json"
LOG_NAME="nas_search_20blk_sexp02.log"

# Search parameters
POPULATION_SIZE=20
N_GENERATIONS=50
LAYER_POPULATION=16
LAYER_GENERATIONS=8
N_BLOCKS_TO_SEARCH=20

# Data parameters
# DATA_ROOT="/mnt/sda/weizixiang/wk/data/imagenet_v2/datasets--imagenet-1k/data/"
# DATA_FORMAT="parquet"
# daigua_4card_machine
DATA_FORMAT="arrow"
DATA_ROOT="/data/wk/kai/data/imagenet_arrow"

# Run search
# Added --dataset imagenet
# Removed --n_blocks_to_search default to allow full search, or user can pass it
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 CUDA_VISIBLE_DEVICES=0 uv run python search_nas.py \
    --dataset imagenet \
    --data_root "$DATA_ROOT" \
    --data_format "$DATA_FORMAT" \
    --population_size $POPULATION_SIZE \
    --n_generations $N_GENERATIONS \
    --layer_population $LAYER_POPULATION \
    --layer_generations $LAYER_GENERATIONS \
    --n_blocks_to_search $N_BLOCKS_TO_SEARCH \
    --output_path $OUTPUT_PATH \
    --log_name $LOG_NAME \
    "$@"