#!/bin/bash

# Default parameters
OUTPUT_PATH="./test_arch_5.json"
POPULATION_SIZE=2
N_GENERATIONS=50
LAYER_POPULATION=16
LAYER_GENERATIONS=8
N_BLOCKS_TO_SEARCH=20

# Run search
CUDA_VISIBLE_DEVICES=0 uv run python search_nas.py \
    --population_size $POPULATION_SIZE \
    --n_generations $N_GENERATIONS \
    --layer_population $LAYER_POPULATION \
    --layer_generations $LAYER_GENERATIONS \
    --n_blocks_to_search $N_BLOCKS_TO_SEARCH \
    --output_path $OUTPUT_PATH \
    "$@"
