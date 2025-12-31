#!/bin/bash

# Default parameters
OUTPUT_PATH="./imagenet_arch.json"
POPULATION_SIZE=20
N_GENERATIONS=50
LAYER_POPULATION=16
LAYER_GENERATIONS=8
N_BLOCKS_TO_SEARCH=20

# Run search
# Added --dataset imagenet
# Removed --n_blocks_to_search default to allow full search, or user can pass it
uv run python search_nas.py \
    --dataset imagenet \
    --population_size $POPULATION_SIZE \
    --n_generations $N_GENERATIONS \
    --layer_population $LAYER_POPULATION \
    --layer_generations $LAYER_GENERATIONS \
    --n_blocks_to_search $N_BLOCKS_TO_SEARCH \
    --output_path $OUTPUT_PATH \
    --use_pareto \
    "$@"