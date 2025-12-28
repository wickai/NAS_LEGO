#!/bin/bash

# Default parameters
OUTPUT_PATH="./test_arch.json"
POPULATION_SIZE=2
N_GENERATIONS=1
LAYER_POPULATION=2
LAYER_GENERATIONS=1
N_BLOCKS_TO_SEARCH=5

# Run search
uv run python search_nas.py \
    --population_size $POPULATION_SIZE \
    --n_generations $N_GENERATIONS \
    --layer_population $LAYER_POPULATION \
    --layer_generations $LAYER_GENERATIONS \
    --n_blocks_to_search $N_BLOCKS_TO_SEARCH \
    --output_path $OUTPUT_PATH \
    --use_pareto \
    "$@"
