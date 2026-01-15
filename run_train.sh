#!/bin/bash

# Default parameters
ARCH_PATH="./gridsearch_output/search_nblk5_pop48_gen300__global_ea.json"
RUN_ID=""
# Use absolute path for CSV to avoid issues in distributed training
CSV_PATH="$(pwd)/gridsearch_output/all_results.csv"
TRAIN_EPOCHS=30
TRAIN_BATCH=128
LR=0.05
LOG_NAME="train_cifar_4card_global5-48-300_test.log"

# Check if --distributed flag is passed
DISTRIBUTED=true
NUM_GPUS=1

for arg in "$@"
do
    if [ "$arg" == "--no-distributed" ]; then
        DISTRIBUTED=false
    fi
    # Check for run_id argument
    if [[ "$arg" == "--run_id="* ]]; then
        RUN_ID="${arg#*=}"
    elif [ "$arg" == "--run_id" ]; then
        # This is tricky in simple loop, usually needs while loop with shift
        # But for now, we assume user passes --run_id=XXXX
        pass
    fi
done

# Helper function to construct arguments
construct_args() {
    local args="--train_epochs $TRAIN_EPOCHS --train_batch $TRAIN_BATCH --lr $LR --log_name $LOG_NAME"
    if [ -n "$RUN_ID" ]; then
        args="$args --run_id $RUN_ID --csv_path $CSV_PATH"
    else
        args="$args --arch_path $ARCH_PATH"
    fi
    echo "$args"
}

CMD_ARGS=$(construct_args)

# If distributed, use torchrun
if [ "$DISTRIBUTED" = true ]; then
    export CUDA_VISIBLE_DEVICES=1,2,3,4
    NUM_GPUS=4
    echo "Running Distributed Training on GPUs $CUDA_VISIBLE_DEVICES ($NUM_GPUS cards)..."
    if [ -n "$RUN_ID" ]; then
        if [ ! -f "$CSV_PATH" ]; then
            echo "CSV file not found at $CSV_PATH. Attempting to generate it..."
            uv run python export_grid_search_summary.py
        fi
        echo "Using Run ID: $RUN_ID from $CSV_PATH"
    else
        echo "Using Arch Path: $ARCH_PATH"
    fi
    
    uv run torchrun --nproc_per_node=$NUM_GPUS train_cifar.py \
        --distributed \
        $CMD_ARGS \
        "$@"
else
    # Single card training
    echo "Running Single Card Training..."
    if [ -n "$RUN_ID" ]; then
        if [ ! -f "$CSV_PATH" ]; then
            echo "CSV file not found at $CSV_PATH. Attempting to generate it..."
            uv run python export_grid_search_summary.py
        fi
        echo "Using Run ID: $RUN_ID from $CSV_PATH"
    else
        echo "Using Arch Path: $ARCH_PATH"
    fi

    uv run python train_cifar.py \
        $CMD_ARGS \
        "$@"
fi