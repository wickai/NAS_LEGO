#!/bin/bash

# Default parameters
ARCH_PATH="./gridsearch_output/search_nblk5_pop48_gen300__global_ea.json"
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
    # Simple check for log_name in args to avoid overwriting if user passes it directly
    if [[ "$arg" == "--log_name"* ]]; then
         # User provided log_name, so we don't set default here (or we parse it properly)
         # But simpler: just pass LOG_NAME as default arg to script, and "$@" overrides it.
         pass
    fi
done

# If distributed, use torchrun
if [ "$DISTRIBUTED" = true ]; then
    export CUDA_VISIBLE_DEVICES=1,2,3,4
    NUM_GPUS=4
    echo "Running Distributed Training on GPUs $CUDA_VISIBLE_DEVICES ($NUM_GPUS cards)..."
    
    uv run torchrun --nproc_per_node=$NUM_GPUS train_cifar.py \
        --distributed \
        --arch_path $ARCH_PATH \
        --train_epochs $TRAIN_EPOCHS \
        --train_batch $TRAIN_BATCH \
        --lr $LR \
        --log_name $LOG_NAME \
        "$@"
else
    # Single card training
    echo "Running Single Card Training..."
    uv run python train_cifar.py \
        --arch_path $ARCH_PATH \
        --train_epochs $TRAIN_EPOCHS \
        --train_batch $TRAIN_BATCH \
        --lr $LR \
        --log_name $LOG_NAME \
        "$@"
fi