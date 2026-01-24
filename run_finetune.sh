#!/bin/bash

# Default parameters
ARCH_PATH="./gridsearch_output/search_nblk20_pop32_gen200__global_ea.json"
TRAIN_EPOCHS=${TRAIN_EPOCHS:-200}
LR=${LR:-0.05}
MIXUP_ALPHA=${MIXUP_ALPHA:-0.2}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
USE_CUTOUT=${USE_CUTOUT:-true} # true or false
CUTOUT_LENGTH=${CUTOUT_LENGTH:-16}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
LOG_NAME=${LOG_NAME:-"finetune_test.log"}
LOG_PATH=${LOG_PATH:-"./logs/finetune"}

# Calculate num GPUs
IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#ADDR[@]}

echo "Running Finetune on GPUs $CUDA_VISIBLE_DEVICES ($NUM_GPUS cards)..."
echo "  Arch: $ARCH_PATH"
echo "  Epochs: $TRAIN_EPOCHS"
echo "  LR: $LR"
echo "  Mixup: $MIXUP_ALPHA"
echo "  Label Smoothing: $LABEL_SMOOTHING"
echo "  Cutout: $USE_CUTOUT (Length: $CUTOUT_LENGTH)"
echo "  Log: $LOG_PATH/$LOG_NAME"

# Construct Cutout arg
if [ "$USE_CUTOUT" = "true" ] || [ "$USE_CUTOUT" = "True" ]; then
    CUTOUT_ARG="--use_cutout --cutout_length $CUTOUT_LENGTH"
else
    CUTOUT_ARG="--no-cutout"
fi

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

uv run torchrun --nproc_per_node=$NUM_GPUS train_cifar.py \
    --distributed \
    --arch_path "$ARCH_PATH" \
    --train_epochs "$TRAIN_EPOCHS" \
    --lr "$LR" \
    --mixup_alpha "$MIXUP_ALPHA" \
    --label_smoothing "$LABEL_SMOOTHING" \
    $CUTOUT_ARG \
    --log_path "$LOG_PATH" \
    --log_name "$LOG_NAME" \
    --train_batch 128
