#!/bin/bash

# Default parameters
ARCH_PATH="./random_arch.json"
TRAIN_EPOCHS=50
TRAIN_BATCH=512

# Run training
uv run python train_cifar.py \
    --arch_path $ARCH_PATH \
    --train_epochs $TRAIN_EPOCHS \
    --train_batch $TRAIN_BATCH \
    --lr 0.2 \
    "$@"
