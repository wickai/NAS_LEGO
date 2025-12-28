#!/bin/bash

# Default parameters
ARCH_PATH="./test_arch.json"
TRAIN_EPOCHS=200
TRAIN_BATCH=128

# Run training
uv run python train_cifar.py \
    --arch_path $ARCH_PATH \
    --train_epochs $TRAIN_EPOCHS \
    --train_batch $TRAIN_BATCH \
    --lr 0.05 \
    "$@"