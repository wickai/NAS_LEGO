#!/bin/bash

# Run random baseline grid search for LAYER EA configurations
# n_blocks: 5, 10, 15, 20

for n in 5 10 15 20; do
    echo "========================================================"
    echo "Running random baseline (Layer EA Ref) for n_blocks=$n..."
    echo "========================================================"
    
    uv run python run_random_baseline_search.py --n_blocks $n --target_mode layer_ea
    
    echo "Finished batch for n_blocks=$n"
    echo ""
done

echo "All random baseline (Layer EA Ref) batch tasks completed."
