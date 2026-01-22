#!/bin/bash

# Run random baseline grid search for GLOBAL EA configurations
# n_blocks: 5, 10, 15, 20

for n in 5 10 15 20; do
    echo "========================================================"
    echo "Running random baseline (Global EA Ref) for n_blocks=$n..."
    echo "========================================================"
    
    uv run python run_random_baseline_search.py --n_blocks $n --target_mode global_ea --num_seeds 3
    
    echo "Finished batch for n_blocks=$n"
    echo ""
done

echo "All random baseline (Global EA Ref) batch tasks completed."
