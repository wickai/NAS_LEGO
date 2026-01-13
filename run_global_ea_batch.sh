#!/bin/bash

# Run global_ea grid search for different n_blocks
# n_blocks: 5, 10, 15, 20

for n in 5 10 15 20; do
    echo "========================================================"
    echo "Running grid search for n_blocks=$n with global_ea..."
    echo "========================================================"
    
    uv run python run_grid_search.py --n_blocks $n --search_mode global_ea
    
    echo "Finished batch for n_blocks=$n"
    echo ""
done

echo "All global_ea batch tasks completed."
