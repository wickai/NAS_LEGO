#!/bin/bash

# Visualize results for n_blocks=[5, 10, 15, 20] with global_ea mode

for n in 5 10 15 20; do
    echo "========================================================"
    echo "Visualizing results for n_blocks=$n with global_ea..."
    echo "========================================================"
    
    uv run --with pandas --with matplotlib --with seaborn python visualize_grid_search.py --n_blocks $n --search_mode global_ea
    
    echo "Finished visualization for n_blocks=$n"
    echo ""
done

echo "All visualization tasks completed."
