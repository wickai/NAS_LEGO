uv run python 0708_cifar10_new_searchingspace.py   \
    --n_generations 200   \
    --population_size 40   \
    --train_epochs 400   \
    --mixup_alpha 0.5   \
    --label_smoothing 0.05   \
    --weight_decay 2e-4   \
    --log_name "exp4_aggressive_only_search.log"   \
    "$@"