import pandas as pd
import os

TOP_CONFIGS_CSV = "gridsearch_output/top_configs.csv"
ALL_RESULTS_CSV = "gridsearch_output/all_results.csv"

def main():
    if not os.path.exists(TOP_CONFIGS_CSV) or not os.path.exists(ALL_RESULTS_CSV):
        print("Error: CSV files not found.")
        return

    # Load data
    top_df = pd.read_csv(TOP_CONFIGS_CSV)
    all_df = pd.read_csv(ALL_RESULTS_CSV)
    
    # Create mapping from ID to Mode
    id_to_mode = dict(zip(all_df['id'].astype(str), all_df['mode']))
    
    # Add mode column to top_df
    top_df['mode'] = top_df['original_id'].astype(str).map(id_to_mode)
    
    # Calculate stats
    total_count = len(top_df)
    layer_ea_count = len(top_df[top_df['mode'] == 'layer_ea'])
    global_ea_count = len(top_df[top_df['mode'] == 'global_ea']) # or just total - layer_ea
    
    ratio = (layer_ea_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Total Top Configs: {total_count}")
    print(f"Layer EA Models: {layer_ea_count} ({ratio:.2f}%)")
    print(f"Global EA Models: {global_ea_count} ({100-ratio:.2f}%)")
    
    print("\n--- Breakdown by Block Count ---")
    blocks = sorted(top_df['n_blocks'].unique())
    for n in blocks:
        block_df = top_df[top_df['n_blocks'] == n]
        b_total = len(block_df)
        b_layer = len(block_df[block_df['mode'] == 'layer_ea'])
        b_ratio = (b_layer / b_total) * 100 if b_total > 0 else 0
        print(f"Block {n}: {b_layer}/{b_total} ({b_ratio:.1f}%) are Layer EA")

if __name__ == "__main__":
    main()
