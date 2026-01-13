import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

OUTPUT_DIR = "gridsearch_output"
RESULTS_FILE = "grid_search_analysis_plots.png"

def parse_filename(filename):
    # Expected format: search_nblk{}_[lpop|pop]{}_[lgen|gen]{}[__mode].json
    # Handle global_ea format: search_nblk5_pop10_gen50__global_ea.json
    match_global = re.match(r"search_nblk(\d+)_pop(\d+)_gen(\d+)(?:__(\w+))?\.json", filename)
    if match_global:
        mode = match_global.group(4) if match_global.group(4) else "default"
        return int(match_global.group(1)), int(match_global.group(2)), int(match_global.group(3)), mode
        
    # Handle layer_ea format: search_nblk5_lpop5_lgen2__layer_ea.json
    match_layer = re.match(r"search_nblk(\d+)_lpop(\d+)_lgen(\d+)(?:__(\w+))?\.json", filename)
    if match_layer:
        mode = match_layer.group(4) if match_layer.group(4) else "default"
        return int(match_layer.group(1)), int(match_layer.group(2)), int(match_layer.group(3)), mode
        
    return None, None, None, None

def load_data(directory, target_nblk=5, target_mode=None):
    data = []
    files = os.listdir(directory)
    for f in files:
        if not f.endswith(".json"):
            continue
            
        nblk, pop, gen, mode = parse_filename(f)
        if nblk is None or nblk != target_nblk:
            continue
        
        if target_mode and mode != target_mode:
            continue
            
        path = os.path.join(directory, f)
        try:
            with open(path, "r") as json_file:
                content = json.load(json_file)
                fitness = content.get("fitness", 0)
                params = content.get("params_mb", 0)
                
                data.append({
                    "N Blocks": nblk,
                    "Population": pop,
                    "Generations": gen,
                    "Mode": mode,
                    "Fitness": fitness,
                    "Params (MB)": params
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.DataFrame(data)

def visualize(df, nblk=5):
    if df.empty:
        print("No data found!")
        return
    
    # Handle duplicates
    if df.duplicated(subset=["Population", "Generations"]).any():
        print("Warning: Duplicate entries found for same Population/Generations. Averaging...")
        df = df.groupby(["Population", "Generations"]).agg({
            "Fitness": "mean",
            "Params (MB)": "mean",
            "N Blocks": "first",
            "Mode": "first"
        }).reset_index()

    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(f"Grid Search Analysis (N_BLOCKS={nblk})", fontsize=16)

    # 1. Heatmap of Fitness
    ax1 = fig.add_subplot(gs[0, 0])
    pivot_fitness = df.pivot(index="Population", columns="Generations", values="Fitness")
    sns.heatmap(pivot_fitness, annot=True, fmt=".0f", cmap="viridis", ax=ax1)
    ax1.set_title("Fitness vs Population & Generations")

    # 2. Heatmap of Params
    ax2 = fig.add_subplot(gs[0, 1])
    pivot_params = df.pivot(index="Population", columns="Generations", values="Params (MB)")
    sns.heatmap(pivot_params, annot=True, fmt=".4f", cmap="magma_r", ax=ax2) 
    ax2.set_title("Params (MB) vs Population & Generations")

    # 3. Scatter Plot: Fitness vs Params
    ax3 = fig.add_subplot(gs[1, :])
    sns.scatterplot(
        data=df, 
        x="Params (MB)", 
        y="Fitness", 
        hue="Generations", 
        size="Population",
        sizes=(50, 400),
        palette="deep",
        ax=ax3
    )
    ax3.set_title("Fitness vs Params (MB)")
    
    # Add labels to points
    for i, row in df.iterrows():
        ax3.text(
            row["Params (MB)"], 
            row["Fitness"], 
            f"P{int(row['Population'])}G{int(row['Generations'])}",
            fontsize=9,
            ha='right'
        )

    plt.tight_layout()
    # Include mode in filename if consistent, otherwise 'mixed'
    modes = df["Mode"].unique()
    mode_str = modes[0] if len(modes) == 1 else "mixed"
    output_path = os.path.join(OUTPUT_DIR, f"analysis_summary_nblk{nblk}_{mode_str}.png")
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")
    
    # Also print correlation/summary
    print("\nTop 5 Configs by Fitness:")
    print(df.sort_values("Fitness", ascending=False).head(5))

    print("\nTop 5 Configs by Efficiency (Fitness/Params - rough proxy):")
    # Avoid division by zero
    df["Efficiency"] = df["Fitness"] / df["Params (MB)"].replace(0, 1e-9)
    print(df.sort_values("Efficiency", ascending=False).head(5))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Grid Search Results")
    parser.add_argument("--n_blocks", type=int, default=5, help="Number of blocks to filter")
    parser.add_argument("--search_mode", type=str, default="global_ea", choices=["layer_ea", "global_ea"], help="Filter by search mode (e.g. global_ea)")
    args = parser.parse_args()
    
    target_nblk = args.n_blocks
    df = load_data(OUTPUT_DIR, target_nblk=target_nblk, target_mode=args.search_mode)
    visualize(df, nblk=target_nblk)
