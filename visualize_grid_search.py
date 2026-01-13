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
    # Expected format: search_nblk{}_lpop{}_lgen{}.json
    match = re.match(r"search_nblk(\d+)_lpop(\d+)_lgen(\d+)\.json", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

def load_data(directory, target_nblk=5):
    data = []
    files = os.listdir(directory)
    for f in files:
        if not f.endswith(".json"):
            continue
            
        nblk, lpop, lgen = parse_filename(f)
        if nblk is None or nblk != target_nblk:
            continue
            
        path = os.path.join(directory, f)
        try:
            with open(path, "r") as json_file:
                content = json.load(json_file)
                fitness = content.get("fitness", 0)
                params = content.get("params_mb", 0)
                
                data.append({
                    "N Blocks": nblk,
                    "Layer Population": lpop,
                    "Layer Generations": lgen,
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

    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(f"Grid Search Analysis (N_BLOCKS={nblk})", fontsize=16)

    # 1. Heatmap of Fitness
    ax1 = fig.add_subplot(gs[0, 0])
    pivot_fitness = df.pivot(index="Layer Population", columns="Layer Generations", values="Fitness")
    sns.heatmap(pivot_fitness, annot=True, fmt=".0f", cmap="viridis", ax=ax1)
    ax1.set_title("Fitness vs Population & Generations")

    # 2. Heatmap of Params
    ax2 = fig.add_subplot(gs[0, 1])
    pivot_params = df.pivot(index="Layer Population", columns="Layer Generations", values="Params (MB)")
    sns.heatmap(pivot_params, annot=True, fmt=".4f", cmap="magma_r", ax=ax2) 
    ax2.set_title("Params (MB) vs Population & Generations")

    # 3. Scatter Plot: Fitness vs Params
    ax3 = fig.add_subplot(gs[1, :])
    sns.scatterplot(
        data=df, 
        x="Params (MB)", 
        y="Fitness", 
        hue="Layer Generations", 
        size="Layer Population",
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
            f"P{int(row['Layer Population'])}G{int(row['Layer Generations'])}",
            fontsize=9,
            ha='right'
        )

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"analysis_summary_nblk{nblk}.png")
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
    args = parser.parse_args()
    
    target_nblk = args.n_blocks
    df = load_data(OUTPUT_DIR, target_nblk=target_nblk)
    visualize(df, nblk=target_nblk)
