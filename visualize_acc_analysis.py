import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_theme(style="whitegrid")

RESULTS_CSV = "gridsearch_output/all_results.csv"
OUTPUT_DIR = "gridsearch_output"
PLOT_FILENAME = "acc_analysis_summary.png"

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Filter only rows with valid test_acc_top1
    df = df.dropna(subset=["test_acc_top1"])
    
    # Convert metrics for plotting
    df["Params (MB)"] = df["params"] / 1e6
    df["FLOPs (M)"] = df["flops"] / 1e6
    df["Conv+FC FLOPs (M)"] = df["total_conv_fc_flops"] / 1e6
    
    return df

def visualize_acc(df):
    if df is None or df.empty:
        print("No valid data found for visualization.")
        return

    # Create a figure with subplots
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3)
    fig.suptitle("Top-1 Accuracy Analysis", fontsize=20)

    # 1. Scatter: Accuracy vs Fitness (SWAP Score)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(
        data=df, x="fitness", y="test_acc_top1", 
        hue="mode", style="n_blocks", s=100, palette="deep", ax=ax1
    )
    ax1.set_title("Accuracy vs Fitness (SWAP Score)")
    ax1.set_xlabel("Fitness (SWAP)")
    ax1.set_ylabel("Top-1 Accuracy (%)")

    # 2. Scatter: Accuracy vs Params
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(
        data=df, x="Params (MB)", y="test_acc_top1", 
        hue="mode", style="n_blocks", s=100, palette="deep", ax=ax2
    )
    ax2.set_title("Accuracy vs Params (MB)")
    ax2.set_xlabel("Params (MB)")

    # 3. Scatter: Accuracy vs FLOPs
    ax3 = fig.add_subplot(gs[0, 2])
    sns.scatterplot(
        data=df, x="FLOPs (M)", y="test_acc_top1", 
        hue="mode", style="n_blocks", s=100, palette="deep", ax=ax3
    )
    ax3.set_title("Accuracy vs FLOPs (M)")
    ax3.set_xlabel("FLOPs (M)")

    # 4. Boxplot: Accuracy Distribution by Mode and Block Count
    ax4 = fig.add_subplot(gs[1, :])
    sns.boxplot(data=df, x="n_blocks", y="test_acc_top1", hue="mode", palette="Set2", ax=ax4)
    ax4.set_title("Accuracy Distribution by Block Count & Search Mode")
    ax4.set_ylabel("Top-1 Accuracy (%)")

    # 5. Scatter: Accuracy vs Conv+FC FLOPs (Efficiency focus)
    ax5 = fig.add_subplot(gs[2, 0])
    sns.scatterplot(
        data=df, x="Conv+FC FLOPs (M)", y="test_acc_top1", 
        hue="mode", style="n_blocks", s=100, palette="deep", ax=ax5
    )
    ax5.set_title("Accuracy vs Conv+FC FLOPs (M)")
    ax5.set_xlabel("Conv+FC FLOPs (M)")

    # 6. Pareto Frontier Approximation (Fitness vs Accuracy) - Optional highlight
    # Just reusing the ax1 concept but highlighting efficiency
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Calculate "Real Efficiency" = Acc / Params
    df["Acc/Params"] = df["test_acc_top1"] / df["Params (MB)"]
    sns.boxplot(data=df, x="n_blocks", y="Acc/Params", hue="mode", palette="Set3", ax=ax6)
    ax6.set_title("Real Efficiency (Acc / Params) by Block Count")

    # 7. Correlation Matrix with Accuracy (Split by Mode)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    metrics = ["test_acc_top1", "fitness", "params", "flops", "total_conv_fc_flops"]
    
    # Calculate correlations separately
    modes = ["global_ea", "layer_ea"]
    y_offset = 0.95
    
    ax7.set_title("Correlation Matrix (Top: Global, Bottom: Layer)")
    
    for mode in modes:
        mode_df = df[df["mode"] == mode][metrics].dropna()
        if not mode_df.empty:
            corr = mode_df.corr().round(2)
            
            # Create a simplified table string or small heatmap manually isn't easy in one subplot
            # So we use two small tables stacked
            
            # Add mode label
            ax7.text(0.5, y_offset, f"{mode} Correlations", ha="center", va="center", fontsize=12, fontweight="bold")
            y_offset -= 0.05
            
            # Draw table
            table_data = []
            for i, row in corr.iterrows():
                table_data.append(row.values)
            
            # Adjust table location based on mode
            # bbox: [x, y, width, height]
            if mode == "global_ea":
                bbox = [0.1, 0.55, 0.8, 0.35]
            else:
                bbox = [0.1, 0.05, 0.8, 0.35]
                
            table = ax7.table(
                cellText=table_data, 
                rowLabels=corr.index, 
                colLabels=corr.columns, 
                loc='center', 
                bbox=bbox
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            
            y_offset -= 0.45 # Move down for next table

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, PLOT_FILENAME)
    plt.savefig(output_path)
    print(f"Accuracy analysis plot saved to {output_path}")
    
    # Print summary stats
    print("\n=== Summary Statistics ===")
    print(df.groupby(["mode", "n_blocks"])["test_acc_top1"].describe())

if __name__ == "__main__":
    df = load_data(RESULTS_CSV)
    visualize_acc(df)
