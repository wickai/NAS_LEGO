import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")

CURRENT_CSV = "gridsearch_output/all_results.csv"
BACKUP_CSV = "gridsearch_output/all_results.csv.bak"
OUTPUT_IMG = "gridsearch_output/distribution_comparison.png"

def main():
    if not os.path.exists(CURRENT_CSV) or not os.path.exists(BACKUP_CSV):
        print("Error: CSV files not found.")
        return

    # Load Data
    current_df = pd.read_csv(CURRENT_CSV)
    backup_df = pd.read_csv(BACKUP_CSV)

    # Filter "Trained Models"
    # Group A: Models that already had accuracy in backup
    group_a = backup_df.dropna(subset=["test_acc_top1"])
    
    # Group B: Models that have accuracy in current but NOT in backup (New ones)
    # We identify them by ID
    current_trained = current_df.dropna(subset=["test_acc_top1"])
    existing_ids = set(group_a["id"].astype(str))
    
    group_b = current_trained[~current_trained["id"].astype(str).isin(existing_ids)]
    
    print(f"Group A (Existing Models): {len(group_a)}")
    print(f"Group B (New Models): {len(group_b)}")
    
    if group_a.empty and group_b.empty:
        print("Both groups are empty. Nothing to compare.")
        return

    # Create Figure with complex layout
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 4)
    fig.suptitle("Comparison: Existing Models (Group A) vs New Models (Group B)", fontsize=20)
    
    # 1. Boxplot of Accuracy (Top Left, spanning 2 cols)
    ax_box = fig.add_subplot(gs[0, :2])
    
    # Combine for boxplot
    group_a["Group"] = "Group A (Existing)"
    group_b["Group"] = "Group B (New)"
    combined_df = pd.concat([group_a, group_b])
    
    sns.boxplot(data=combined_df, x="Group", y="test_acc_top1", palette={"Group A (Existing)": "gray", "Group B (New)": "blue"}, ax=ax_box)
    ax_box.set_title("Top-1 Accuracy Distribution")
    ax_box.set_ylabel("Top-1 Accuracy (%)")
    
    # Annotate Min/Max
    for i, group_name in enumerate(["Group A (Existing)", "Group B (New)"]):
        subset = combined_df[combined_df["Group"] == group_name]
        if not subset.empty:
            max_val = subset["test_acc_top1"].max()
            min_val = subset["test_acc_top1"].min()
            
            # Add text
            ax_box.text(i, max_val, f"Max: {max_val:.2f}%", ha='center', va='bottom', fontweight='bold')
            ax_box.text(i, min_val, f"Min: {min_val:.2f}%", ha='center', va='top', fontweight='bold')

    # 2. KDE Plots for Metrics (Top Right, 2 plots)
    metrics_kde = [
        ("fitness", "Fitness (SWAP)", gs[0, 2]),
        ("params", "Parameters", gs[0, 3])
    ]
    
    for col, title, pos in metrics_kde:
        ax = fig.add_subplot(pos)
        sns.kdeplot(data=group_a, x=col, fill=True, color="gray", alpha=0.3, label="Group A", ax=ax)
        sns.kdeplot(data=group_b, x=col, fill=True, color="blue", alpha=0.5, label="Group B", ax=ax)
        ax.set_title(f"Distribution of {title}")
        ax.legend()

    # 3. Scatter Plots: Accuracy vs Metrics (Middle Row)
    scatter_metrics = [
        ("fitness", "Fitness (SWAP)"),
        ("params", "Parameters"),
        ("flops", "FLOPs"),
        ("total_conv_fc_flops", "Conv+FC FLOPs")
    ]
    
    for i, (col, title) in enumerate(scatter_metrics):
        ax = fig.add_subplot(gs[1, i])
        sns.scatterplot(
            data=combined_df, x=col, y="test_acc_top1", 
            hue="Group", style="Group", 
            palette={"Group A (Existing)": "gray", "Group B (New)": "blue"},
            alpha=0.7, s=80, ax=ax
        )
        ax.set_title(f"Accuracy vs {title}")
        ax.legend()
        
    # 4. Correlation Matrices (Bottom Row)
    # Group A Correlation
    ax_corr_a = fig.add_subplot(gs[2, :2])
    cols_corr = ["test_acc_top1", "fitness", "params", "flops", "total_conv_fc_flops"]
    corr_a = group_a[cols_corr].corr().round(2)
    sns.heatmap(corr_a, annot=True, cmap="Greys", ax=ax_corr_a, cbar=False)
    ax_corr_a.set_title("Correlation Matrix - Group A (Existing)")
    
    # Group B Correlation
    ax_corr_b = fig.add_subplot(gs[2, 2:])
    corr_b = group_b[cols_corr].corr().round(2)
    sns.heatmap(corr_b, annot=True, cmap="Blues", ax=ax_corr_b, cbar=False)
    ax_corr_b.set_title("Correlation Matrix - Group B (New)")

    plt.tight_layout()
    output_img_incremental = "gridsearch_output/distribution_comparison_incremental_v2.png"
    plt.savefig(output_img_incremental)
    print(f"\nComparison plot saved to {output_img_incremental}")

if __name__ == "__main__":
    main()
