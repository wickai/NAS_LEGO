import os
import json
import re
import pandas as pd
import hashlib

OUTPUT_DIR = "gridsearch_output"
MASTER_CSV = "all_results.csv"
TOP_CONFIGS_CSV = "top_configs.csv"

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

def load_all_data(directory):
    data = []
    files = os.listdir(directory)
    
    for f in files:
        if not f.endswith(".json"):
            continue
            
        nblk, pop, gen, mode = parse_filename(f)
        if nblk is None:
            print(f"Skipping {f}: Could not parse filename")
            continue
            
        path = os.path.join(directory, f)
        try:
            with open(path, "r") as json_file:
                content = json.load(json_file)
                
                # Extract metrics
                fitness = content.get("fitness", 0)
                params_mb = content.get("params_mb", 0)
                params = content.get("params", 0)
                flops = content.get("flops", 0)
                total_conv_fc_flops = content.get("total_conv_fc_flops", 0)
                acts = content.get("acts", 0)
                
                # Extract codes (convert to string for CSV storage)
                op_codes = str(content.get("op_codes", content.get("op_codes_prefix", [])))
                width_codes = str(content.get("width_codes", []))
                
                # Generate a deterministic ID based on filename and content
                # This ensures consistent IDs across different machines
                unique_str = f"{f}_{op_codes}_{width_codes}"
                run_id = hashlib.md5(unique_str.encode()).hexdigest()[:8]
                
                data.append({
                    "id": run_id,
                    "filename": f,
                    "n_blocks": nblk,
                    "population": pop,
                    "generations": gen,
                    "mode": mode,
                    "fitness": fitness,
                    "params_mb": params_mb,
                    "params": params,
                    "flops": flops,
                    "total_conv_fc_flops": total_conv_fc_flops,
                    "acts": acts,
                    "op_codes": op_codes,
                    "width_codes": width_codes
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.DataFrame(data)

def get_top_configs(df):
    top_configs = []
    
    # Define metrics for ranking
    # Metric Name, Ascending (False=Higher is better, True=Lower is better), Description
    metrics = [
        ("fitness", False, "Fitness"),
        ("efficiency_params", False, "Efficiency1 (Fitness/Params)"),
        ("efficiency_flops", False, "Efficiency2 (Fitness/FLOPs)"),
        ("efficiency_conv_flops", False, "Efficiency3 (Fitness/Conv+FC FLOPs)")
    ]
    
    # Pre-calculate efficiency metrics
    # Use small epsilon to avoid division by zero
    eps = 1e-9
    df["efficiency_params"] = df["fitness"] / (df["params"].replace(0, eps))
    df["efficiency_flops"] = df["fitness"] / (df["flops"].replace(0, eps))
    df["efficiency_conv_flops"] = df["fitness"] / (df["total_conv_fc_flops"].replace(0, eps))
    
    blocks = sorted(df["n_blocks"].unique())
    
    for n in blocks:
        block_df = df[df["n_blocks"] == n]
        
        for metric_col, ascending, description in metrics:
            sorted_df = block_df.sort_values(metric_col, ascending=ascending)
            top_5 = sorted_df.head(5)
            
            for rank, (_, row) in enumerate(top_5.iterrows(), 1):
                # Generate deterministic rank_id
                rank_str = f"{row['id']}_{description}_{rank}"
                rank_id = hashlib.md5(rank_str.encode()).hexdigest()[:8]
                
                top_configs.append({
                    "rank_id": rank_id,
                    "original_id": row["id"],
                    "n_blocks": n,
                    "ranking_metric": description,
                    "rank": rank,
                    "metric_value": row[metric_col],
                    "fitness": row["fitness"],
                    "params": row["params"],
                    "flops": row["flops"],
                    "op_codes": row["op_codes"],
                    "width_codes": row["width_codes"]
                })
                
    return pd.DataFrame(top_configs)

def main():
    print(f"Scanning {OUTPUT_DIR}...")
    df = load_all_data(OUTPUT_DIR)
    
    if df.empty:
        print("No data found.")
        return

    # Save Master CSV
    master_csv_path = os.path.join(OUTPUT_DIR, MASTER_CSV)
    df.to_csv(master_csv_path, index=False)
    print(f"Saved master CSV to {master_csv_path} ({len(df)} records)")
    
    # Generate and Save Top Configs CSV
    top_df = get_top_configs(df)
    top_csv_path = os.path.join(OUTPUT_DIR, TOP_CONFIGS_CSV)
    top_df.to_csv(top_csv_path, index=False)
    print(f"Saved top configs CSV to {top_csv_path} ({len(top_df)} records)")

if __name__ == "__main__":
    main()
