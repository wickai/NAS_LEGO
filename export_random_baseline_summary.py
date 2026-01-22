import os
import json
import re
import pandas as pd
import hashlib

OUTPUT_DIR = "gridsearch_output"
RANDOM_RESULTS_CSV = "random_baseline_results.csv"

def parse_filename(filename):
    # Expected format: random_run_nblk{}_samples{}_seed{}.json
    # Example: random_run_nblk5_samples100_seed0.json
    match = re.match(r"random_run_nblk(\d+)_samples(\d+)_seed(\d+)\.json", filename)
    if match:
        nblk = int(match.group(1))
        samples = int(match.group(2))
        seed = int(match.group(3))
        return nblk, samples, seed
    return None, None, None

def load_random_data(directory):
    data = []
    files = os.listdir(directory)
    
    print(f"Scanning {directory} for random_run_*.json files...")
    
    for f in files:
        if not f.endswith(".json") or not f.startswith("random_run_"):
            continue
            
        nblk, samples, seed = parse_filename(f)
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
                
                # Extract codes
                op_codes = str(content.get("op_codes", content.get("op_codes_prefix", [])))
                width_codes = str(content.get("width_codes", []))
                
                # Generate deterministic ID
                unique_str = f"{f}_{op_codes}_{width_codes}"
                run_id = hashlib.md5(unique_str.encode()).hexdigest()[:8]
                
                data.append({
                    "id": run_id,
                    "filename": f,
                    "n_blocks": nblk,
                    "samples": samples,
                    "seed": seed,
                    "mode": "random", # Hardcoded for this script
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

def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Directory {OUTPUT_DIR} not found.")
        return

    df = load_random_data(OUTPUT_DIR)
    
    if df.empty:
        print("No random baseline data found.")
        return

    # Sort for readability
    df = df.sort_values(by=["n_blocks", "samples", "seed"])

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, RANDOM_RESULTS_CSV)
    df.to_csv(csv_path, index=False)
    print(f"Saved random baseline results to {csv_path} ({len(df)} records)")

if __name__ == "__main__":
    main()
