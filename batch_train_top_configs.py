import os
import pandas as pd
import subprocess
import json
import re
import time
import argparse

TOP_CONFIGS_CSV = "gridsearch_output/top_configs.csv"
ALL_RESULTS_CSV = "gridsearch_output/all_results.csv"
PROGRESS_FILE = "training_progress.json"
LOG_DIR = "logs"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_ids": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=4)

def update_arch_file(filename, acc):
    filepath = os.path.join("gridsearch_output", filename)
    if not os.path.exists(filepath):
        print(f"Warning: Arch file {filepath} not found.")
        return False
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        data["test_acc_top1"] = acc
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Updated {filepath} with test_acc_top1={acc}")
        return True
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def update_master_csv(run_id, acc):
    if not os.path.exists(ALL_RESULTS_CSV):
        print(f"Error: {ALL_RESULTS_CSV} not found.")
        return False
    
    try:
        df = pd.read_csv(ALL_RESULTS_CSV)
        # Check if column exists, if not create it
        if "test_acc_top1" not in df.columns:
            df["test_acc_top1"] = None
            
        # Update row
        mask = df['id'].astype(str) == str(run_id)
        if mask.any():
            df.loc[mask, "test_acc_top1"] = acc
            df.to_csv(ALL_RESULTS_CSV, index=False)
            print(f"Updated {ALL_RESULTS_CSV} for ID {run_id}")
            return True
        else:
            print(f"Warning: ID {run_id} not found in master CSV.")
            return False
    except Exception as e:
        print(f"Error updating master CSV: {e}")
        return False

def parse_accuracy_from_log(log_path):
    if not os.path.exists(log_path):
        return None
    
    # Read the log file
    # We look for "Final Accuracy of Best Model (Top-1): XX.XX%"
    # Or "Final Test Accuracy (Best Val Model): Top1=XX.XX%"
    
    final_acc = None
    try:
        with open(log_path, "r") as f:
            for line in f:
                # Pattern 1: Final Accuracy of Best Model (Top-1): 72.33%
                m1 = re.search(r"Final Accuracy of Best Model \(Top-1\):\s+([\d\.]+)", line)
                if m1:
                    final_acc = float(m1.group(1))
                
                # Pattern 2: Final Test Accuracy (Best Val Model): Top1=72.33%
                m2 = re.search(r"Final Test Accuracy \(Best Val Model\): Top1=([\d\.]+)", line)
                if m2:
                    final_acc = float(m2.group(1))
                    
    except Exception as e:
        print(f"Error reading log {log_path}: {e}")
        
    return final_acc

def main():
    if not os.path.exists(TOP_CONFIGS_CSV):
        print(f"Error: {TOP_CONFIGS_CSV} not found.")
        return

    # Load top configs
    top_df = pd.read_csv(TOP_CONFIGS_CSV)
    
    # Load master csv to get filenames (needed for JSON update)
    if os.path.exists(ALL_RESULTS_CSV):
        master_df = pd.read_csv(ALL_RESULTS_CSV)
        # Create a mapping from ID to filename
        id_to_filename = dict(zip(master_df['id'].astype(str), master_df['filename']))
    else:
        print("Error: Master CSV not found, cannot map IDs to filenames.")
        return

    progress = load_progress()
    completed_ids = set(progress["completed_ids"])
    
    print(f"Found {len(top_df)} configs to train. {len(completed_ids)} already completed.")
    
    for idx, row in top_df.iterrows():
        original_id = str(row['original_id'])
        rank_id = str(row['rank_id'])
        
        # Use rank_id as the unique key for progress tracking
        # because one original_id might appear multiple times (e.g. top in fitness AND top in efficiency)
        # BUT we only need to train it once per original_id.
        # So actually we should check original_id in completed_ids.
        
        if original_id in completed_ids:
            print(f"Skipping {original_id} (already completed)")
            continue
            
        print(f"\n[{idx+1}/{len(top_df)}] Starting training for ID: {original_id}")
        print(f"Config: Block={row['n_blocks']}, Metric={row['ranking_metric']}, Rank={row['rank']}")
        
        # Construct log name
        log_name = f"train_{original_id}.log"
        log_path = os.path.join(LOG_DIR, log_name)
        
        # Run training command
        # bash run_train.sh --run_id=XXXX --log_name=XXXX
        cmd = ["bash", "run_train.sh", f"--run_id={original_id}", f"--log_name={log_name}"]
        
        try:
            # Run process and stream output
            # We assume run_train.sh handles distributed launch
            process = subprocess.run(cmd, check=True)
            
            # After training, parse log
            acc = parse_accuracy_from_log(log_path)
            
            if acc is not None:
                print(f"Training finished. Parsed Top-1 Acc: {acc}%")
                
                # Update files
                filename = id_to_filename.get(original_id)
                if filename:
                    update_arch_file(filename, acc)
                else:
                    print(f"Warning: Could not find filename for ID {original_id}")
                    
                update_master_csv(original_id, acc)
                
                # Mark as complete
                progress["completed_ids"].append(original_id)
                save_progress(progress)
                
            else:
                print(f"Error: Could not parse accuracy from log {log_path}")
                # We don't mark as complete so it can be retried
                
        except subprocess.CalledProcessError as e:
            print(f"Training failed for {original_id}. Exit code: {e.returncode}")
            # Do not mark as complete
            
        # Optional: Sleep briefly
        time.sleep(2)

if __name__ == "__main__":
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    main()
