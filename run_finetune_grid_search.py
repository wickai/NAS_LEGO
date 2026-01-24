#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import itertools
import time
import re
import pandas as pd

# Grid Search Space
EPOCHS_LIST = [30, 60, 90, 120, 200, 300]
MIXUP_LIST = [0.2, -1] # -1 means disabled
LABEL_SMOOTHING_LIST = [0.1, 0]
USE_CUTOUT_LIST = [True, False]
LR_LIST = [0.2, 0.1, 0.05]

# Script to run
SCRIPT_PATH = "./run_finetune.sh"
LOG_DIR = "./logs/finetune_grid"

def get_log_name(epochs, lr, mixup, ls, cutout):
    cut_str = "on" if cutout else "off"
    # Clean up floats for filename
    return f"ft_e{epochs}_lr{lr}_mix{mixup}_ls{ls}_cut{cut_str}.log"

def parse_log_result(log_path):
    """
    Parse the log file to find the final test accuracy.
    Looking for: "Final Test Accuracy (Best Val Model): Top1=94.50%, Top5=99.80%"
    """
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
        
    match = re.search(r"Final Test Accuracy \(Best Val Model\): Top1=([\d\.]+)%", content)
    if match:
        return float(match.group(1))
    return None

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Generate all combinations
    combinations = list(itertools.product(
        EPOCHS_LIST, LR_LIST, MIXUP_LIST, LABEL_SMOOTHING_LIST, USE_CUTOUT_LIST
    ))
    
    print(f"Total combinations to run: {len(combinations)}")
    
    results = []
    
    for i, (epochs, lr, mixup, ls, cutout) in enumerate(combinations):
        log_name = get_log_name(epochs, lr, mixup, ls, cutout)
        log_path = os.path.join(LOG_DIR, log_name)
        
        print(f"\n[{i+1}/{len(combinations)}] Running: E={epochs}, LR={lr}, Mix={mixup}, LS={ls}, Cut={cutout}")
        print(f"  Log: {log_path}")
        
        # Check if already done
        existing_acc = parse_log_result(log_path)
        
        # Prepare result entry
        result_entry = {
            "epochs": epochs,
            "lr": lr,
            "mixup": mixup,
            "label_smoothing": ls,
            "cutout": cutout,
            "top1": None,
            "log_file": log_name
        }

        if existing_acc is not None:
            print(f"  -> Skipping (Found existing result: Top1={existing_acc}%)")
            result_entry["top1"] = existing_acc
            results.append(result_entry)
            
            # Update CSV incrementally even for skipped items (to ensure complete CSV)
            pd.DataFrame(results).to_csv(os.path.join(LOG_DIR, "grid_search_summary.csv"), index=False)
            continue
            
        # Set environment variables
        env = os.environ.copy()
        env["TRAIN_EPOCHS"] = str(epochs)
        env["LR"] = str(lr)
        env["MIXUP_ALPHA"] = str(mixup)
        env["LABEL_SMOOTHING"] = str(ls)
        env["USE_CUTOUT"] = "true" if cutout else "false"
        env["LOG_NAME"] = log_name
        env["LOG_PATH"] = LOG_DIR
        
        # Run script
        try:
            subprocess.run(["bash", SCRIPT_PATH], env=env, check=True)
            
            # Parse result immediately
            acc = parse_log_result(log_path)
            if acc is not None:
                print(f"  -> Finished. Top1={acc}%")
                result_entry["top1"] = acc
            else:
                print("  -> Finished but could not parse accuracy.")
                result_entry["top1"] = -1.0
                
            results.append(result_entry)
            
            # Save Summary Incrementally
            summary_path = os.path.join(LOG_DIR, "grid_search_summary.csv")
            pd.DataFrame(results).to_csv(summary_path, index=False)
            print(f"  -> Saved progress to {summary_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"  -> Error running experiment: {e}")
            
    # Final Save
    summary_path = os.path.join(LOG_DIR, "grid_search_summary.csv")
    df = pd.DataFrame(results)
    df.to_csv(summary_path, index=False)
    print(f"\nGrid Search Complete. Summary saved to {summary_path}")
    
    # Print Top 5
    if not df.empty and 'top1' in df.columns:
        print("\nTop 5 Configurations:")
        print(df.sort_values(by="top1", ascending=False).head(5))

if __name__ == "__main__":
    main()
