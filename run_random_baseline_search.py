import os
import subprocess
import itertools
import threading
from queue import Queue
import time
import argparse
import shutil

# Parse arguments
parser = argparse.ArgumentParser(description="Run Random Baseline Grid Search for NAS")
parser.add_argument("--n_blocks", type=int, default=5, help="Number of blocks to search")
parser.add_argument("--target_mode", type=str, default="layer_ea", choices=["layer_ea", "global_ea"], help="Target mode to baseline against")
parser.add_argument("--num_seeds", type=int, default=3, help="Number of random seeds to run for each configuration")
args = parser.parse_args()

# Configuration mirrored from run_grid_search.py
LAYER_POPULATION_LIST = [5, 10, 16, 20, 40]
LAYER_GENERATIONS_LIST = [2, 4, 8, 12]

POPULATION_SIZE_LIST = [10, 16, 32, 48]
N_GENERATIONS_LIST = [50, 100, 200, 300]

N_BLOCKS_TO_SEARCH = args.n_blocks
TARGET_MODE = args.target_mode

# Output directory
OUTPUT_DIR = "gridsearch_output"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# GPUs available
GPUS = [2, 3, 4, 5, 6, 7]

# Task Queue
task_queue = Queue()

# Generate combinations
raw_combinations = []

if TARGET_MODE == "global_ea":
    # Global EA baseline
    base_combos = list(itertools.product(POPULATION_SIZE_LIST, N_GENERATIONS_LIST))
    for pop, gen in base_combos:
        n_samples = pop * gen
        raw_combinations.append({
            "n_samples": n_samples,
            "ref_pop": pop,
            "ref_gen": gen,
            "ref_mode": "global_ea"
        })
else:
    # Layer EA baseline
    base_combos = list(itertools.product(LAYER_POPULATION_LIST, LAYER_GENERATIONS_LIST))
    for l_pop, l_gen in base_combos:
        n_samples = l_pop * l_gen * N_BLOCKS_TO_SEARCH
        raw_combinations.append({
            "n_samples": n_samples,
            "ref_pop": l_pop,
            "ref_gen": l_gen,
            "ref_mode": "layer_ea"
        })

# Deduplicate tasks by n_samples
unique_tasks = {}
for combo in raw_combinations:
    n_samples = combo["n_samples"]
    if n_samples not in unique_tasks:
        unique_tasks[n_samples] = []
    unique_tasks[n_samples].append(combo)

print(f"Total raw combinations: {len(raw_combinations)}")
print(f"Unique sample sizes to run: {len(unique_tasks)}")

# Put unique tasks into queue
# We use the FIRST combination as the "primary" runner
# But now we need to run for each seed
for n_samples, combo_list in unique_tasks.items():
    primary_combo = combo_list[0]
    # Attach the full list of aliases (including the primary itself) to the task
    primary_combo["aliases"] = combo_list
    
    # Add seed tasks
    for seed in range(args.num_seeds):
        # Create a copy of the task for this seed
        seed_task = primary_combo.copy()
        seed_task["seed"] = seed + 42 # Offset seed to be deterministic but different (42, 43, 44...)
        seed_task["seed_idx"] = seed
        task_queue.put(seed_task)

def worker(gpu_id):
    while True:
        try:
            if task_queue.empty():
                break
            
            task = task_queue.get_nowait()
            n_samples = task["n_samples"]
            aliases = task["aliases"]
            seed = task["seed"]
            seed_idx = task["seed_idx"]
            
            # Primary output filename (based on the primary combo attributes)
            # We construct a canonical filename for the actual run: e.g. random_run_nblk5_samples100_seed0.json
            canonical_filename = f"random_run_nblk{N_BLOCKS_TO_SEARCH}_samples{n_samples}_seed{seed_idx}.json"
            canonical_path = os.path.join(OUTPUT_DIR, canonical_filename)
            
        except Exception:
            break
            
        os.makedirs(os.path.dirname(canonical_path), exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Check if canonical run exists
        run_needed = True
        if os.path.exists(canonical_path):
             print(f"[GPU {gpu_id}] Skipping run (canonical exists): {canonical_filename}")
             run_needed = False
        
        if run_needed:
            print(f"[GPU {gpu_id}] Starting Unique Random Search: Samples={n_samples}, Seed={seed}")
            
            cmd = [
                "uv", "run", "python", "search_nas.py",
                "--search_mode", "random",
                "--population_size", "1",
                "--n_generations", str(n_samples),
                "--n_blocks_to_search", str(N_BLOCKS_TO_SEARCH),
                "--output_path", canonical_path,
                "--seed", str(seed),
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            log_filename = f"random_run_nblk{N_BLOCKS_TO_SEARCH}_samples{n_samples}_seed{seed_idx}.log"
            log_path = os.path.join(LOG_DIR, log_filename)

            try:
                with open(log_path, "w") as log_file:
                    subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
                print(f"[GPU {gpu_id}] Finished Unique Run: {canonical_filename}")
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu_id}] Failed Unique Run: {canonical_filename}. Check logs at {log_path}")
                task_queue.task_done()
                continue # Skip aliasing if failed

        # Now handle aliasing (copy/link canonical result to all required filenames)
        for alias in aliases:
            ref_pop = alias["ref_pop"]
            ref_gen = alias["ref_gen"]
            ref_mode = alias["ref_mode"]
            
            target_filename = f"random_baseline_nblk{N_BLOCKS_TO_SEARCH}_samples{n_samples}_ref_{ref_mode}_p{ref_pop}_g{ref_gen}_seed{seed_idx}.json"
            target_path = os.path.join(OUTPUT_DIR, target_filename)
            
            if not os.path.exists(target_path):
                try:
                    shutil.copy2(canonical_path, target_path)
                    # print(f"[GPU {gpu_id}] Created alias: {target_filename}")
                except Exception as e:
                    print(f"[GPU {gpu_id}] Failed to copy alias {target_filename}: {e}")
            
        task_queue.task_done()

# Start Worker Threads
threads = []
for gpu in GPUS:
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("All random baseline tasks completed.")
