import os
import subprocess
import itertools
import threading
from queue import Queue
import time
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Run Random Baseline Grid Search for NAS")
parser.add_argument("--n_blocks", type=int, default=5, help="Number of blocks to search")
parser.add_argument("--target_mode", type=str, default="layer_ea", choices=["layer_ea", "global_ea"], help="Target mode to baseline against")
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
GPUS = [0, 1, 2, 3, 4, 5, 6]

# Task Queue
task_queue = Queue()

# Generate combinations and calculate equivalent random samples
combinations = []

if TARGET_MODE == "global_ea":
    # Global EA baseline
    # Samples = Pop * Gen
    base_combos = list(itertools.product(POPULATION_SIZE_LIST, N_GENERATIONS_LIST))
    for pop, gen in base_combos:
        n_samples = pop * gen
        combinations.append({
            "n_samples": n_samples,
            "ref_pop": pop,
            "ref_gen": gen,
            "ref_mode": "global_ea"
        })
else:
    # Layer EA baseline
    # Samples = (Pop * Gen) * N_BLOCKS
    # Because Layer EA runs (Pop * Gen) evaluations PER BLOCK.
    base_combos = list(itertools.product(LAYER_POPULATION_LIST, LAYER_GENERATIONS_LIST))
    for l_pop, l_gen in base_combos:
        # Layer EA evals = l_pop * l_gen * n_blocks
        n_samples = l_pop * l_gen * N_BLOCKS_TO_SEARCH
        combinations.append({
            "n_samples": n_samples,
            "ref_pop": l_pop,
            "ref_gen": l_gen,
            "ref_mode": "layer_ea"
        })

print(f"Total random baseline tasks: {len(combinations)}")

for combo in combinations:
    task_queue.put(combo)

def worker(gpu_id):
    while True:
        try:
            if task_queue.empty():
                break
            
            task = task_queue.get_nowait()
            n_samples = task["n_samples"]
            ref_pop = task["ref_pop"]
            ref_gen = task["ref_gen"]
            ref_mode = task["ref_mode"]
            
        except Exception:
            break
            
        # Construct Output Path
        # Naming convention: random_baseline_nblk{}_samples{}_ref_{mode}_p{}_g{}.json
        output_filename = f"random_baseline_nblk{N_BLOCKS_TO_SEARCH}_samples{n_samples}_ref_{ref_mode}_p{ref_pop}_g{ref_gen}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        if os.path.exists(output_path):
             print(f"[GPU {gpu_id}] Skipping task (already exists): {output_filename}")
             task_queue.task_done()
             continue

        print(f"[GPU {gpu_id}] Starting Random Search: Samples={n_samples} (Ref: {ref_mode} P={ref_pop} G={ref_gen})")
        
        # Construct Command
        # We misuse --population_size and --n_generations to pass n_samples to search_nas.py
        # search_nas.py logic: n_samples = args.n_generations * args.population_size
        # So we set pop=1, gen=n_samples
        
        cmd = [
            "uv", "run", "python", "search_nas.py",
            "--search_mode", "random",
            "--population_size", "1",
            "--n_generations", str(n_samples),
            "--n_blocks_to_search", str(N_BLOCKS_TO_SEARCH),
            "--output_path", output_path,
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        log_filename = f"random_baseline_nblk{N_BLOCKS_TO_SEARCH}_samples{n_samples}_ref_{ref_mode}_p{ref_pop}_g{ref_gen}.log"
        log_path = os.path.join(LOG_DIR, log_filename)

        try:
            with open(log_path, "w") as log_file:
                subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            print(f"[GPU {gpu_id}] Finished: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Failed: {output_filename}. Check logs at {log_path}")
        finally:
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
