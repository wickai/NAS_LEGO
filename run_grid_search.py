import os
import subprocess
import itertools
import threading
from queue import Queue
import time
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Run Grid Search for NAS")
parser.add_argument("--n_blocks", type=int, default=5, help="Number of blocks to search")
args = parser.parse_args()

# Configuration
LAYER_POPULATION_LIST = [5, 10, 16, 20, 40]
LAYER_GENERATIONS_LIST = [2, 4, 8, 12]
N_BLOCKS_TO_SEARCH = args.n_blocks

# Fixed parameters from run_search.sh
POPULATION_SIZE = 2
N_GENERATIONS = 50

# Output directory
OUTPUT_DIR = "gridsearch_output"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# GPUs available (User mentioned 7 cards)
# Assuming IDs 0 to 6. If IDs are different (e.g. 1-7), this might need adjustment.
GPUS = [0, 1, 2, 3, 4, 5, 6]

# Task Queue
task_queue = Queue()

# Generate all combinations
combinations = list(itertools.product(
    LAYER_POPULATION_LIST,
    LAYER_GENERATIONS_LIST
))

print(f"Total tasks: {len(combinations)}")

for combo in combinations:
    task_queue.put(combo)

def worker(gpu_id):
    while True:
        try:
            # Get a task with non-blocking call
            if task_queue.empty():
                break
            l_pop, l_gen = task_queue.get_nowait()
        except Exception:
            break
            
        # Construct Output Path
        # Format: search_nblk{}_lpop{}_lgen{}.json
        output_filename = f"search_nblk{N_BLOCKS_TO_SEARCH}_lpop{l_pop}_lgen{l_gen}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Check if already done (Resume capability)
        if os.path.exists(output_path):
             print(f"[GPU {gpu_id}] Skipping task (already exists): {output_filename}")
             task_queue.task_done()
             continue

        print(f"[GPU {gpu_id}] Starting task: N_BLK={N_BLOCKS_TO_SEARCH}, L_POP={l_pop}, L_GEN={l_gen}")
        
        # Construct Command
        cmd = [
            "uv", "run", "python", "search_nas.py",
            "--population_size", str(POPULATION_SIZE),
            "--n_generations", str(N_GENERATIONS),
            "--layer_population", str(l_pop),
            "--layer_generations", str(l_gen),
            "--n_blocks_to_search", str(N_BLOCKS_TO_SEARCH),
            "--output_path", output_path,
        ]
        
        # Environment with specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Log file path
        log_filename = f"search_nblk{N_BLOCKS_TO_SEARCH}_lpop{l_pop}_lgen{l_gen}.log"
        log_path = os.path.join(LOG_DIR, log_filename)

        try:
            # Run command with log redirection
            with open(log_path, "w") as log_file:
                subprocess.run(cmd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            print(f"[GPU {gpu_id}] Finished task: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Task failed: {output_filename}. Check logs at {log_path}. Error: {e}")
        finally:
            task_queue.task_done()

# Start Worker Threads
threads = []
for gpu in GPUS:
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)

# Wait for all threads to finish
for t in threads:
    t.join()

print("All grid search tasks completed.")
