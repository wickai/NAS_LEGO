import os
import subprocess
import itertools
import time
import threading
from queue import Queue

# Configuration
N_GENERATIONS_LIST = [50, 100, 200]
N_BLOCKS_TO_SEARCH_LIST = [20]
LAYER_POPULATION_LIST = [8, 16, 20]
LAYER_GENERATIONS_LIST = [4, 8, 12]

# GPUs available
GPUS = [0, 1, 2, 3]

# Task Queue
task_queue = Queue()

# Generate all combinations
combinations = list(itertools.product(
    N_GENERATIONS_LIST,
    N_BLOCKS_TO_SEARCH_LIST,
    LAYER_POPULATION_LIST,
    LAYER_GENERATIONS_LIST
))

print(f"Total tasks: {len(combinations)}")

for combo in combinations:
    task_queue.put(combo)

def worker(gpu_id):
    while not task_queue.empty():
        try:
            # Get a task with non-blocking call to avoid hanging if empty
            n_gen, n_blocks, l_pop, l_gen = task_queue.get_nowait()
        except Exception:
            break
            
        # Construct Output Path
        # Format: imagenet_ngen{}_blk{}_lpop{}_lgen{}.json
        output_filename = f"imagenet_ngen{n_gen}_blk{n_blocks}_lpop{l_pop}_lgen{l_gen}.json"
        output_path = os.path.join(".", "grid_results", output_filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"[GPU {gpu_id}] Starting task: N_GEN={n_gen}, BLOCKS={n_blocks}, L_POP={l_pop}, L_GEN={l_gen}")
        
        # Construct Command
        # Note: We assume search_nas.py is in the current directory
        # We use 'uv run python' as requested in previous steps
        cmd = [
            "uv", "run", "python", "search_nas.py",
            "--dataset", "imagenet",
            "--data_root", "/mnt/sda/weizixiang/wk/data/imagenet_v2/datasets--imagenet-1k/data/",
            "--data_format", "parquet",
            "--population_size", "20", # Keeping default or from script? Script used POPULATION_SIZE variable.
                                       # But user didn't ask to vary POPULATION_SIZE, only N_GENERATIONS.
                                       # However, run_search_imagenet.sh used POPULATION_SIZE=20. 
                                       # But N_GENERATIONS was 50.
                                       # I will set population_size to 20 (fixed) and n_generations to the varied value.
            "--n_generations", str(n_gen),
            "--n_blocks_to_search", str(n_blocks),
            "--layer_population", str(l_pop),
            "--layer_generations", str(l_gen),
            "--output_path", output_path,
            "--use_pareto",
            "--device", "cuda" # Ensure it tries to use CUDA
        ]
        
        # Environment with specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        try:
            # Run command
            subprocess.run(cmd, env=env, check=True)
            print(f"[GPU {gpu_id}] Finished task: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Task failed: {output_filename}. Error: {e}")
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
