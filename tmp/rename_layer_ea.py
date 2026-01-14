import os
import re

GRID_SEARCH_DIR = "/Users/user/code/github/NAS_LEGO/gridsearch_output"

def rename_files():
    if not os.path.exists(GRID_SEARCH_DIR):
        print(f"Directory not found: {GRID_SEARCH_DIR}")
        return

    count = 0
    for filename in os.listdir(GRID_SEARCH_DIR):
        if not filename.endswith(".json"):
            continue
            
        # Skip if already marked as global_ea or layer_ea
        if "__global_ea" in filename or "__layer_ea" in filename:
            continue
            
        # Match pattern: search_nblk{}_lpop{}_lgen{}.json
        # Note: Previous ls output shows format like search_nblk10_lpop10_lgen12.json
        # We want to append __layer_ea before .json
        
        # Simple check: if it starts with search_ and ends with .json, and doesn't have double underscore suffix
        if filename.startswith("search_") and filename.endswith(".json"):
            new_name = filename.replace(".json", "__layer_ea.json")
            
            old_path = os.path.join(GRID_SEARCH_DIR, filename)
            new_path = os.path.join(GRID_SEARCH_DIR, new_name)
            
            print(f"Renaming: {filename} -> {new_name}")
            os.rename(old_path, new_path)
            count += 1
            
    print(f"Total files renamed: {count}")

if __name__ == "__main__":
    rename_files()
