#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
import argparse
import os
from nas_common import MobileNetSearchSpace

def main():
    parser = argparse.ArgumentParser(description="Generate random architecture based on a template")
    parser.add_argument("--base_arch", type=str, default="./block5_arch.json", help="Path to the base architecture JSON")
    parser.add_argument("--output_path", type=str, default="./random_arch.json", help="Path to save the generated JSON")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Load base architecture
    if not os.path.exists(args.base_arch):
        print(f"Error: Base architecture file {args.base_arch} not found.")
        return

    with open(args.base_arch, "r") as f:
        base_data = json.load(f)

    op_codes = base_data["op_codes"][:]
    width_codes = base_data["width_codes"][:]

    # Initialize Search Space to get valid ranges
    sp = MobileNetSearchSpace()
    num_ops = len(sp.op_list)
    num_widths = len(sp.width_choices)
    
    # Identify "zero" op index
    zero_op_idx = -1
    if "zero" in sp.op_list:
        zero_op_idx = sp.op_list.index("zero")
    
    # Create a list of valid op indices (excluding zero)
    valid_op_indices = [i for i in range(num_ops) if i != zero_op_idx]

    print("Original op_codes:", op_codes)
    print("Original width_codes:", width_codes)
    print(f"Zero op index: {zero_op_idx}")
    print(f"Valid op indices: {valid_op_indices}")

    # Randomize first 5 layers of op_codes (indices 0 to 4)
    # Note: Ensure we don't go out of bounds if op_codes is shorter than 5
    num_op_random = min(5, len(op_codes))
    for i in range(num_op_random):
        op_codes[i] = random.choice(valid_op_indices)
    
    # Randomize first 3 layers of width_codes (indices 0 to 2)
    # Note: Ensure we don't go out of bounds if width_codes is shorter than 3
    num_width_random = min(3, len(width_codes))
    for i in range(num_width_random):
        width_codes[i] = random.randrange(num_widths)

    print("-" * 20)
    print("New op_codes:", op_codes)
    print("New width_codes:", width_codes)

    # Create new JSON object
    new_arch = {
        "op_codes": op_codes,
        "width_codes": width_codes,
        "fitness": base_data.get("fitness", 0.0), # Inherit or reset fitness? keeping it might be misleading, but keeping for structure
        "params_mb": base_data.get("params_mb", 0.0), # Params will change, but we don't recalculate here without model
        "note": "Generated from base arch with randomized prefix"
    }

    # Save to file
    with open(args.output_path, "w") as f:
        json.dump(new_arch, f, indent=4)
    
    print(f"Randomized architecture saved to {args.output_path}")

if __name__ == "__main__":
    main()
