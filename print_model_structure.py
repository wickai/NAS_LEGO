#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os
import torch
from nas_common import MobileNetSearchSpace

def main():
    parser = argparse.ArgumentParser(description="Print PyTorch model structure from architecture JSON")
    parser.add_argument("--arch_path", type=str, default="./block5_arch.json", help="Path to the architecture JSON")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for the model")
    parser.add_argument("--small_input", action="store_true", default=True, help="Adapt for CIFAR-10 small input")
    args = parser.parse_args()

    # Load architecture
    if not os.path.exists(args.arch_path):
        print(f"Error: Architecture file {args.arch_path} not found.")
        return

    with open(args.arch_path, "r") as f:
        arch_data = json.load(f)

    op_codes = arch_data["op_codes"]
    width_codes = arch_data["width_codes"]

    print(f"Loading architecture from: {args.arch_path}")
    print(f"Op codes: {op_codes}")
    print(f"Width codes: {width_codes}")
    print("-" * 40)

    # Initialize Search Space and Model
    sp = MobileNetSearchSpace(num_classes=args.num_classes, small_input=args.small_input)
    model = sp.get_model(op_codes, width_codes)

    # Print model structure
    print(model)
    
    # Optional: Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print("-" * 40)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

if __name__ == "__main__":
    main()
