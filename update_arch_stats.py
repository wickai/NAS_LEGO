#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import torch
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

# Import from common
from nas_common import MobileNetSearchSpace

def format_number(num):
    """
    Format a number with K/M/G suffixes for readability.
    """
    if num > 1e9:
        return f"{num/1e9:.2f}G"
    elif num > 1e6:
        return f"{num/1e6:.2f}M"
    else:
        return f"{num/1e3:.2f}K"

def get_model_complexity_info(model, inputs):
    """
    Return model params, activations, and FLOPs.
    """
    # Calculate FLOPs
    flops = FlopCountAnalysis(model, inputs)
    flops.unsupported_ops_warnings(False)
    
    op_flops = flops.by_operator()
    
    conv_flops = sum(v for k, v in op_flops.items() if 'conv' in k.lower())
    linear_flops = sum(v for k, v in op_flops.items() if 'linear' in k.lower() or 'matmul' in k.lower())
    
    total_conv_fc_flops = conv_flops + linear_flops

    acts = ActivationCountAnalysis(model, inputs)
    
    # Calculate trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "params": num_params,
        "acts": acts.total(),
        "flops": flops.total(),
        "total_conv_fc_flops": total_conv_fc_flops
    }

def update_json_with_stats(json_path, num_classes=10, small_input=True, input_size=32):
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return False

    try:
        with open(json_path, "r") as f:
            arch_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON {json_path}: {e}")
        return False
    
    op_codes = arch_data.get('op_codes')
    if op_codes is None:
        op_codes = arch_data.get('op_codes_prefix')
    
    if op_codes is None:
        print(f"Skipping {json_path}: No 'op_codes' or 'op_codes_prefix' found.")
        return False
        
    width_codes = arch_data['width_codes']
    
    # Reconstruct Model
    sp = MobileNetSearchSpace(num_classes=num_classes, small_input=small_input)
    
    if len(op_codes) < sp.total_blocks:
        model = sp.get_prefix_model(op_codes, width_codes)
    else:
        model = sp.get_model(op_codes, width_codes)

    # Prepare input
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # Calculate Complexity
    info = get_model_complexity_info(model, dummy_input)
    
    # Update JSON data
    stats = {
        "params": info['params'],
        "params_str": format_number(info['params']),
        "flops": info['flops'],
        "flops_str": format_number(info['flops']),
        "total_conv_fc_flops": info['total_conv_fc_flops'],
        "total_conv_fc_flops_str": format_number(info['total_conv_fc_flops']),
        "acts": info['acts'],
        "acts_str": format_number(info['acts'])
    }
    
    # Merge stats into arch_data
    arch_data.update(stats)
    
    # Write back to JSON
    with open(json_path, "w") as f:
        json.dump(arch_data, f, indent=4)
        
    print(f"Updated {json_path}: Params={stats['params_str']}, FLOPs={stats['flops_str']}")
    return True

def main():
    parser = argparse.ArgumentParser("Update Architecture JSON with Stats")
    parser.add_argument("--arch_path", required=True, type=str, help="Path to the architecture JSON file or directory")
    parser.add_argument("--recursive", action="store_true", help="If arch_path is directory, recursively update all .json files")
    parser.add_argument("--file_prefix", type=str, default=None, help="Only process files starting with this prefix (when recursive)")
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--small_input", action="store_true", default=True, help="Use 32x32 input (CIFAR)")
    parser.add_argument("--input_size", default=32, type=int, help="Input image size")
    
    args = parser.parse_args()

    if os.path.isdir(args.arch_path):
        if not args.recursive:
            print(f"Error: {args.arch_path} is a directory. Use --recursive to process all JSON files.")
            sys.exit(1)
            
        print(f"Scanning directory: {args.arch_path}")
        if args.file_prefix:
            print(f"Filtering files with prefix: '{args.file_prefix}'")
            
        count = 0
        for root, dirs, files in os.walk(args.arch_path):
            for file in files:
                if not file.endswith(".json"):
                    continue
                    
                if args.file_prefix and not file.startswith(args.file_prefix):
                    continue
                    
                full_path = os.path.join(root, file)
                if update_json_with_stats(full_path, args.num_classes, args.small_input, args.input_size):
                    count += 1
        print(f"Processed {count} files.")
        
    else:
        update_json_with_stats(args.arch_path, args.num_classes, args.small_input, args.input_size)

if __name__ == "__main__":
    main()
