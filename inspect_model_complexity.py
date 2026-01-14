#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import torch
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

# Import from common
from nas_common import MobileNetSearchSpace, count_parameters_in_MB

def format_number(num):
    """
    Format a number with K/M/G suffixes for readability.
    Reference: /Users/user/code/github/NAS_LEGO/validation.py#L147-158
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
    # Don't print warnings
    flops.unsupported_ops_warnings(False)
    
    op_flops = flops.by_operator()
    
    # Try to sum conv and linear flops. 
    # fvcore keys might vary (e.g. 'conv' vs 'conv2d'), so let's be robust or debug.
    # Common keys: 'conv', 'linear', 'matmul', etc.
    # We will try to match 'conv' substring or specific known keys.
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
        "total_conv_fc_flops": total_conv_fc_flops,
        "op_flops": op_flops
    }

def main():
    parser = argparse.ArgumentParser("Inspect Model Complexity (FLOPs, Params)")
    parser.add_argument("--arch_path", required=True, type=str, help="Path to the architecture JSON file")
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--small_input", action="store_true", default=True, help="Use 32x32 input (CIFAR)")
    parser.add_argument("--input_size", default=32, type=int, help="Input image size (e.g. 32 for CIFAR)")
    
    args = parser.parse_args()

    # Check file existence
    if not os.path.exists(args.arch_path):
        print(f"Error: File not found: {args.arch_path}")
        sys.exit(1)

    # Load Architecture
    with open(args.arch_path, "r") as f:
        arch_data = json.load(f)
    
    op_codes = arch_data.get('op_codes')
    if op_codes is None:
        op_codes = arch_data.get('op_codes_prefix')
    
    if op_codes is None:
        print("Error: No 'op_codes' or 'op_codes_prefix' found in JSON.")
        sys.exit(1)
        
    width_codes = arch_data['width_codes']
    
    print(f"Loaded architecture from {args.arch_path}")
    print(f"  op_codes ({len(op_codes)}): {op_codes}")
    print(f"  width_codes: {width_codes}")

    # Reconstruct Model
    sp = MobileNetSearchSpace(num_classes=args.num_classes, small_input=args.small_input)
    
    if len(op_codes) < sp.total_blocks:
        print(f"Building PREFIX model with {len(op_codes)} blocks (total {sp.total_blocks})")
        model = sp.get_prefix_model(op_codes, width_codes)
    else:
        print(f"Building FULL model with {len(op_codes)} blocks")
        model = sp.get_model(op_codes, width_codes)

    # Prepare input
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    input_res = args.input_size
    dummy_input = torch.randn(1, 3, input_res, input_res).to(device)

    # Calculate Complexity
    info = get_model_complexity_info(model, dummy_input)
    
    # Print Results
    print("\n" + "="*40)
    print(f"Model Complexity Analysis")
    print("="*40)
    
    print(f"{'Metric':<20} | {'Raw Value':<15} | {'Formatted':<10}")
    print("-" * 50)
    
    metrics = [
        ("Params", info['params']),
        ("FLOPs (Total)", info['flops']),
        ("FLOPs (Conv+FC)", info['total_conv_fc_flops']),
        ("Activations", info['acts'])
    ]
    
    for name, value in metrics:
        print(f"{name:<20} | {value:<15.0f} | {format_number(value)}")
        
    print("-" * 50)
    print("Detailed OP FLOPs:")
    for k, v in info['op_flops'].items():
        if v > 0:
            print(f"  {k:<20}: {format_number(v)}")
    print("="*40)

if __name__ == "__main__":
    main()
