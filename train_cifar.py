#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys

# Import from common
from nas_common import (
    set_seed, setup_logger, MobileNetSearchSpace,
    train_and_eval, get_cifar10_dataloaders
)

def parse_args():
    p = argparse.ArgumentParser("MobileNetV2 Training (Load from JSON)")
    p.add_argument("--log_path", default="./logs", type=str)
    p.add_argument("--log_name", default="train_cifar.log", type=str)
    p.add_argument("--data_path", default="./data", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=42, type=int)
    
    # Architecture source
    p.add_argument("--arch_path", required=True, type=str, help="Path to the architecture JSON file")

    # Training parameters
    p.add_argument("--train_batch", default=128, type=int)
    p.add_argument("--train_epochs", default=200, type=int)
    p.add_argument("--lr", default=0.05, type=float)
    p.add_argument("--small_input", action="store_true", default=True)
    p.add_argument("--num_classes", default=10, type=int)
    p.add_argument("--use_cutout", action="store_true", default=True)
    p.add_argument("--cutout_length", default=16, type=int)
    p.add_argument("--mixup_alpha", default=0.2, type=float)
    p.add_argument("--label_smoothing", default=0.1, type=float)
    p.add_argument("--weight_decay", default=5e-4, type=float)
    
    return p.parse_args()

def main():
    args = parse_args()
    setup_logger(args.log_path, args.log_name)
    logging.info("Args:\n" + json.dumps(vars(args), indent=4))
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load Architecture
    if not os.path.exists(args.arch_path):
        logging.error(f"Architecture file not found: {args.arch_path}")
        sys.exit(1)
        
    with open(args.arch_path, "r") as f:
        arch_data = json.load(f)
    
    logging.info(f"Loaded architecture from {args.arch_path}")
    
    op_codes = arch_data.get('op_codes')
    if op_codes is None:
        op_codes = arch_data.get('op_codes_prefix')
        logging.info("Using 'op_codes_prefix' from JSON.")
    
    if op_codes is None:
        logging.error("No 'op_codes' or 'op_codes_prefix' found in JSON.")
        sys.exit(1)
        
    width_codes = arch_data['width_codes']

    logging.info(f"  op_codes: {op_codes}")
    logging.info(f"  width_codes: {width_codes}")
    
    # Reconstruct Model
    sp = MobileNetSearchSpace(num_classes=args.num_classes, small_input=args.small_input)
    
    # Determine if we should build a prefix model or full model
    # If op_codes length matches total blocks, build full model.
    # Otherwise, assume it's a prefix model.
    if len(op_codes) < sp.total_blocks:
        logging.info(f"Building PREFIX model with {len(op_codes)} blocks (total {sp.total_blocks})")
        model = sp.get_prefix_model(op_codes, width_codes)
    else:
        logging.info(f"Building FULL model with {len(op_codes)} blocks")
        model = sp.get_model(op_codes, width_codes)
    
    # Training
    logging.info(f"Parameters: lr={args.lr}, train_batch={args.train_batch}, "
                 f"train_epochs={args.train_epochs}, mixup_alpha={args.mixup_alpha}, "
                 f"label_smoothing={args.label_smoothing}")

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        root=args.data_path, batch_size=args.train_batch, num_workers=2,
        use_cutout=args.use_cutout, cutout_length=args.cutout_length
    )

    final_top1 = train_and_eval(model, train_loader, val_loader, test_loader, device=device, args=args)
    logging.info(f"Final Accuracy of Best Model (Top-1): {final_top1*100:.2f}%")

    # Optional: Save model weights
    save_path = os.path.join(args.log_path, "best_model.pth")
    torch.save(model, save_path)
    logging.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
