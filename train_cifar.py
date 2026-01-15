#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import datasets, transforms
import sys
import ast

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
    p.add_argument("--arch_path", type=str, help="Path to the architecture JSON file")
    p.add_argument("--run_id", type=str, help="Run ID from CSV to load architecture")
    p.add_argument("--csv_path", default="./gridsearch_output/all_results.csv", type=str, help="Path to the results CSV")
    
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
    
    # Distributed Training
    p.add_argument("--local_rank", default=-1, type=int, help="Local rank for distributed training")
    p.add_argument("--distributed", action="store_true", help="Enable distributed training")
    
    return p.parse_args()

def main():
    args = parse_args()
    
    # Distributed Setup
    if args.distributed:
        # For torchrun, local_rank is set via env variable
        if "LOCAL_RANK" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])
        
        if args.local_rank == -1:
            logging.error("Distributed training enabled but local_rank is -1. Use torchrun or set LOCAL_RANK.")
            sys.exit(1)
            
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.device = f"cuda:{args.local_rank}"
        
        # Adjust batch size for per-GPU
        # args.train_batch is total batch size? Or per-GPU?
        # Usually users specify total batch size in args, but here let's assume args.train_batch is per-GPU 
        # or we divide it. Let's assume args.train_batch is per-GPU as standard in this codebase context?
        # Standard practice: args.batch_size is per-GPU.
        pass
    else:
        args.local_rank = 0 # Default for single card

    setup_logger(args.log_path, args.log_name)
    if args.local_rank == 0:
        logging.info("Args:\n" + json.dumps(vars(args), indent=4))
    
    set_seed(args.seed + args.local_rank) # Different seed for different rank? 
    # Usually for DDP, we want same init weights (seed same), but different data shuffle (handled by sampler).
    # set_seed(args.seed) is fine if it sets torch.manual_seed. 
    # DDP broadcasts model weights from rank 0 anyway.
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.local_rank == 0:
        logging.info(f"Using device: {device}")

    # Load Architecture
    op_codes = None
    width_codes = None
    
    if args.run_id:
        if not os.path.exists(args.csv_path):
            logging.error(f"CSV file not found: {args.csv_path}")
            sys.exit(1)
            
        try:
            df = pd.read_csv(args.csv_path)
            row = df[df['id'].astype(str) == str(args.run_id)]
            
            if row.empty:
                logging.error(f"Run ID {args.run_id} not found in {args.csv_path}")
                sys.exit(1)
            
            # Extract codes
            # They are stored as strings in CSV, so we need to parse them
            op_codes_str = row.iloc[0]['op_codes']
            width_codes_str = row.iloc[0]['width_codes']
            
            try:
                op_codes = ast.literal_eval(op_codes_str)
                width_codes = ast.literal_eval(width_codes_str)
            except (ValueError, SyntaxError) as e:
                # Fallback if simple eval fails (e.g. if json formatted)
                try:
                    op_codes = json.loads(op_codes_str)
                    width_codes = json.loads(width_codes_str)
                except Exception:
                    logging.error(f"Failed to parse codes from CSV: op={op_codes_str}, width={width_codes_str}")
                    sys.exit(1)
                    
            if args.local_rank == 0:
                logging.info(f"Loaded architecture from CSV (ID={args.run_id})")
                
        except Exception as e:
            logging.error(f"Error reading CSV or parsing ID: {e}")
            sys.exit(1)
            
    elif args.arch_path:
        if not os.path.exists(args.arch_path):
            logging.error(f"Architecture file not found: {args.arch_path}")
            sys.exit(1)
            
        with open(args.arch_path, "r") as f:
            arch_data = json.load(f)
        
        if args.local_rank == 0:
            logging.info(f"Loaded architecture from {args.arch_path}")
        
        op_codes = arch_data.get('op_codes')
        if op_codes is None:
            op_codes = arch_data.get('op_codes_prefix')
            if args.local_rank == 0:
                logging.info("Using 'op_codes_prefix' from JSON.")
                
        width_codes = arch_data['width_codes']
    else:
        logging.error("Either --arch_path or --run_id must be provided.")
        sys.exit(1)

    if op_codes is None:
        logging.error("No 'op_codes' or 'op_codes_prefix' found.")
        sys.exit(1)
        
    if args.local_rank == 0:
        logging.info(f"  op_codes: {op_codes}")
        logging.info(f"  width_codes: {width_codes}")
    
    # Reconstruct Model
    sp = MobileNetSearchSpace(num_classes=args.num_classes, small_input=args.small_input)
    
    # Determine if we should build a prefix model or full model
    if len(op_codes) < sp.total_blocks:
        if args.local_rank == 0:
            logging.info(f"Building PREFIX model with {len(op_codes)} blocks (total {sp.total_blocks})")
        model = sp.get_prefix_model(op_codes, width_codes)
    else:
        if args.local_rank == 0:
            logging.info(f"Building FULL model with {len(op_codes)} blocks")
        model = sp.get_model(op_codes, width_codes)
    
    model = model.to(device)
    
    # Wrap DDP
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Training
    if args.local_rank == 0:
        logging.info(f"Parameters: lr={args.lr}, train_batch={args.train_batch}, "
                     f"train_epochs={args.train_epochs}, mixup_alpha={args.mixup_alpha}, "
                     f"label_smoothing={args.label_smoothing}")

    num_workers = 4 if args.distributed else 2
    
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        root=args.data_path, batch_size=args.train_batch, num_workers=num_workers,
        use_cutout=args.use_cutout, cutout_length=args.cutout_length,
        distributed=args.distributed
    )

    final_top1 = train_and_eval(model, train_loader, val_loader, test_loader, device=device, args=args, rank=args.local_rank)
    
    if args.local_rank == 0:
        logging.info(f"Final Accuracy of Best Model (Top-1): {final_top1*100:.2f}%")

        # Optional: Save model weights
        # Use log_name to determine model filename (replace .log with .pth)
        if args.log_name.endswith(".log"):
            model_filename = args.log_name.replace(".log", ".pth")
        else:
            model_filename = args.log_name + ".pth"
            
        save_path = os.path.join(args.log_path, model_filename)
        # Save underlying model if wrapped
        model_to_save = model.module if args.distributed else model
        torch.save(model_to_save, save_path)
        logging.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
