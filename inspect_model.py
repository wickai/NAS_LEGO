import torch
import json
import os
import sys
from nas_common import MobileNetSearchSpace

def main():
    model_path = "train_cifar_4card.pth"
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    if isinstance(checkpoint, torch.nn.Module):
        print("Checkpoint is a model object.")
        if hasattr(checkpoint, 'op_codes'):
            print(f"Model op_codes: {checkpoint.op_codes}")
        else:
            print("Model has no op_codes attribute.")
            
        if hasattr(checkpoint, 'width_codes'):
            print(f"Model width_codes: {checkpoint.width_codes}")
        else:
            print("Model has no width_codes attribute.")
            
        # Check first layer weight shape
        # MobileNetV2 usually has stem
        if hasattr(checkpoint, 'stem'):
            # stem usually is a sequential or conv
            # Let's see nas_common.py
            print("Checking stem...")
            print(checkpoint.stem)
            
    else:
        print("Checkpoint is NOT a model object.")

    # Load test_arch.json for comparison
    with open("test_arch.json", "r") as f:
        arch_data = json.load(f)
    print(f"JSON op_codes: {arch_data.get('op_codes')}")
    print(f"JSON width_codes: {arch_data.get('width_codes')}")

if __name__ == "__main__":
    main()
