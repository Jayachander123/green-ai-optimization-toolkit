"""
12_pruning_and_quantization.py
Green AI Optimization Toolkit - Part 2

This script demonstrates algorithmic pruning (removing redundant weights) 
and dynamic quantization (compressing 32-bit floats to 8-bit integers).
Goal: Drastically reduce model size for cheaper, faster production inference 
without a significant drop in accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os

# Define a simple feed-forward network for demonstration
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def print_model_size(model, name="Model"):
    """Utility to print the physical file size of the model."""
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    print(f"{name} Size: {size_mb:.2f} MB")

def apply_pruning_and_quantization():
    print("Initializing standard 32-bit model...")
    model = SimpleModel()
    print_model_size(model, "Original 32-bit Float")

    # 1. Pruning
    # We remove 20% of the connections in fc1 that are closest to zero (L1 Unstructured)
    print("\nApplying L1 Unstructured Pruning (20%)...")
    prune.l1_unstructured(model.fc1, name="weight", amount=0.2)
    
    # Make the pruning permanent (removes the pruning mask and physically updates weights)
    prune.remove(model.fc1, 'weight')
    print("[FinOps Win] Redundant mathematical operations removed.")

    # 2. Dynamic Quantization
    # Compress the Linear layers from 32-bit floating point down to 8-bit integers
    print("\nApplying Dynamic Quantization (Float32 -> Int8)...")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, # Only quantize the linear layers
        dtype=torch.qint8
    )
    
    print_model_size(quantized_model, "Quantized 8-bit Integer")
    print("[FinOps Win] Model size drastically reduced for production deployment on cheaper edge hardware.")

if __name__ == "__main__":
    apply_pruning_and_quantization()
