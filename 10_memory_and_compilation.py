"""
10_memory_and_compilation.py
Green AI Optimization Toolkit - Part 2

This script demonstrates how to trade a small amount of compute time for 
massive VRAM savings (Checkpointing) and how to fuse kernels for free speed (torch.compile).
Goal: Fit larger batch sizes on cheaper GPUs and maximize memory bandwidth.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

def optimize_memory_and_speed(model_name="bert-base-uncased"):
    print(f"Loading {model_name} for Memory Optimization...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # 1. Gradient Checkpointing (Memory Optimization)
    # Instead of storing all activations for backprop, we recompute them.
    # This allows us to increase batch size by up to 4x on the same GPU.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[FinOps Win] Gradient Checkpointing Enabled: VRAM usage drastically reduced.")
    else:
        print("Model does not support native gradient checkpointing.")

    # 2. Compiler Fusion (Speed Optimization - PyTorch 2.0+)
    # Fuses multiple operations (like an activation + dropout) into a single GPU kernel.
    # This reduces memory read/writes, which is the main bottleneck in modern AI.
    try:
        print("Compiling model via torch.compile()...")
        # 'reduce-overhead' is great for smaller batch sizes to minimize Python overhead
        optimized_model = torch.compile(model, mode="reduce-overhead")
        print("[FinOps Win] Kernel Fusion Enabled: Execution speed increased with zero code changes.")
    except Exception as e:
        print(f"Compilation failed (ensure you are on PyTorch 2.0+): {e}")
        optimized_model = model
        
    return optimized_model

if __name__ == "__main__":
    optimized_model = optimize_memory_and_speed()
