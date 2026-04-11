"""
09_peft_lora_warm_start.py
Green AI Optimization Toolkit - Part 2

This script demonstrates Parameter-Efficient Fine-Tuning (PEFT) using LoRA, 
and how to "warm-start" a model by injecting pre-trained embeddings.
Goal: Train massive models on single GPUs by freezing 99% of the base weights.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def apply_lora_to_model(model_name="gpt2"):
    print(f"Loading base model: {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 1. Define the LoRA Configuration
    # We target specific projection layers to minimize trainable parameters
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["c_attn"], # HuggingFace GPT-2 attention block
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 2. Wrap the model
    efficient_model = get_peft_model(base_model, config)
    
    # Print the FinOps savings
    trainable_params, all_param = efficient_model.get_nb_trainable_parameters()
    print(f"Total Parameters: {all_param:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Compute Savings: We are only training {(trainable_params / all_param) * 100:.2f}% of the model!")
    
    return efficient_model

def warm_start_embeddings(model, new_vocab_size, pretrained_tensor):
    """
    Injects pre-trained domain-specific embeddings (e.g., medical vocab) 
    to skip expensive early-epoch representation learning.
    """
    print("\nApplying Warm-Start Embeddings...")
    old_embeddings = model.get_input_embeddings()
    embedding_dim = old_embeddings.embedding_dim
    
    # Create new embedding layer for the expanded vocabulary
    new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
    
    # Copy over the pre-trained weights to skip learning from scratch
    # Assuming pretrained_tensor shape matches (new_vocab_size, embedding_dim)
    with torch.no_grad():
        new_embeddings.weight.data.copy_(pretrained_tensor)
        
    # Freeze the embeddings to save compute during backprop
    new_embeddings.requires_grad_(False)
    model.set_input_embeddings(new_embeddings)
    print("Embeddings injected and frozen. Early-epoch compute slashed.")
    
    return model

if __name__ == "__main__":
    # Example Execution
    model = apply_lora_to_model()
    
    # Dummy tensor representing pre-trained domain embeddings
    dummy_medical_embeddings = torch.randn(50257, 768) 
    model = warm_start_embeddings(model, 50257, dummy_medical_embeddings)
