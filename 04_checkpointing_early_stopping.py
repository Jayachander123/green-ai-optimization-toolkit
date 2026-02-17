import torch
import torch.nn as nn
import os

# CONFIG
CHECKPOINT_DIR = "./checkpoints"
PATIENCE = 3 # Stop if no improvement for 3 epochs

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def save_checkpoint(model, epoch, loss):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, path)
    print(f"💾 Checkpoint saved: {path}")

# SIMULATION
print("--- 🛑 Early Stopping & Checkpointing Demo ---")
model = nn.Linear(10, 1)
stopper = EarlyStopper(patience=PATIENCE)

# Simulated Loss Values (Notice it goes down, then goes up/plateaus)
losses = [0.9, 0.8, 0.7, 0.65, 0.64, 0.64, 0.65, 0.66]

for epoch, val_loss in enumerate(losses):
    print(f"Epoch {epoch+1}: Validation Loss = {val_loss}")
    save_checkpoint(model, epoch, val_loss)
    
    if stopper.early_stop(val_loss):
        print(f"❌ Early Stopping Triggered at Epoch {epoch+1}! Saving compute.")
        break
