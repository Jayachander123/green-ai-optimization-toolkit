import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import time

# --- CONFIGURATION ---
BATCH_SIZE = 64        # Target batch size
MICRO_BATCH = 8        # What actually fits in GPU memory
ACCUM_STEPS = BATCH_SIZE // MICRO_BATCH
EPOCHS = 2

def run_training():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running on: {device}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: You are running on CPU. Mixed Precision (AMP) requires an NVIDIA GPU to work.")
        print("    The code will run, but 'autocast' will be skipped/ineffective.\n")

    # 1. Setup Dummy Data (No download needed)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FakeData(size=1000, image_size=(3, 224, 224), num_classes=10, transform=transform)
    loader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=True)

    # 2. Simple Model (ResNet-style)
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() # The magic component for FP16

    print(f"📉 Starting Training (Target Batch: {BATCH_SIZE} | Micro Batch: {MICRO_BATCH})")
    start_time = time.time()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            # --- THE ARTICLE LOGIC STARTS HERE ---
            
            # 1. The Toggle: Run forward pass in FP16 context
            # (If on CPU, autocast is a no-op, so it won't crash)
            with autocast(enabled=(device.type == 'cuda')):
                output = model(data)
                loss = criterion(output, target)
                loss = loss / ACCUM_STEPS # Normalize loss for accumulation

            # 2. Scale gradients and accumulate
            # (If on CPU, scaler just does normal backward)
            scaler.scale(loss).backward()

            # 3. Step only after N micro-batches
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # --- END ARTICLE LOGIC ---

            if i % 20 == 0:
                print(f"   Epoch {epoch+1} | Step {i} | Loss: {loss.item()*ACCUM_STEPS:.4f}")

    print(f"✅ Training Complete. Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    run_training()
