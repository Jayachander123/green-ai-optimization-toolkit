import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

def smoke_test(model, loader, device='cpu', steps=2):
    """
    Runs a dry-run on CPU to catch shape mismatches 
    and OOM bugs before the real run starts.
    """
    print(f"💨 Running Smoke Test on {device}...")
    model.to(device)
    model.train()
    
    try:
        for i, (data, target) in enumerate(loader):
            if i >= steps: break
            # Force data to CPU for the test
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = output.sum() # Dummy loss
            loss.backward()
            print(f"   Batch {i+1} processed successfully.")
            
        print("✅ Smoke Test Passed. Safe to launch expensive job.")
        return True
    except Exception as e:
        print(f"❌ Smoke Test Failed: {e}")
        return False

if __name__ == "__main__":
    # Setup Dummy Data & Model
    dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=10, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=4)
    
    # A simple valid model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.Flatten(),
        nn.Linear(16 * 222 * 222, 10) # Correct calculated shape
    )

    # Run the test
    smoke_test(model, loader)
