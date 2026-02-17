import torch
import torch.nn as nn

def find_max_batch_size(model, input_shape, max_batch=1024, start_batch=16):
    """
    Tries to fit the largest batch size into memory by doubling until OOM.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Probing VRAM on {device}...")
    model.to(device)
    model.train()
    
    current_batch = start_batch
    last_successful = start_batch
    
    try:
        while current_batch <= max_batch:
            # Create dummy data
            dummy_input = torch.randn(current_batch, *input_shape).to(device)
            
            # Try forward/backward
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            
            print(f"   ✅ Batch Size {current_batch} fits.")
            last_successful = current_batch
            current_batch *= 2
            
            # Clean up memory
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            del dummy_input, output, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"   ❌ Batch Size {current_batch} OOM! Backing off.")
        else:
            print(f"   ❌ Error: {e}")
            
    print(f"🎯 Optimal Batch Size Found: {last_successful}")
    return last_successful

# RUN DEMO
if __name__ == "__main__":
    # A heavy model (simulated)
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 10)
    )
    # Input shape excluding batch dim
    input_shape = (1024,)
    
    find_max_batch_size(model, input_shape)
