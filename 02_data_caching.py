import torch
from torch.utils.data import Dataset, DataLoader
import time
import os
import shutil

# --- CONFIGURATION ---
CACHE_DIR = "./_data_cache"
DATASET_SIZE = 50
SIMULATED_DELAY = 0.05 # Simulates expensive processing (e.g., resizing high-res images)

class ExpensiveDataset(Dataset):
    def __init__(self, size, use_cache=False):
        self.size = size
        self.use_cache = use_cache
        
        if use_cache and not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            print(f"📁 Created cache directory: {CACHE_DIR}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        cache_path = os.path.join(CACHE_DIR, f"sample_{idx}.pt")
        
        # STRATEGY: Check Cache First
        if self.use_cache and os.path.exists(cache_path):
            return torch.load(cache_path)
        
        # SIMULATE EXPENSIVE WORK (The Bottleneck)
        time.sleep(SIMULATED_DELAY) 
        data = torch.randn(3, 224, 224) # Fake Image
        
        # STRATEGY: Save to Cache
        if self.use_cache:
            torch.save(data, cache_path)
            
        return data

def run_benchmark():
    print("--- 🐢 Run 1: Uncached (Simulating Slow I/O) ---")
    dataset_slow = ExpensiveDataset(DATASET_SIZE, use_cache=False)
    loader_slow = DataLoader(dataset_slow, batch_size=10)
    
    start = time.time()
    for batch in loader_slow:
        pass
    print(f"❌ Time taken: {time.time() - start:.2f}s (Bottlenecked)\n")

    print("--- 💾 Run 2: Caching (First Pass - Generating Cache) ---")
    dataset_caching = ExpensiveDataset(DATASET_SIZE, use_cache=True)
    loader_caching = DataLoader(dataset_caching, batch_size=10)
    
    start = time.time()
    for batch in loader_caching:
        pass
    print(f"⚠️  Time taken: {time.time() - start:.2f}s (Building Cache)\n")

    print("--- 🚀 Run 3: Cached (Second Pass - The Speedup) ---")
    # Re-initialize to prove we are reading from disk
    dataset_cached = ExpensiveDataset(DATASET_SIZE, use_cache=True) 
    loader_cached = DataLoader(dataset_cached, batch_size=10)
    
    start = time.time()
    for batch in loader_cached:
        pass
    print(f"✅ Time taken: {time.time() - start:.2f}s (Optimized!)")
    
    # Cleanup
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print("\n🧹 Cleanup: Cache directory removed.")

if __name__ == "__main__":
    run_benchmark()
