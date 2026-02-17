import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

def run_profiler_demo():
    print("--- 📊 PyTorch Profiler Demo ---")
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    # Use CPU only for demo compatibility
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print("\n✅ Profiling complete. Use this to find bottlenecks.")

if __name__ == "__main__":
    run_profiler_demo()
