# Green AI Optimization Toolkit 🌿

**Practical Implementations of AI FinOps & Sustainable Training**

> *"A single training run can emit as much CO₂ as five cars do in a year."* — UMass Amherst

This repository contains the reference implementations and code patterns discussed in the InfoWorld article:  
**"The Toggle-Away Efficiencies: Cutting AI Costs Inside the Training Loop"**

## 📖 Overview
Training large models is expensive. The industry often focuses on hardware solutions (buying more H100s), but significant efficiency gains are available through software optimizations.

This toolkit provides **drop-in PyTorch patterns** to reduce compute costs, lower carbon footprints, and increase training throughput without altering model architecture.

## 📂 Repository Contents

This toolkit is organized to match the sections of the article.

### 1. The Compute Levers
*Optimizing math and throughput.*
- **`01_mixed_precision.py`**  
  Implements **Automatic Mixed Precision (AMP)** and **Gradient Accumulation**.  
  *Goal:* Run larger effective batch sizes on smaller GPUs and reduce memory footprint by ~40%.

### 2. The Data Levers
*Solving the I/O bottleneck.*
- **`02_data_caching.py`**  
  Demonstrates a local caching strategy to bypass expensive pre-processing (like resizing or tokenization) after the first epoch.  
  *Goal:* Prevent GPU starvation caused by slow data loaders.

### 3. The Operational Levers
*Safety nets for Spot Instances and expensive runs.*
- **`03_smoke_test.py`**  
  The "Smoke Test" Protocol. Runs a cheap CPU-based dry run to catch shape mismatches and OOM bugs before you provision expensive GPU instances.
- **`04_checkpointing_early_stopping.py`**  
  Implements robust checkpointing (for Spot Instance recovery) and Early Stopping logic to prevent "polishing noise."

### 4. The Rapid-Fire Checklist
*Tactical scripts for the "long tail" of optimization.*
- **`05_dynamic_batch_sizer.py`** 
  Automatically probes VRAM at launch to find the maximum safe batch size.
- **`06_profiler_demo.py`**   
  A template for PyTorch Profiler to identify bottlenecks in the training loop.
- **`07_data_deduplication.py`**  
  A hashing utility to remove near-duplicate samples from raw datasets.
- **`08_stale_artifact_cleanup.py`**  
  A utility to auto-archive or delete checkpoints older than N days to reduce storage costs.
---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- NVIDIA GPU (Recommended for AMP demos, but scripts will gracefully fallback to CPU)

### Installation
```bash
git clone https://github.com/Jayachander123/green-ai-optimization-toolkit.git
cd green-ai-optimization-toolkit
pip install -r requirements.txt
