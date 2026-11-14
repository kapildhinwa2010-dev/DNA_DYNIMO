"""
Example: Load and test 1B model
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from src.model_generator import Frozen1BModel

print("="*70)
print("🧪 LOADING & TESTING 1B MODEL")
print("="*70)

# Load checkpoint
print("\n1. Loading checkpoint...")
checkpoint = torch.load('frozen_1b_model.pt', map_location='cpu')
print(f"   ✅ Loaded! Keys: {len(checkpoint['model_state_dict'])}")

# Create model
print("\n2. Creating model structure...")
config = checkpoint['model_config']
model = Frozen1BModel(d=config['d'], num_layers=config['num_layers'])

# Load weights
print("3. Loading weights...")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"   ✅ Model ready! Total params: {total_params:,}")

# Test forward pass
print("\n4. Testing forward pass...")
x_test = torch.randn(1, 1)

# Warmup
for _ in range(3):
    with torch.no_grad():
        _ = model(x_test)

# Measure
print("   Measuring inference time (10 passes)...")
times = []
for _ in range(10):
    start = time.time()
    with torch.no_grad():
        output = model(x_test)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
print(f"   ✅ Average inference time: {avg_time*1000:.2f} ms")
print(f"   Output: {output.item():.4f}")

# Memory
import psutil
process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024 / 1024

print("\n" + "="*70)
print("✅ TEST COMPLETE!")
print("="*70)
print(f"Inference Time: {avg_time*1000:.2f} ms")
print(f"Memory Usage: {mem_mb:.1f} MB")
