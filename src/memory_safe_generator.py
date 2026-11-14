"""
Memory-safe 1B generation: layer-by-layer, gc.collect() har layer ke baad.
(original: DNA repo src/generated.py)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import gc
import time
from src.fourier_dna import FourierDNA

def generate_and_save_1b():
    device = "cpu"  # मेमोरी बचाने के लिए CPU का उपयोग करें
    print(f"🧠 Device: {device} (Using CPU to prevent OOM during 1B generation)")

    # 1. ट्रेन किए गए DNA को लोड करें
    print("📂 Loading Trained DNA...")
    dna = FourierDNA(hidden=128, num_freqs=64).to(device)
    dna.load_state_dict(torch.load("saved_models/trained_dna.pt", map_location=device))
    dna.eval()  # Evaluation mode
    print("✅ DNA loaded successfully!")

    # 2. 1B मॉडल का कॉन्फ़िगरेशन
    num_layers = 15
    layer_size = 8192
    total_params = 0
    state_dict = {}

    print("🧬 Generating 1B Parameters Layer-by-Layer...")
    print("⚠️ This will take a few minutes. Do not close the tab.")
    start_time = time.time()

    # I/O Projections (छोटे लेयर्स)
    state_dict['input_proj.weight'] = torch.randn(layer_size, 1) * 0.01
    state_dict['input_proj.bias'] = torch.zeros(layer_size)
    state_dict['output_proj.weight'] = torch.randn(1, layer_size) * 0.01
    state_dict['output_proj.bias'] = torch.zeros(1)

    # 3. लेयर-बाय-लेयर जनरेशन (Memory Safe)
    for i in range(num_layers):
        print(f"  Generating Layer {i+1}/{num_layers} (8192x8192)...", end=" ", flush=True)

        # DNA से वेट्स जनरेट करें
        with torch.no_grad():
            W = dna.generate_weights(layer_size, layer_size)

        # State dict में डालें (Transposed for nn.Linear)
        state_dict[f'layers.{i}.weight'] = W.T.clone()
        state_dict[f'layers.{i}.bias'] = torch.zeros(layer_size)

        total_params += W.numel() + layer_size

        # 🚨 महत्वपूर्ण: RAM खाली करें ताकि क्रैश न हो
        del W
        gc.collect()

        print("✓ Done")

    gen_time = time.time() - start_time
    print(f"✅ Generated {total_params:,} parameters in {gen_time:.1f} seconds!")

    # 4. सीधे डिस्क पर सेव करें (RAM में पूरा मॉडल लोड नहीं होगा)
    print("💾 Saving 3.75 GB Model to Disk...")
    save_start = time.time()

    os.makedirs("saved_models", exist_ok=True)
    save_path = "saved_models/frozen_1b_model.pt"
    torch.save({
        'model_state_dict': state_dict,
        'config': {'d': layer_size, 'num_layers': num_layers},
        'metadata': {'total_params': total_params, 'gen_time': gen_time}
    }, save_path)

    save_time = time.time() - save_start
    file_size_gb = os.path.getsize(save_path) / (1024**3)

    print(f"✅ Saved in {save_time:.1f} seconds!")
    print(f"📦 File Size: {file_size_gb:.2f} GB")
    print(f"📁 Location: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    generate_and_save_1b()
