"""
Train a FourierDNA to memorize a structured target matrix.
(original: DNA repo src/traindna.py)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from src.fourier_dna import FourierDNA

def train_dna():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Device: {device}")

    # 1. DNA को initialize करें
    dna = FourierDNA(hidden=128, num_freqs=64).to(device)
    print(f"🧬 DNA Parameters: {dna.count_parameters():,}")

    # 2. एक Target मैट्रिक्स बनाएं (जिसे DNA को सीखना है)
    # हम एक smooth wave pattern बना रहे हैं (256x256)
    size = 256
    i = torch.linspace(-1, 1, size, device=device)
    j = torch.linspace(-1, 1, size, device=device)
    grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')

    # Target: एक structured pattern (जैसे न्यूरल नेटवर्क्स में अक्सर होता है)
    target_W = torch.sin(grid_i * 5) * torch.cos(grid_j * 5)
    target_W = target_W * (2.0 / (size + size)) ** 0.5  # Xavier scale

    # 3. ट्रेनिंग सेटअप
    optimizer = torch.optim.Adam(dna.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    epochs = 1000

    print("🚀 Starting DNA Training...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        # DNA से 256x256 मैट्रिक्स जनरेट करें
        pred_W = dna.generate_weights(size, size)

        # Loss निकालें
        loss = loss_fn(pred_W, target_W)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    # 4. ट्रेन किए गए DNA को सेव करें
    os.makedirs("saved_models", exist_ok=True)
    save_path = "saved_models/trained_dna.pt"
    torch.save(dna.state_dict(), save_path)
    print(f"✅ DNA trained and saved to: {save_path}")

if __name__ == "__main__":
    train_dna()
