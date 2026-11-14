"""
Model generator: DNA se frozen model banao aur load karo.

Ye file DNA repo ke examples ke saath kaam karne ke liye hai:
  - generate_1b_model(dna, output_path, num_layers, layer_size)
  - Frozen1BModel(d, num_layers)  # load_state_dict ke saath compatible
"""
import os
import time
import torch
import torch.nn as nn

from .fourier_dna import FourierDNA


def generate_1b_model(dna, output_path='frozen_1b_model.pt',
                      num_layers=15, layer_size=8192):
    """
    DNA se ek frozen weight model generate karke .pt file mein save karo.

    Har layer: Q, K, V, O projections + FFN (4x) weights.
    Sab weights DNA se generate hote hain — phir DNA delete ho sakta hai.
    """
    start = time.time()
    state = {}
    total = 0

    print(f"🧬 Generating {num_layers} layers × {layer_size} hidden ...")
    for l in range(num_layers):
        for name in ('W_q', 'W_k', 'W_v', 'W_o'):
            W = dna.generate_weights(layer_size, layer_size)
            state[f'layers.{l}.{name}'] = W.detach().cpu()
            total += W.numel()
        W1 = dna.generate_weights(layer_size * 4, layer_size)
        W2 = dna.generate_weights(layer_size, layer_size * 4)
        state[f'layers.{l}.W_ffn1'] = W1.detach().cpu()
        state[f'layers.{l}.W_ffn2'] = W2.detach().cpu()
        total += W1.numel() + W2.numel()
        print(f"  ✓ Layer {l + 1}/{num_layers}: cumulative {total:,} params")

    elapsed = time.time() - start
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save({
        'model_state_dict': state,
        'model_config': {'d': layer_size, 'num_layers': num_layers},
        'total_parameters': total,
        'generation_time': elapsed,
    }, output_path)

    size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"✅ Saved {output_path}: {total:,} params, {elapsed:.1f}s, {size_gb:.2f} GB")
    return {'total_parameters': total, 'generation_time': elapsed, 'file_size_gb': size_gb}


class Frozen1BModel(nn.Module):
    """
    Frozen model jo generated weights se forward pass karta hai.
    state dict keys: layers.{l}.{W_q,W_k,W_v,W_o,W_ffn1,W_ffn2}
    """
    def __init__(self, d=8192, num_layers=15):
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        for l in range(num_layers):
            self.register_buffer(f'layers_{l}_W_q', torch.zeros(d, d))
            self.register_buffer(f'layers_{l}_W_k', torch.zeros(d, d))
            self.register_buffer(f'layers_{l}_W_v', torch.zeros(d, d))
            self.register_buffer(f'layers_{l}_W_o', torch.zeros(d, d))
            self.register_buffer(f'layers_{l}_W_ffn1', torch.zeros(d * 4, d))
            self.register_buffer(f'layers_{l}_W_ffn2', torch.zeros(d, d * 4))

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            q = h @ getattr(self, f'layers_{l}_W_q').T
            k = h @ getattr(self, f'layers_{l}_W_k').T
            v = h @ getattr(self, f'layers_{l}_W_v').T
            att = torch.softmax(q @ k.T / (self.d ** 0.5), dim=-1) @ v
            h = h + att @ getattr(self, f'layers_{l}_W_o').T
            h = h + torch.nn.functional.silu(h @ getattr(self, f'layers_{l}_W_ffn1').T) @ getattr(self, f'layers_{l}_W_ffn2').T
        return h.sum()
