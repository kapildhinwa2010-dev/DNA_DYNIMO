"""
Model Generator - Generate 1B models from DNA
(original: DNA repo src/modle_genration.py)
"""
import torch
import torch.nn as nn
import os
import time
import gc

class Frozen1BModel(nn.Module):
    """
    Standard frozen model structure for 1B parameters.

    Args:
        d: Layer dimension (default: 8192)
        num_layers: Number of layers (default: 15)
    """
    def __init__(self, d=8192, num_layers=15):
        super().__init__()
        self.d = d
        self.num_layers = num_layers

        self.input_proj = nn.Linear(1, d)
        self.output_proj = nn.Linear(d, 1)
        self.layers = nn.ModuleList([
            nn.Linear(d, d) for _ in range(num_layers)
        ])

    def forward(self, x):
        """Forward pass"""
        h = torch.relu(self.input_proj(x))
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.output_proj(h)


def generate_1b_model(dna, output_path, num_layers=15, layer_size=8192):
    """
    Generate 1B model from DNA and save to disk.

    Args:
        dna: FourierDNA instance
        output_path: Path to save model
        num_layers: Number of layers
        layer_size: Size of each layer
    """
    print("="*70)
    print("🧬 GENERATING 1B MODEL FROM DNA")
    print("="*70)

    total_params = 0
    start_time = time.time()
    state_dict = {}

    # Add I/O projections
    print("Adding input/output projections...")
    state_dict['input_proj.weight'] = torch.randn(layer_size, 1) * 0.01
    state_dict['input_proj.bias'] = torch.zeros(layer_size)
    state_dict['output_proj.weight'] = torch.randn(1, layer_size) * 0.01
    state_dict['output_proj.bias'] = torch.zeros(1)
    total_params += layer_size * 2 + 2

    # Generate layers
    for i in range(num_layers):
        print(f"  Layer {i+1}/{num_layers}...", end=" ", flush=True)

        # Generate weights
        W = dna.generate_weights(layer_size, layer_size)

        # Add to state_dict
        state_dict[f'layers.{i}.weight'] = W.T.clone()
        state_dict[f'layers.{i}.bias'] = torch.zeros(layer_size)

        layer_params = W.numel() + layer_size
        total_params += layer_params

        # Free memory
        del W
        gc.collect()

        print(f"✓ {layer_params:,} params")

    generation_time = time.time() - start_time

    print(f"✅ Generated {total_params:,} parameters in {generation_time:.1f}s")

    # Save to disk
    print("💾 Saving to disk...")
    save_start = time.time()

    torch.save({
        'model_state_dict': state_dict,
        'model_config': {
            'd': layer_size,
            'num_layers': num_layers,
        },
        'metadata': {
            'total_parameters': total_params,
            'generation_time': generation_time,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }, output_path)

    save_time = time.time() - save_start
    file_size_gb = os.path.getsize(output_path) / (1024**3)

    print(f"✅ Saved in {save_time:.1f}s")
    print(f"📦 File size: {file_size_gb:.2f} GB")

    # Free memory
    del state_dict
    gc.collect()

    return {
        'total_parameters': total_params,
        'generation_time': generation_time,
        'save_time': save_time,
        'file_size_gb': file_size_gb
    }
