"""Generate and load dense models initialized from FourierDNA outputs.

The 1B-parameter preset requires several GiB of available RAM. This module
checks an estimate before starting, but it keeps the complete state dict in
memory and is not a sharded/streaming checkpoint writer.
"""

import gc
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import torch.nn as nn


class Frozen1BModel(nn.Module):
    """Dense feed-forward model used by the materialized-checkpoint example."""

    def __init__(self, d=8192, num_layers=15):
        super().__init__()
        if not isinstance(d, int) or not isinstance(num_layers, int) or d < 1 or num_layers < 1:
            raise ValueError("d and num_layers must be positive integers")
        self.d = d
        self.num_layers = num_layers
        self.input_proj = nn.Linear(1, d)
        self.output_proj = nn.Linear(d, 1)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(num_layers)])

    def forward(self, x):
        hidden = torch.relu(self.input_proj(x))
        for layer in self.layers:
            hidden = torch.relu(layer(hidden))
        return self.output_proj(hidden)


def estimate_model_parameters(num_layers, layer_size):
    """Return exact dense parameter count for the checkpoint architecture."""
    if not isinstance(num_layers, int) or not isinstance(layer_size, int):
        raise TypeError("num_layers and layer_size must be integers")
    if num_layers < 1 or layer_size < 1:
        raise ValueError("num_layers and layer_size must be positive")
    return num_layers * (layer_size * layer_size + layer_size) + 3 * layer_size + 1


def generate_1b_model(dna, output_path, num_layers=15, layer_size=8192):
    """Materialize a dense model initialized from FourierDNA and save atomically.

    The function is inference-only with respect to the generator. It stores all
    model tensors in host RAM before serialization, so this is not a streaming
    checkpoint writer. A RAM preflight rejects runs that appear unsafe.
    """
    if not isinstance(dna, torch.nn.Module):
        raise TypeError("dna must be a torch.nn.Module")
    total_params = estimate_model_parameters(num_layers, layer_size)
    estimated_bytes = total_params * 4
    # Leave headroom for Python, model buffers, serialization and OS processes.
    required_bytes = int(estimated_bytes * 1.25) + 512 * 1024**2
    available_bytes = psutil.virtual_memory().available
    if available_bytes < required_bytes:
        raise MemoryError(
            "Insufficient available RAM for dense checkpoint generation: "
            f"estimated weights={estimated_bytes / 1024**3:.2f} GiB, "
            f"recommended free RAM={required_bytes / 1024**3:.2f} GiB, "
            f"currently available={available_bytes / 1024**3:.2f} GiB. "
            "Use a smaller layer_size/num_layers or a sharded generation strategy."
        )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimated_disk_bytes = int(estimated_bytes * 1.10) + 64 * 1024**2
    available_disk_bytes = shutil.disk_usage(output_path.parent).free
    if available_disk_bytes < estimated_disk_bytes:
        raise OSError(
            "Insufficient free disk space for checkpoint generation: "
            f"estimated need={estimated_disk_bytes / 1024**3:.2f} GiB, "
            f"available={available_disk_bytes / 1024**3:.2f} GiB"
        )
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    start_time = time.time()
    state_dict = {
        "input_proj.weight": torch.randn(layer_size, 1) * 0.01,
        "input_proj.bias": torch.zeros(layer_size),
        "output_proj.weight": torch.randn(1, layer_size) * 0.01,
        "output_proj.bias": torch.zeros(1),
    }

    print(f"Generating {total_params:,} dense parameters ({estimated_bytes / 1024**3:.2f} GiB weights).")
    try:
        dna.eval()
        with torch.inference_mode():
            for layer_index in range(num_layers):
                print(f"  Layer {layer_index + 1}/{num_layers}...", end=" ", flush=True)
                generated = dna.generate_weights(layer_size, layer_size)
                state_dict[f"layers.{layer_index}.weight"] = generated.detach().to("cpu").contiguous()
                state_dict[f"layers.{layer_index}.bias"] = torch.zeros(layer_size)
                del generated
                gc.collect()
                print("done")

        generation_time = time.time() - start_time
        checkpoint = {
            "model_state_dict": state_dict,
            "model_config": {"d": layer_size, "num_layers": num_layers},
            "metadata": {
                "total_parameters": total_params,
                "generation_time_seconds": generation_time,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        save_start = time.time()
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, output_path)
        save_time = time.time() - save_start
        file_size_gb = output_path.stat().st_size / 1024**3
        print(f"Saved checkpoint: {output_path} ({file_size_gb:.2f} GiB)")
        return {
            "total_parameters": total_params,
            "generation_time": generation_time,
            "save_time": save_time,
            "file_size_gb": file_size_gb,
            "output_path": str(output_path),
        }
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        del state_dict
        gc.collect()
