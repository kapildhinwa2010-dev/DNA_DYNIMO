"""Load and benchmark a checkpoint created by generate_1b_model.py."""

import argparse
from pathlib import Path
import time

import psutil
import torch

from dna_dynimo.model_generator import Frozen1BModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or not {"model_config", "model_state_dict"}.issubset(checkpoint):
        raise ValueError("Unsupported checkpoint: expected model_config and model_state_dict")
    config = checkpoint["model_config"]
    if not isinstance(config, dict) or not {"d", "num_layers"}.issubset(config):
        raise ValueError("Checkpoint model_config must contain d and num_layers")

    model = Frozen1BModel(d=int(config["d"]), num_layers=int(config["num_layers"]))
    model.load_state_dict(checkpoint["model_state_dict"], assign=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "saved_models" / "frozen_1b_model.pt",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    x = torch.randn(1, 1)
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            output = model(x)
            timings.append(time.perf_counter() - start)

    average_ms = sum(timings) / len(timings) * 1000
    memory_mb = psutil.Process().memory_info().rss / 1024**2
    print(f"Model parameters: {parameter_count:,}")
    print(f"Average CPU inference: {average_ms:.2f} ms")
    print(f"Output: {output.item():.4f}")
    print(f"Resident memory: {memory_mb:.1f} MiB")


if __name__ == "__main__":
    main()
