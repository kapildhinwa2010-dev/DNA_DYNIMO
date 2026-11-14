"""CPU-oriented 1B checkpoint entrypoint.

This reuses the canonical checkpoint format and memory preflight. It is not a
streaming writer: the complete dense state dict must fit in host RAM.
"""

import argparse
from pathlib import Path

import torch

from .fourier_dna import FourierDNA
from .model_generator import generate_1b_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_and_save_1b(
    checkpoint_path=None,
    output_path=None,
    num_layers=15,
    layer_size=8192,
):
    """Load a trained DNA checkpoint and save the canonical dense model format."""
    checkpoint_path = Path(checkpoint_path or PROJECT_ROOT / "saved_models" / "trained_dna.pt")
    output_path = Path(output_path or PROJECT_ROOT / "saved_models" / "frozen_1b_model.pt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained DNA checkpoint not found: {checkpoint_path}. "
            "Run the DNA training example first or pass --checkpoint."
        )

    dna = FourierDNA(hidden=128, num_freqs=64, chunk_size=16_384)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dna.load_state_dict(state_dict)
    dna.eval()
    return generate_1b_model(
        dna=dna,
        output_path=output_path,
        num_layers=num_layers,
        layer_size=layer_size,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "saved_models" / "trained_dna.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "saved_models" / "frozen_1b_model.pt")
    parser.add_argument("--num-layers", type=int, default=15)
    parser.add_argument("--layer-size", type=int, default=8192)
    args = parser.parse_args()
    generate_and_save_1b(args.checkpoint, args.output, args.num_layers, args.layer_size)


if __name__ == "__main__":
    main()
