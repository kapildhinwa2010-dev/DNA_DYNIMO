"""Generate a dense checkpoint initialized from FourierDNA.

Default runs a small smoke example. Use --num-layers 15 --layer-size 8192
--confirm-large only on a machine with several GiB of free RAM.
"""

import argparse
from pathlib import Path

from dna_dynimo.fourier_dna import FourierDNA
from dna_dynimo.model_generator import estimate_model_parameters, generate_1b_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "saved_models" / "frozen_1b_model.pt")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--layer-size", type=int, default=128)
    parser.add_argument("--confirm-large", action="store_true", help="confirm generation of >100M parameters")
    args = parser.parse_args()

    parameter_count = estimate_model_parameters(args.num_layers, args.layer_size)
    if parameter_count > 100_000_000 and not args.confirm_large:
        parser.error(
            f"requested {parameter_count:,} parameters; add --confirm-large after checking available RAM"
        )

    dna = FourierDNA(hidden=128, num_freqs=64, chunk_size=16_384)
    print(f"DNA generator parameters: {dna.count_parameters():,}")
    result = generate_1b_model(
        dna=dna,
        output_path=args.output,
        num_layers=args.num_layers,
        layer_size=args.layer_size,
    )
    print(
        f"Generated {result['total_parameters']:,} parameters in "
        f"{result['generation_time']:.1f}s; checkpoint: {result['output_path']}"
    )


if __name__ == "__main__":
    main()
