"""
Example: Generate 1B model from DNA
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fourier_dna import FourierDNA
from src.model_generator import generate_1b_model

# Create DNA
print("Creating DNA...")
dna = FourierDNA(hidden=128, num_freqs=64)
print(f"DNA parameters: {dna.count_parameters():,}")

# Generate 1B model
print("\nGenerating 1B model...")
result = generate_1b_model(
    dna=dna,
    output_path='frozen_1b_model.pt',
    num_layers=15,
    layer_size=8192
)

print("\n" + "="*70)
print("✅ GENERATION COMPLETE!")
print("="*70)
print(f"Total Parameters: {result['total_parameters']:,}")
print(f"Generation Time: {result['generation_time']:.1f}s")
print(f"File Size: {result['file_size_gb']:.2f} GB")
