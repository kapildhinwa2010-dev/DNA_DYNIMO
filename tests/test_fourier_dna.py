"""
Unit tests for FourierDNA + Complexity Router + HKSP.
Run: python tests/test_fourier_dna.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.fourier_dna import FourierDNA
from src.complexity_router import DNAEngine, ComplexityRouter


def test_shapes():
    """Generated matrix ka shape sahi hona chahiye."""
    dna = FourierDNA(hidden=64, num_freqs=32)
    for out_dim, in_dim in [(8, 8), (64, 32), (256, 128)]:
        W = dna.generate_weights(out_dim, in_dim)
        assert W.shape == (out_dim, in_dim), f"shape mismatch: {W.shape}"
    print("✅ test_shapes PASSED")


def test_determinism():
    """Same DNA, same call → same weights."""
    torch.manual_seed(0)
    dna = FourierDNA(hidden=64, num_freqs=32)
    W1 = dna.generate_weights(64, 64)
    W2 = dna.generate_weights(64, 64)
    assert torch.allclose(W1, W2), "generation deterministic nahi hai"
    print("✅ test_determinism PASSED")


def test_compression_ratio():
    """DNA params << generated params."""
    torch.manual_seed(0)
    dna = FourierDNA(hidden=128, num_freqs=64)
    dna_params = dna.count_parameters()
    W = dna.generate_weights(4096, 4096)
    ratio = W.numel() / dna_params
    assert ratio > 100, f"compression bahut kam: {ratio:.1f}x"
    print(f"✅ test_compression_ratio PASSED ({ratio:.0f}x)")


def test_xavier_scale():
    """Generated weights ka std Xavier range mein hona chahiye."""
    torch.manual_seed(0)
    dna = FourierDNA(hidden=128, num_freqs=64)
    W = dna.generate_weights(512, 512)
    expected = (2.0 / (512 + 512)) ** 0.5
    assert 0 < W.std().item() < expected * 10, f"scale galat: std={W.std():.4f}"
    print("✅ test_xavier_scale PASSED")


def test_router_end_to_end():
    """Router: text → complexity → allocated params → weights."""
    engine = DNAEngine(hidden=64, num_freqs=32)
    router = ComplexityRouter(engine)
    result = router.route("Explain quantum entanglement in detail.")
    assert 0.0 <= result['complexity'] <= 1.0
    assert result['allocated_params'] > 0
    assert len(result['weights']) == result['num_layers']
    print("✅ test_router_end_to_end PASSED")


if __name__ == "__main__":
    test_shapes()
    test_determinism()
    test_compression_ratio()
    test_xavier_scale()
    test_router_end_to_end()
    print("\n🎉 ALL TESTS PASSED")
