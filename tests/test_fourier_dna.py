"""
Unit tests for FourierDNA + Complexity Router + HKSP.
Run: python -m pytest test_fourier_dna.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dna_dynimo.fourier_dna import FourierDNA
from dna_dynimo.complexity_router import DNAEngine, ComplexityRouter


def test_shapes():
    """Generated matrix ka shape sahi hona chahiye."""
    dna = FourierDNA(hidden=64, num_freqs=32)
    with torch.inference_mode():
        for out_dim, in_dim in [(8, 8), (64, 32), (256, 128)]:
            W = dna.generate_weights(out_dim, in_dim)
            assert W.shape == (out_dim, in_dim), f"shape mismatch: {W.shape}"
    print("✅ test_shapes PASSED")


def test_determinism():
    """Same DNA, same call → same weights."""
    torch.manual_seed(0)
    dna = FourierDNA(hidden=64, num_freqs=32)
    with torch.inference_mode():
        W1 = dna.generate_weights(64, 64)
        W2 = dna.generate_weights(64, 64)
    assert torch.allclose(W1, W2), "generation deterministic nahi hai"
    print("✅ test_determinism PASSED")


def test_compression_ratio():
    """DNA params << generated params."""
    torch.manual_seed(0)
    dna = FourierDNA(hidden=32, num_freqs=16, chunk_size=8192)
    dna_params = dna.count_parameters()
    with torch.inference_mode():
        W = dna.generate_weights(1024, 1024)
    ratio = W.numel() / dna_params
    assert ratio > 100, f"compression bahut kam: {ratio:.1f}x"
    print(f"✅ test_compression_ratio PASSED ({ratio:.0f}x)")


def test_xavier_scale():
    """Generated weights ka std Xavier range mein hona chahiye."""
    torch.manual_seed(0)
    dna = FourierDNA(hidden=128, num_freqs=64)
    with torch.inference_mode():
        W = dna.generate_weights(512, 512)
    expected = (2.0 / (512 + 512)) ** 0.5
    assert 0 < W.std().item() < expected * 10, f"scale galat: std={W.std():.4f}"
    print("✅ test_xavier_scale PASSED")


def test_router_end_to_end():
    """Router: text → complexity → allocated params → weights.

    🐛 FIX: max_params=12M pass karna zaroori hai — warna RAM-based limit
    (16GB machine pe ~3B params) pe router 19GB+ weights generate karne
    ki koshish karta aur test OOM/hang ho jata.
    """
    engine = DNAEngine(hidden=64, num_freqs=32)
    router = ComplexityRouter(engine)
    budget = 200_000
    result = router.route("Explain quantum entanglement in detail.", max_params=budget)
    assert 0.0 <= result['complexity'] <= 1.0
    assert result['allocated_params'] > 0
    assert result['allocated_params'] <= budget
    assert result['generated_params'] <= result['allocated_params']
    assert len(result['weights']) == result['num_layers']
    print("✅ test_router_end_to_end PASSED")


def test_router_is_deterministic_and_validates_budget():
    router = ComplexityRouter(DNAEngine(hidden=16, num_freqs=8))
    text = "Compare these matrix products: A @ B and C @ D."
    assert router.calculate_complexity(text) == router.calculate_complexity(text)
    try:
        router.route("too small", max_params=10)
    except ValueError:
        pass
    else:
        raise AssertionError("router accepted a budget smaller than one complete layer")


if __name__ == "__main__":
    test_shapes()
    test_determinism()
    test_compression_ratio()
    test_xavier_scale()
    test_router_end_to_end()
    test_router_is_deterministic_and_validates_budget()
    print("\n🎉 ALL TESTS PASSED")
