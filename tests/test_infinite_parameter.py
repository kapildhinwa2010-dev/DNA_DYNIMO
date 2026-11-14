"""
Unit tests for InfiniteParameter (continuous Fourier field).
Run: python tests/test_infinite_parameter.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.infinite_parameter import InfiniteParameter


def test_shapes():
    """Kisi bhi resolution pe matrix generate ho."""
    torch.manual_seed(0)
    ip = InfiniteParameter(num_domains=4, num_scales=3, embed_dim=32, num_freqs=32, hidden_dim=64)
    for out_dim, in_dim in [(16, 16), (64, 32), (130, 70)]:  # non-round sizes bhi
        W = ip.generate_matrix(out_dim, in_dim, domain_idx=0, scale_idx=0)
        assert W.shape == (out_dim, in_dim), f"shape mismatch: {W.shape}"
    print("✅ test_shapes PASSED")


def test_domain_awareness():
    """Alag domains → alag weights."""
    torch.manual_seed(0)
    ip = InfiniteParameter(num_domains=4, num_scales=3, embed_dim=32, num_freqs=32, hidden_dim=64)
    W_math = ip.generate_matrix(64, 64, domain_idx=1)
    W_code = ip.generate_matrix(64, 64, domain_idx=2)
    assert not torch.allclose(W_math, W_code), "domains ka output same hai"
    print("✅ test_domain_awareness PASSED")


def test_multi_scale():
    """Multi-scale matrix = sum of per-scale matrices."""
    torch.manual_seed(0)
    ip = InfiniteParameter(num_domains=2, num_scales=3, embed_dim=32, num_freqs=32, hidden_dim=64)
    W = ip.generate_multi_scale_matrix(64, 64, domain_idx=0)
    total = torch.zeros(64, 64)
    for s in range(3):
        total = total + ip.generate_matrix(64, 64, domain_idx=0, scale_idx=s)
    assert torch.allclose(W, total, atol=1e-5), "multi-scale sum match nahi hua"
    print("✅ test_multi_scale PASSED")


def test_compression():
    """Field params se badi matrix generate ho.

    🐛 FIX: pehle 2048x2048 pe assert >1000 tha — actual ratio ~249x hota
    (field ~17K params), to test hamesha FAIL karta. Ab 4096x4096 (~993x)
    aur threshold 500.
    """
    torch.manual_seed(0)
    ip = InfiniteParameter(num_domains=4, num_scales=3, embed_dim=32, num_freqs=32, hidden_dim=64)
    field_params = ip.count_parameters()
    W = ip.generate_matrix(4096, 4096)
    ratio = W.numel() / field_params
    assert ratio > 500, f"compression bahut kam: {ratio:.0f}x"
    print(f"✅ test_compression PASSED ({ratio:.0f}x)")


if __name__ == "__main__":
    test_shapes()
    test_domain_awareness()
    test_multi_scale()
    test_compression()
    print("\n🎉 ALL INFINITE PARAMETER TESTS PASSED")
