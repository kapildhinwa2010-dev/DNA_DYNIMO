"""
Unit tests for DNATransformer — DNA weights se asli forward/backward.
Run: python -m pytest test_dna_transformer.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dna_dynimo.dna_transformer import DNATransformer


def _tiny_model(**kw):
    torch.manual_seed(0)
    return DNATransformer(vocab_size=32, d_model=64, num_heads=4, num_layers=2,
                          d_ff=128, max_seq_len=32, hidden=32, num_freqs=16, **kw)


def test_forward_shapes():
    model = _tiny_model()
    idx = torch.randint(0, 32, (2, 16))
    logits, _ = model(idx)
    assert logits.shape == (2, 16, 32), f"bad logits shape: {logits.shape}"
    print("✅ test_forward_shapes PASSED")


def test_loss_decreases():
    """Ek step training pe loss kam hona chahiye (grads flow ho rahe hain)."""
    model = _tiny_model()
    idx = torch.randint(0, 32, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    _, loss0 = model(idx[:, :-1], targets=idx[:, 1:])
    opt.zero_grad(); loss0.backward(); opt.step()
    _, loss1 = model(idx[:, :-1], targets=idx[:, 1:])
    assert loss1.item() < loss0.item(), f"loss kam nahi hua: {loss0:.4f} -> {loss1:.4f}"
    print("✅ test_loss_decreases PASSED")


def test_causal_mask():
    """Position t ka prediction sirf <= t tokens pe depend kare."""
    model = _tiny_model().eval()
    idx = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        logits_full, _ = model(idx)
        idx2 = idx.clone(); idx2[0, -1] = (idx[0, -1] + 1) % 32
        logits_mod, _ = model(idx2)
    assert torch.allclose(logits_full[0, :-1], logits_mod[0, :-1], atol=1e-5), \
        "causal mask leak ho raha hai"
    print("✅ test_causal_mask PASSED")


def test_weight_accounting():
    """DNA-generated params >> DNA storage params (compression story)."""
    model = _tiny_model()
    gen = model.dna_generated_params()
    storage = model.dna_storage_params()
    trainable = model.trainable_params()
    assert gen > storage * 10, f"compression kam: gen={gen:,} storage={storage:,}"
    assert trainable > 0
    print(f"✅ test_weight_accounting PASSED "
          f"(generated={gen:,}, dna_storage={storage:,}, trainable={trainable:,})")


def test_frozen_mode():
    """Freeze mode mein sirf wrapper params train hon."""
    model = _tiny_model().freeze_dna_weights()
    idx = torch.randint(0, 32, (2, 16))
    _, loss = model(idx[:, :-1], targets=idx[:, 1:])
    loss.backward()
    frozen_grads = [n for n, p in model.named_parameters()
                    if ('W_q' in n or 'W_ffn' in n or 'W1' in n or 'W2' in n
                        or 'W_k' in n or 'W_v' in n or 'W_o' in n) and p.grad is not None]
    assert len(frozen_grads) == 0, f"frozen weights pe grad aaya: {frozen_grads[:3]}"
    print("✅ test_frozen_mode PASSED")


def test_regenerate_trains_generator():
    """Regenerate mode must preserve gradients into the shared DNA generator."""
    model = _tiny_model(regenerate=True)
    idx = torch.randint(0, 32, (2, 6))
    _, loss = model(idx[:, :-1], targets=idx[:, 1:])
    loss.backward()
    dna_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.dna.parameters()
        if parameter.grad is not None
    )
    assert dna_grad > 0.0, "DNA generator received no gradients"
    assert all(block.attn.W_q is None for block in model.blocks)
    print("✅ test_regenerate_trains_generator PASSED")


if __name__ == "__main__":
    test_forward_shapes()
    test_loss_decreases()
    test_causal_mask()
    test_weight_accounting()
    test_frozen_mode()
    test_regenerate_trains_generator()
    print("\n🎉 ALL DNA TRANSFORMER TESTS PASSED")
