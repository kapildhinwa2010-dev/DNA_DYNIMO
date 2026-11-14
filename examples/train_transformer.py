"""
Train DNATransformer on a toy copy task — proof that DNA weights learn.

Toy task: model ko sequence copy karna sikhao.
Yeh CPU pe minutes mein chalta hai aur validate karta hai ki
DNA-generated weights se asli language modeling possible hai.

Run: python examples/train_transformer.py
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dna_transformer import DNATransformer

VOCAB = 16
SEQ = 16
STEPS = 500


def make_batch(batch_size=32, device='cpu'):
    """Random sequences — target = input khud (copy task, teacher forcing)."""
    x = torch.randint(0, VOCAB, (batch_size, SEQ), device=device)
    return x, x


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Device: {device}")
    print("🧬 Training DNATransformer on copy task...")

    model = DNATransformer(
        vocab_size=VOCAB, d_model=64, num_heads=4, num_layers=3,
        d_ff=256, max_seq_len=SEQ + 1,
        hidden=64, num_freqs=32,
        regenerate=False,   # True karo to har step DNA se fresh weights (slow but "infinite DNA")
    ).to(device)

    print(f"   DNA storage:      {model.dna_storage_params():,} params")
    print(f"   DNA generated:    {model.dna_generated_params():,} params")
    print(f"   Trainable wrapper:{model.trainable_params():,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    start = time.time()
    for step in range(1, STEPS + 1):
        x, y = make_batch(device=device)
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0 or step == 1:
            print(f"   step {step:4d} | loss {loss.item():.4f} | "
                  f"ppl {math.exp(loss.item()):.2f} | {time.time()-start:.0f}s")

    # Evaluation: greedy generation
    model.eval()
    prompt = torch.randint(0, VOCAB, (1, SEQ // 2), device=device)
    out = prompt.clone()
    for _ in range(SEQ // 2):
        logits, _ = model(out)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)

    print("\n🎲 Sample generation (copy task):")
    print(f"   prompt: {prompt[0].tolist()}")
    print(f"   output: {out[0].tolist()}")

    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'saved_models/dna_transformer.pt')
    print("\n✅ Training complete! Saved: saved_models/dna_transformer.pt")


if __name__ == "__main__":
    main()
