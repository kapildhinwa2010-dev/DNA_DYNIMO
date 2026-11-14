"""Train DNATransformer on a small synthetic copy task.

The default trains materialized dense matrices. Pass --regenerate to train the
shared FourierDNA generator through live generated weights; this is much slower.
The copy task is only a functional smoke/demo, not a language-model benchmark.
"""

import argparse
import math
from pathlib import Path
import time

import torch
from dna_dynimo.dna_transformer import DNATransformer

VOCAB = 16
SEQ = 16
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_batch(batch_size=32, device="cpu"):
    """Random sequences with the sequence itself as the target."""
    tokens = torch.randint(0, VOCAB, (batch_size, SEQ), device=device)
    return tokens, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "saved_models" / "dna_transformer.pt")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}; shared-generator training: {args.regenerate}")
    model = DNATransformer(
        vocab_size=VOCAB,
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=256,
        max_seq_len=SEQ,
        hidden=64,
        num_freqs=32,
        regenerate=args.regenerate,
    ).to(device)
    print(f"DNA storage: {model.dna_storage_params():,} parameters")
    print(f"Generated matrix elements per forward: {model.dna_generated_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = make_batch(device=device)
        _, loss = model(inputs, targets=targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | ppl {math.exp(loss.item()):.2f} | {time.time() - start:.0f}s")

    model.eval()
    prompt = torch.randint(0, VOCAB, (1, SEQ // 2), device=device)
    output = prompt.clone()
    with torch.inference_mode():
        for _ in range(SEQ // 2):
            logits, _ = model(output)
            output = torch.cat((output, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)

    print(f"Copy-task prompt: {prompt[0].tolist()}")
    print(f"Greedy output:    {output[0].tolist()}")
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved checkpoint: {output_path}")


if __name__ == "__main__":
    main()
