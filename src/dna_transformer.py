"""
🧬 DNA Transformer — asli transformer jo DNA-generated weights se chalta hai.

Ab tak sirf GENERATION demonstrate hua tha; yahan pehla working model hai jo
DNA weights se forward + backward karta hai (end-to-end trainable).

Architecture:
  - Embedding + output head: NORMAL trainable params (chhote, fast)
  - Attention Q/K/V/O + FFN weights: DNA se generate (compressed)
  - LayerNorm: normal params (stability ke liye)

Training tip: DNA ko freeze karke sirf embedding/head/LayerNorm train karo
(fast start), ya sab kuch fine-tune karo (full capacity).

⚠️ NOTE (regenerate=True): har forward pe naye nn.Parameter assign hote
hain — optimizer purane param objects ko hold karta hai, isliye fresh
weights optimizer state se bahar rehte hain (sirf wrapper params update
honge). regenerate=True ko experimental "infinite DNA" mode ki tarah use
karo; real training ke liye regenerate=False (default) sahi hai.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_dna import FourierDNA


class DNAAttention(nn.Module):
    """Multi-head attention jiske Q/K/V/O projections DNA se generate hote hain."""

    def __init__(self, dna: FourierDNA, d_model, num_heads, regenerate=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.dna = dna
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.regenerate = regenerate

        self.W_q = nn.Parameter(dna.generate_weights(d_model, d_model))
        self.W_k = nn.Parameter(dna.generate_weights(d_model, d_model))
        self.W_v = nn.Parameter(dna.generate_weights(d_model, d_model))
        self.W_o = nn.Parameter(dna.generate_weights(d_model, d_model))

    def _maybe_regen(self, param, out_dim, in_dim):
        """regenerate=True ho to har forward pe DNA se fresh weights lo.

        ⚠️ Experimental: naya Parameter optimizer ke state se bahar hota
        hai (upar module docstring dekho).
        """
        if self.regenerate and self.training:
            return nn.Parameter(self.dna.generate_weights(out_dim, in_dim))
        return param

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        self.W_q = self._maybe_regen(self.W_q, self.d_model, self.d_model)
        self.W_k = self._maybe_regen(self.W_k, self.d_model, self.d_model)
        self.W_v = self._maybe_regen(self.W_v, self.d_model, self.d_model)
        self.W_o = self._maybe_regen(self.W_o, self.d_model, self.d_model)

        q = (x @ self.W_q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = (x @ self.W_k).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = (x @ self.W_v).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return out @ self.W_o


class DNAFFN(nn.Module):
    """Feed-forward jiske dono matrices DNA se generate hote hain."""

    def __init__(self, dna: FourierDNA, d_model, d_ff, regenerate=False):
        super().__init__()
        self.dna = dna
        self.d_model = d_model
        self.d_ff = d_ff
        self.regenerate = regenerate
        self.W1 = nn.Parameter(dna.generate_weights(d_ff, d_model))
        self.W2 = nn.Parameter(dna.generate_weights(d_model, d_ff))

    def forward(self, x):
        if self.regenerate and self.training:
            self.W1 = nn.Parameter(self.dna.generate_weights(self.d_ff, self.d_model))
            self.W2 = nn.Parameter(self.dna.generate_weights(self.d_model, self.d_ff))
        return F.silu(x @ self.W1) @ self.W2


class DNABlock(nn.Module):
    """Pre-norm transformer block with DNA-generated weights."""

    def __init__(self, dna: FourierDNA, d_model, num_heads, d_ff, regenerate=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = DNAAttention(dna, d_model, num_heads, regenerate)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = DNAFFN(dna, d_model, d_ff, regenerate)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DNATransformer(nn.Module):
    """
    Complete model: DNA weights + minimal trainable wrapper.

    Args:
        vocab_size: vocabulary size
        d_model: model dimension
        num_heads: attention heads
        num_layers: number of blocks
        d_ff: FFN dimension (default 4x)
        max_seq_len: max sequence length
        regenerate: har training step pe DNA se fresh weights (experimental)

    Weight accounting:
        dna_generated_params()  → DNA se aaye weights (compressed idea)
        trainable_params()      → jo actually train hote hain (embeddings, norms, heads)
    """
    def __init__(self, vocab_size=256, d_model=128, num_heads=4, num_layers=4,
                 d_ff=None, max_seq_len=128, regenerate=False, hidden=128, num_freqs=64):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.dna = FourierDNA(hidden=hidden, num_freqs=num_freqs)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            DNABlock(self.dna, d_model, num_heads, d_ff, regenerate)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device)).unsqueeze(0)
        x = tok + pos

        # causal mask
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def trainable_params(self):
        """Embeddings + LayerNorms + head (DNA weights ke alawa sab)."""
        total = 0
        total += self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        total += self.head.weight.numel()
        for block in self.blocks:
            total += sum(p.numel() for p in block.ln1.parameters())
            total += sum(p.numel() for p in block.ln2.parameters())
        total += sum(p.numel() for p in self.ln_f.parameters())
        return total

    def dna_generated_params(self):
        """DNA se generate hue weight params (Q/K/V/O + FFN, saare layers)."""
        total = 0
        for block in self.blocks:
            total += block.attn.W_q.numel() + block.attn.W_k.numel()
            total += block.attn.W_v.numel() + block.attn.W_o.numel()
            total += block.ffn.W1.numel() + block.ffn.W2.numel()
        return total

    def dna_storage_params(self):
        """DNA ka asli storage (yehi asli compression hai)."""
        return self.dna.count_parameters()

    def freeze_dna_weights(self):
        """DNA-generated weights freeze karo — sirf wrapper params train honge (fast)."""
        for block in self.blocks:
            for p in list(block.attn.parameters()) + list(block.ffn.parameters()):
                p.requires_grad = False
        return self
