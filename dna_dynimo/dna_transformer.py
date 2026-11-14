"""Transformer blocks backed by FourierDNA-generated weight matrices.

With ``regenerate=False`` (the default), DNA outputs are materialized as dense
trainable matrices. Optimizers update those matrices, not the DNA generator.
With ``regenerate=True``, matrices are generated on each forward pass and
backpropagation flows into the shared FourierDNA generator; this is slower and
uses more activation memory.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_dna import FourierDNA


class DNAAttention(nn.Module):
    """Multi-head self-attention whose projection matrices come from FourierDNA."""

    def __init__(self, dna: FourierDNA, d_model, num_heads, regenerate=False):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        self.dna = dna
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.regenerate = regenerate

        if regenerate:
            for name in ("W_q", "W_k", "W_v", "W_o"):
                self.register_parameter(name, None)
        else:
            self.W_q = nn.Parameter(dna.generate_weights(d_model, d_model))
            self.W_k = nn.Parameter(dna.generate_weights(d_model, d_model))
            self.W_v = nn.Parameter(dna.generate_weights(d_model, d_model))
            self.W_o = nn.Parameter(dna.generate_weights(d_model, d_model))

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, width = x.shape
        if width != self.d_model:
            raise ValueError(f"expected input width {self.d_model}, got {width}")
        if self.regenerate:
            w_q = self.dna.generate_weights(self.d_model, self.d_model)
            w_k = self.dna.generate_weights(self.d_model, self.d_model)
            w_v = self.dna.generate_weights(self.d_model, self.d_model)
            w_o = self.dna.generate_weights(self.d_model, self.d_model)
        else:
            w_q, w_k, w_v, w_o = self.W_q, self.W_k, self.W_v, self.W_o

        q = (x @ w_q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = (x @ w_k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = (x @ w_v).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        output = (attention @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, width)
        return output @ w_o


class DNAFFN(nn.Module):
    """Feed-forward block whose two matrices come from FourierDNA."""

    def __init__(self, dna: FourierDNA, d_model, d_ff, regenerate=False):
        super().__init__()
        if d_model < 1 or d_ff < 1:
            raise ValueError("d_model and d_ff must be positive")
        self.dna = dna
        self.d_model = d_model
        self.d_ff = d_ff
        self.regenerate = regenerate
        if regenerate:
            self.register_parameter("W1", None)
            self.register_parameter("W2", None)
        else:
            self.W1 = nn.Parameter(dna.generate_weights(d_model, d_ff))
            self.W2 = nn.Parameter(dna.generate_weights(d_ff, d_model))

    def forward(self, x):
        if self.regenerate:
            w1 = self.dna.generate_weights(self.d_model, self.d_ff)
            w2 = self.dna.generate_weights(self.d_ff, self.d_model)
        else:
            w1, w2 = self.W1, self.W2
        return F.silu(x @ w1) @ w2


class DNABlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, dna: FourierDNA, d_model, num_heads, d_ff, regenerate=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = DNAAttention(dna, d_model, num_heads, regenerate)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = DNAFFN(dna, d_model, d_ff, regenerate)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        return x + self.ffn(self.ln2(x))


class DNATransformer(nn.Module):
    """A compact causal transformer with DNA-generated or materialized weights."""

    def __init__(
        self,
        vocab_size=256,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=None,
        max_seq_len=128,
        regenerate=False,
        hidden=128,
        num_freqs=64,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        if min(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len) < 1:
            raise ValueError("vocab_size, dimensions, layer count, and max_seq_len must be positive")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dna = FourierDNA(hidden=hidden, num_freqs=num_freqs)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [DNABlock(self.dna, d_model, num_heads, d_ff, regenerate) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch_size, sequence_length)")
        batch_size, seq_len = idx.shape
        if seq_len < 1 or seq_len > self.max_seq_len:
            raise ValueError(f"sequence length must be in [1, {self.max_seq_len}]")
        if idx.dtype != torch.long:
            raise TypeError("idx must use torch.long token IDs")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")

        positions = torch.arange(seq_len, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions).unsqueeze(0)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).view(1, 1, seq_len, seq_len)
        for block in self.blocks:
            x = block(x, causal_mask)
        logits = self.head(self.ln_f(x))

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def trainable_params(self):
        """Count wrapper parameters (embeddings, LayerNorms, and output head)."""
        total = self.tok_emb.weight.numel() + self.pos_emb.weight.numel() + self.head.weight.numel()
        for block in self.blocks:
            total += sum(parameter.numel() for parameter in block.ln1.parameters())
            total += sum(parameter.numel() for parameter in block.ln2.parameters())
        total += sum(parameter.numel() for parameter in self.ln_f.parameters())
        return total

    def dna_generated_params(self):
        """Count the theoretical Q/K/V/O and FFN matrix elements."""
        total = 0
        for block in self.blocks:
            d_model, d_ff = block.attn.d_model, block.ffn.d_ff
            total += 4 * d_model * d_model + 2 * d_model * d_ff
        return total

    def dna_storage_params(self):
        """Return the trainable parameter count of the FourierDNA generator."""
        return self.dna.count_parameters()

    def freeze_dna_weights(self):
        """Freeze the generator and materialized DNA matrices; leave wrapper trainable."""
        for parameter in self.dna.parameters():
            parameter.requires_grad = False
        for block in self.blocks:
            for module in (block.attn, block.ffn):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        return self
