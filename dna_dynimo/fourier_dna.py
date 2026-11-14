"""Fourier-feature neural field for generating weight matrices."""

import math

import torch
import torch.nn as nn


class FourierDNA(nn.Module):
    """Generate weight matrices from coordinates using Fourier features.

    Generation is chunked to bound intermediate activation memory. Gradients
    remain enabled by default; wrap inference-only calls in ``torch.no_grad()``
    or ``torch.inference_mode()`` to avoid retaining the autograd graph.

    Args:
        hidden: Hidden layer width.
        num_freqs: Number of Fourier frequencies.
        chunk_size: Maximum number of coordinates processed at once.
    """

    def __init__(self, hidden=128, num_freqs=64, chunk_size=16_384):
        super().__init__()
        if hidden < 1 or num_freqs < 1 or chunk_size < 1:
            raise ValueError("hidden, num_freqs, and chunk_size must be positive integers")

        self.hidden = int(hidden)
        self.num_freqs = int(num_freqs)
        self.chunk_size = int(chunk_size)

        self.register_buffer("B", torch.randn(self.num_freqs, 2) * 20.0)
        input_dim = self.num_freqs * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.01))

    def generate_weights(self, out_dim, in_dim, *, chunk_size=None):
        """Generate a matrix of shape ``(out_dim, in_dim)``.

        Args:
            out_dim: Number of output coordinates; must be positive.
            in_dim: Number of input coordinates; must be positive.
            chunk_size: Optional per-call override for the coordinate batch size.
        """
        if not isinstance(out_dim, int) or not isinstance(in_dim, int):
            raise TypeError("out_dim and in_dim must be integers")
        if out_dim < 1 or in_dim < 1:
            raise ValueError("out_dim and in_dim must be positive")
        batch_size = self.chunk_size if chunk_size is None else chunk_size
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("chunk_size must be a positive integer")

        device, dtype = self.B.device, self.B.dtype
        row_values = torch.linspace(-1, 1, out_dim, device=device, dtype=dtype)
        col_values = torch.linspace(-1, 1, in_dim, device=device, dtype=dtype)
        total = out_dim * in_dim
        flat_indices = torch.arange(total, device=device)
        outputs = []

        for start in range(0, total, batch_size):
            flat = flat_indices[start : start + batch_size]
            rows = torch.div(flat, in_dim, rounding_mode="floor")
            cols = flat.remainder(in_dim)
            coords = torch.stack((row_values[rows], col_values[cols]), dim=-1)
            projection = (2.0 * math.pi) * (coords @ self.B.T)
            features = torch.cat((torch.sin(projection), torch.cos(projection)), dim=-1)
            outputs.append(self.net(features).squeeze(-1))

        raw_weights = torch.cat(outputs).reshape(out_dim, in_dim)
        xavier_scale = math.sqrt(2.0 / (in_dim + out_dim))
        return raw_weights * self.scale * xavier_scale

    def count_parameters(self):
        """Return the number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
