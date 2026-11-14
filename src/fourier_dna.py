"""
DNA Architecture - Fourier-based weight generation
"""
import torch
import torch.nn as nn
import numpy as np

class FourierDNA(nn.Module):
    """
    DNA that generates weight matrices from coordinates using Fourier features.

    Args:
        hidden: Hidden layer size (default: 128)
        num_freqs: Number of Fourier frequencies (default: 64)
    """
    def __init__(self, hidden=128, num_freqs=64):
        super().__init__()
        self.hidden = hidden
        self.num_freqs = num_freqs

        # Fourier projection matrix (fixed)
        self.register_buffer('B', torch.randn(num_freqs, 2) * 20.0)
        input_dim = num_freqs * 2

        # DNA network (trainable)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )

        # Learnable scale
        self.scale = nn.Parameter(torch.tensor(0.01))

    def generate_weights(self, out_dim, in_dim):
        """
        Generate weight matrix of specified size.

        Args:
            out_dim: Output dimension
            in_dim: Input dimension

        Returns:
            Weight matrix of shape (out_dim, in_dim)
        """
        device = self.B.device

        # Generate coordinate grid
        i = torch.linspace(-1, 1, out_dim, device=device)
        j = torch.linspace(-1, 1, in_dim, device=device)
        grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
        coords = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)

        # Fourier features
        proj = 2 * np.pi * coords @ self.B.T
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        # Generate weights
        raw_W = self.net(feats).view(out_dim, in_dim)
        W = raw_W * self.scale

        # Xavier initialization scaling
        scale_factor = (2.0 / (in_dim + out_dim)) ** 0.5
        W = W * scale_factor

        return W

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters())
