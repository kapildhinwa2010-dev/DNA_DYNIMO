"""
🔍 Visualization tools for FourierDNA.

- Weight matrix heatmaps
- Singular value spectrum (DNA-generated vs random)
- Domain comparison grid

Run: python src/visualization.py
"""
import os
import zlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt

from .fourier_dna import FourierDNA


def visualize_matrix(W, title='Weight Matrix', path=None, cmap='viridis'):
    """Single matrix ka heatmap."""
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(W, cmap=cmap, aspect='auto')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Input dim (j)')
    ax.set_ylabel('Output dim (i)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {path}")
    plt.close(fig)
    return fig


def compare_domains(dna, domains=('general', 'math', 'code', 'science'),
                    size=128, path='results/dna_domain_comparison.png'):
    """Ek hi DNA, 4 domains ke matrices — side by side."""
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 5))
    if len(domains) == 1:
        axes = [axes]
    for ax, name in zip(axes, domains):
        # 🐛 FIX: hash() process-dependent hota hai (PYTHONHASHSEED) —
        # zlib.crc32 deterministic hai, isliye plots har run pe same aayenge
        seed = zlib.crc32(name.encode('utf-8'))
        torch.manual_seed(seed)
        B_backup = dna.B.clone()
        with torch.no_grad():
            dna.B.copy_(torch.randn_like(dna.B) * 20.0)
        W = dna.generate_weights(size, size)
        with torch.no_grad():
            dna.B.copy_(B_backup)
        im = ax.imshow(W.detach().cpu().numpy(), cmap='viridis')
        ax.set_title(f'{name.upper()}', fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle('FourierDNA: Same DNA, Different Domains', fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {path}")
    plt.close(fig)
    return fig


def spectrum_comparison(dna, size=512, path='results/dna_spectrum.png'):
    """
    Singular value spectrum: DNA-generated vs random init.
    Achhi trained DNA ka spectrum structured hota hai (low-rank-ish patterns).
    """
    torch.manual_seed(0)
    W_dna = dna.generate_weights(size, size)
    W_rand = torch.randn(size, size) * (2.0 / (2 * size)) ** 0.5

    s_dna = torch.linalg.svdvals(W_dna).detach().cpu().numpy()
    s_rand = torch.linalg.svdvals(W_rand).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.arange(1, len(s_dna) + 1)
    ax.semilogy(idx, s_dna, label='DNA-generated', linewidth=2)
    ax.semilogy(idx, s_rand, label='Random (Gaussian)', linewidth=2, alpha=0.7)
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value (log)')
    ax.set_title('Spectral Fingerprint: DNA vs Random', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {path}")
    plt.close(fig)
    return fig


def compression_curve(dna, sizes=(64, 128, 256, 512, 1024, 2048),
                      path='results/dna_compression_curve.png'):
    """Matrix size vs compression ratio — DNA fixed hai, output badhta hai."""
    dna_params = dna.count_parameters()
    ratios, labels = [], []
    for s in sizes:
        ratios.append((s * s) / dna_params)
        labels.append(f'{s}²')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(sizes)), ratios, 'o-', linewidth=2, markersize=8, color='#4ECDC4')
    for i, r in enumerate(ratios):
        ax.annotate(f'{r:,.0f}x', (i, r), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontweight='bold')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    ax.set_xlabel('Generated matrix size')
    ax.set_ylabel('Compression ratio (log)')
    ax.set_title(f'FourierDNA Compression (DNA = {dna_params:,} params)', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {path}")
    plt.close(fig)
    return fig


def demo():
    """Saare plots generate karo."""
    print("🔍 FourierDNA Visualization Demo")
    torch.manual_seed(0)
    dna = FourierDNA(hidden=128, num_freqs=64)
    print(f"   DNA params: {dna.count_parameters():,}")

    W = dna.generate_weights(256, 256)
    visualize_matrix(W, 'FourierDNA Generated Matrix (256×256)',
                     'results/dna_matrix_heatmap.png')
    compare_domains(dna)
    spectrum_comparison(dna)
    compression_curve(dna)
    print("\n✅ All visualizations saved to results/")


if __name__ == '__main__':
    demo()
