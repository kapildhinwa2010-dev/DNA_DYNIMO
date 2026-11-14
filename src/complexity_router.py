# ============================================================
# 🧠 COMPLEXITY ROUTER + DNA ENGINE
# Adaptive Parameter Generation Based on Input Complexity
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import numpy as np
import os

# ============================================================
# COMPONENT 1: RESOURCE MONITOR (VRAM/RAM Scanner)
# ============================================================
class ResourceMonitor:
    """
    System ka VRAM aur RAM scan karta hai aur calculate karta hai
    ki maximum kitne parameters generate kar sakte hain.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Device: {self.device}")

    def get_available_memory(self):
        """Available VRAM (GPU) ya RAM (CPU) return karta hai (GB mein)"""
        if self.device == "cuda":
            # GPU VRAM check
            free, total = torch.cuda.mem_get_info()
            available_gb = free / (1024**3)
            print(f"💾 Available VRAM: {available_gb:.2f} GB")
            return available_gb
        else:
            # CPU RAM check
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            print(f"💾 Available RAM: {available_gb:.2f} GB")
            return available_gb

    def calculate_max_params(self, safety_margin=0.8):
        """
        Calculate karta hai ki kitne parameters safely generate kar sakte hain.

        Args:
            safety_margin: Kitna memory buffer rakhna hai (0.8 = 80% use karo)

        Returns:
            max_params: Maximum parameters jo generate kar sakte hain
        """
        available_gb = self.get_available_memory()

        # FP32 = 4 bytes per parameter
        bytes_per_param = 4
        max_params = int((available_gb * safety_margin * 1024**3) / bytes_per_param)

        print(f"📊 Max Parameters Possible: {max_params:,} ({max_params/1e6:.1f}M)")
        return max_params


# ============================================================
# COMPONENT 2: KNOWLEDGE CLASSIFIER (Input Analyzer)
# ============================================================
class KnowledgeClassifier(nn.Module):
    """
    Ek lightweight neural network jo input text ko analyze karke
    complexity score (0.0 to 1.0) return karta hai.
    """
    def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128):
        super().__init__()

        # Simple embedding + classifier
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: 0.0 to 1.0
        )

    def forward(self, input_ids):
        """
        Args:
            input_ids: Tokenized input (batch_size, seq_len)

        Returns:
            complexity_score: 0.0 (simple) to 1.0 (complex)
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        # Mean pooling (average over sequence)
        pooled = embedded.mean(dim=1)  # (batch, embed_dim)

        # Classification
        score = self.classifier(pooled)  # (batch, 1)

        return score.squeeze(-1)  # (batch,)


# ============================================================
# COMPONENT 3: COMPLEXITY ROUTER (The Brain)
# ============================================================
class ComplexityRouter:
    """
    Master controller jo sab kuch decide karta hai:
    1. Hardware scan
    2. Input complexity calculation
    3. DNA ko parameter generation order
    """
    def __init__(self, dna_engine):
        self.resource_monitor = ResourceMonitor()
        self.knowledge_classifier = KnowledgeClassifier()
        self.dna_engine = dna_engine

        # Simple tokenizer (character-level for demo)
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """Simple character-level vocabulary"""
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-"
        vocab = {char: idx for idx, char in enumerate(chars)}
        vocab['<unk>'] = len(chars)
        return vocab

    def tokenize(self, text, max_len=100):
        """Text ko token IDs mein convert karta hai"""
        tokens = [self.vocab.get(char, self.vocab['<unk>']) for char in text]

        # Pad or truncate
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return torch.tensor(tokens).unsqueeze(0)  # (1, max_len)

    def calculate_complexity(self, text):
        """
        Input text ki complexity calculate karta hai.

        Returns:
            complexity_score: 0.0 (simple) to 1.0 (complex)
        """
        # Step 1: Tokenize
        input_ids = self.tokenize(text)

        # Step 2: Get knowledge score from classifier
        with torch.no_grad():
            knowledge_score = self.knowledge_classifier(input_ids).item()

        # Step 3: Calculate complexity based on multiple factors
        input_length = len(text)
        length_factor = min(input_length / 500.0, 1.0)  # Normalize

        # Complexity formula
        # Low knowledge + Long input = High complexity
        complexity = (
            0.4 * (1.0 - knowledge_score) +  # Low knowledge = high complexity
            0.4 * length_factor +             # Long input = high complexity
            0.2 * np.random.random()          # Small randomness for demo
        )

        complexity = min(max(complexity, 0.0), 1.0)  # Clamp to [0, 1]

        print(f"📝 Input: '{text[:50]}...'")
        print(f"🧠 Knowledge Score: {knowledge_score:.3f}")
        print(f"📏 Length Factor: {length_factor:.3f}")
        print(f"🎯 Final Complexity: {complexity:.3f}")

        return complexity

    def allocate_parameters(self, complexity, max_params=None):
        """
        Complexity ke hisaab se kitne parameters generate karne hain, decide karta hai.

        Args:
            complexity: 0.0 to 1.0
            max_params: Maximum allowed parameters (optional)

        Returns:
            allocated_params: Kitne parameters generate karne hain
        """
        if max_params is None:
            max_params = self.resource_monitor.calculate_max_params()

        # Simple linear mapping: complexity * max_params
        allocated = int(complexity * max_params)

        # Minimum threshold (at least 10M params)
        allocated = max(allocated, 10_000_000)

        print(f"💡 Allocated Parameters: {allocated:,} ({allocated/1e6:.1f}M)")

        return allocated

    def route(self, text):
        """
        Master function: Input leke DNA ko order deta hai.

        Args:
            text: User input text

        Returns:
            allocated_params: Kitne parameters generate hue
        """
        print("\n" + "="*60)
        print("🧠 COMPLEXITY ROUTER ACTIVATED")
        print("="*60)

        # Step 1: Calculate complexity
        complexity = self.calculate_complexity(text)

        # Step 2: Check hardware limits
        max_params = self.resource_monitor.calculate_max_params()

        # Step 3: Allocate parameters
        allocated = self.allocate_parameters(complexity, max_params)

        # Step 4: Order DNA to generate weights
        print(f"\n🧬 Ordering DNA Engine to generate {allocated:,} parameters...")

        # Calculate model dimensions based on allocated params
        # Simplified: assume 12 layers, calculate hidden_dim
        num_layers = 12
        params_per_layer = allocated // num_layers

        # params_per_layer ≈ hidden_dim^2 * 4 (for Q,K,V,O projections)
        hidden_dim = int(np.sqrt(params_per_layer / 4))
        hidden_dim = max(hidden_dim, 256)  # Minimum 256

        print(f"📐 Model Config: {num_layers} layers × {hidden_dim} hidden_dim")

        # Generate weights using DNA
        weights = self.dna_engine.generate_adaptive_weights(
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )

        print("✅ Weights generated successfully!")

        return {
            'complexity': complexity,
            'allocated_params': allocated,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'weights': weights
        }


# ============================================================
# COMPONENT 4: DNA ENGINE (Weight Generator)
# ============================================================
class DNAEngine(nn.Module):
    """
    FourierDNA-based weight generator jo Complexity Router ke orders follow karta hai.
    """
    def __init__(self, hidden=128, num_freqs=64):
        super().__init__()

        self.hidden = hidden
        self.num_freqs = num_freqs

        # Fourier projection
        self.register_buffer('B', torch.randn(num_freqs, 2) * 20.0)
        input_dim = num_freqs * 2

        # DNA network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

        self.scale = nn.Parameter(torch.tensor(0.01))

    def generate_weight_matrix(self, out_dim, in_dim):
        """Single weight matrix generate karta hai"""
        device = self.B.device

        i = torch.linspace(-1, 1, out_dim, device=device)
        j = torch.linspace(-1, 1, in_dim, device=device)
        grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
        coords = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)

        # Fourier features
        proj = 2 * np.pi * coords @ self.B.T
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        # Generate weights
        raw_W = self.net(feats).view(out_dim, in_dim)

        # Xavier scaling
        scale_factor = (2.0 / (in_dim + out_dim)) ** 0.5
        W = raw_W * self.scale * scale_factor

        return W

    def generate_adaptive_weights(self, num_layers, hidden_dim):
        """
        Complexity Router ke order ke hisaab se weights generate karta hai.

        Args:
            num_layers: Kitne layers chahiye
            hidden_dim: Har layer ka dimension

        Returns:
            weights: Dictionary of generated weight matrices
        """
        print(f"\n🧬 DNA Engine: Generating {num_layers} layers × {hidden_dim}×{hidden_dim}")

        weights = {}
        total_params = 0

        for layer_idx in range(num_layers):
            # Generate Q, K, V, O projection matrices for attention
            W_q = self.generate_weight_matrix(hidden_dim, hidden_dim)
            W_k = self.generate_weight_matrix(hidden_dim, hidden_dim)
            W_v = self.generate_weight_matrix(hidden_dim, hidden_dim)
            W_o = self.generate_weight_matrix(hidden_dim, hidden_dim)

            # Generate FFN weights
            W_ffn1 = self.generate_weight_matrix(hidden_dim * 4, hidden_dim)
            W_ffn2 = self.generate_weight_matrix(hidden_dim, hidden_dim * 4)

            weights[f'layer_{layer_idx}'] = {
                'W_q': W_q,
                'W_k': W_k,
                'W_v': W_v,
                'W_o': W_o,
                'W_ffn1': W_ffn1,
                'W_ffn2': W_ffn2
            }

            layer_params = (
                W_q.numel() + W_k.numel() + W_v.numel() + W_o.numel() +
                W_ffn1.numel() + W_ffn2.numel()
            )
            total_params += layer_params

            print(f"  ✓ Layer {layer_idx+1}/{num_layers}: {layer_params:,} params")

        print(f"\n✅ Total Generated: {total_params:,} parameters ({total_params/1e6:.1f}M)")

        return weights


# ============================================================
# DEMO: COMPLETE SYSTEM IN ACTION
# ============================================================
def demo():
    """Complete system ka demonstration"""

    print("="*60)
    print("🚀 DNA COMPLEXITY ROUTER - DEMO")
    print("="*60)

    # Initialize components
    dna_engine = DNAEngine(hidden=128, num_freqs=64)
    router = ComplexityRouter(dna_engine)

    # Test cases
    test_inputs = [
        "Hello! How are you?",  # Simple
        "What is the capital of France?",  # Medium
        "Explain quantum entanglement and its implications for quantum computing in detail.",  # Complex
    ]

    for text in test_inputs:
        result = router.route(text)
        print("\n" + "-"*60 + "\n")

    print("="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)


# Run the demo
if __name__ == "__main__":
    demo()
