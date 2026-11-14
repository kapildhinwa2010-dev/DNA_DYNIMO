# 🧬 DNA_DYNIMO — FourierDNA + Complexity Router

**Ek chhota "DNA" (~50 KB) jo kisi bhi size ki weight matrix on-demand generate karta hai — aur ek Complexity Router jo decide karta hai ki kab kitne parameters generate karne hain.**

Ye repo ek self-contained, copy-friendly toolkit hai: koi bhi is code ko copy karke apna **FourierDNA + Complexity Router** system bana sakta hai. Source tech: [kapildhinwa2010-dev/DNA](https://github.com/kapildhinwa2010-dev/DNA).

---

## 📐 Core Concept

Weight matrix ko image ki tarah treat karo. Har cell ka ek coordinate (i, j) hai:

    W[i,j] = net( sin(2π·(i,j)·Bᵀ), cos(2π·(i,j)·Bᵀ) ) · scale

1. **Coordinate grid** — matrix ki har cell (i, j) ∈ [-1, 1]
2. **Fourier projection** — fixed random matrix B (num_freqs × 2, scale 20): proj = 2π · coords @ Bᵀ
3. **Features** — [sin(proj), cos(proj)] → 128-dim feature per coordinate (NeRF/SIREN-style encoding)
4. **DNA network** — chhota MLP (128→128→128→1, SiLU) per-cell weight predict karta hai
5. **Xavier scaling** — W ko (2/(in+out))^0.5 se scale karo

**Result:** ~257K trainable params se 1B params generate (≈3913x compression). Jitni badi matrix, utna zyada compression — DNA ka size fixed hai.

## 🧠 HKSP — HyperKnowledge Super Parameter

FourierDNA + domain conditioning:

    W[i,j] = f(coords, domain)

- **HSP part** = Fourier coordinate features (structure)
- **KSP part** = learnable domain embedding (general / math / code / science)

Ek hi super-parameter 4 domains ke alag-alag matrices deta hai. Caching (100x speedup) aur hybrid mode (chhoti layers normal, badi layers HKSP) included.

## 🔀 Complexity Router

Adaptive parameter allocation:

1. **ResourceMonitor** — VRAM/RAM scan → max generate-able params
2. **KnowledgeClassifier** — input text ka complexity score (0–1)
3. **Router** — complexity × hardware limit = kitne params generate karne hain (min 10M), model config (layers × hidden_dim) decide
4. **DNAEngine** — us order ke hisaab se Q/K/V/O + FFN weights generate

Simple question → chhota model. Complex question → bada model. Hardware ke hisaab se capped.

---

## 📁 Structure

    src/
      fourier_dna.py        # FourierDNA — core coordinate→weight generator
      complexity_router.py  # ResourceMonitor + KnowledgeClassifier + Router + DNAEngine
      hksp.py               # HKSP (3 versions: original, optimized+caching, comparison)
      analog_ternary.py     # Analog (signal-based) + Ternary (BitNet-style) parameter layers
      model_generator.py    # 1B model generation + Frozen1BModel loader
    examples/
      generate_1b_model.py  # DNA se 1B model banao
      load_and_test.py      # Frozen model load + inference benchmark
    tests/
      test_fourier_dna.py   # Unit tests

## 🚀 Quickstart

    pip install -r requirements.txt

    # 1. Core tech test karo
    python tests/test_fourier_dna.py

    # 2. Complexity Router demo
    python src/complexity_router.py

    # 3. HKSP comparison (compression vs speed)
    python src/hksp.py

    # 4. 1B model generate karo (GPU recommended)
    python examples/generate_1b_model.py

## 💡 Khud ka system banane ke liye (copy-paste)

    from src.fourier_dna import FourierDNA
    import torch

    dna = FourierDNA(hidden=128, num_freqs=64)
    print(f"DNA params: {dna.count_parameters():,}")   # ~257K

    W = dna.generate_weights(1024, 1024)               # on-demand weight matrix
    print(W.shape)                                     # (1024, 1024)

Router ke saath:

    from src.complexity_router import DNAEngine, ComplexityRouter

    engine = DNAEngine(hidden=128, num_freqs=64)
    router = ComplexityRouter(engine)
    result = router.route("Explain quantum entanglement in detail.")
    # → complexity score, allocated params, generated weights

## 📊 Known trade-offs

- ✅ Massive compression (DNA size fixed, output unlimited)
- ✅ Domain-aware generation (HKSP)
- ⚠️ Training slower (weights generate karne padte hain; caching se mitigate)
- ⚠️ Abhi tak sirf generation demonstrated — real language-task training open research hai

## 📜 License

MIT — apna FourierDNA banao, share karo. 🧬
