# DNA_DYNIMO

[![CI](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO/actions/workflows/ci.yml) [![CodeQL](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO/actions/workflows/codeql.yml) ![Python](https://img.shields.io/badge/python-3.10--3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c) ![License](https://img.shields.io/badge/license-MIT-green)

DNA_DYNIMO is a research toolkit for generating matrices with Fourier-feature neural fields, experimenting with budget-aware allocation, and running a small causal transformer. The project is **not a production language model or a validated serving stack**; real-task quality, training behavior, and performance still require independent evaluation.

## Components

### FourierDNA

`FourierDNA` maps normalized matrix coordinates through Fourier features and an MLP. Generation is chunked to limit intermediate memory while preserving autograd. The default generator has about **49K trainable parameters**; generated matrices are dense tensors and still require memory proportional to their output size. The parameter ratio alone is not evidence of equivalent model quality or effective compression in a deployed model.

### InfiniteParameter

`InfiniteParameter` adds domain and scale embeddings and supports chunked matrix generation. It can generate different resolutions, but “infinite” describes the coordinate field concept—not infinite compute, precision, or output storage. Use `torch.no_grad()` / `torch.inference_mode()` for inference-only generation.

### DNATransformer

The transformer has two distinct operating modes:

- `regenerate=False` (default) materializes dense matrices at initialization. Training updates those matrices and the wrapper; it **does not train the FourierDNA generator**.
- `regenerate=True` generates matrices on each forward pass and propagates gradients into the shared FourierDNA generator. This is substantially more computationally expensive.

The included copy-task example is a smoke/demo task, not evidence of useful language-model performance.

### ComplexityRouter

The router uses a deterministic, hand-built text heuristic (length, lexical diversity, and structural symbols) and a hard parameter budget. It is not a trained semantic classifier. Routed weights are inference tensors and the router retains them in memory, so keep the explicit cap appropriate for the target machine.

## Install

Python 3.10–3.12 is supported. Install the package and test extras from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

For CPU-only installations, install the CPU PyTorch wheel first, then the project:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

## Examples

```bash
# Router demonstration
python -m dna_dynimo.complexity_router

# Train a FourierDNA field on a synthetic target
python examples/train_dna.py

# Train the transformer copy-task demo using materialized weights
python examples/train_transformer.py

# Optional: train the shared FourierDNA generator on a tiny run (slower)
python examples/train_transformer.py --steps 10 --regenerate

# Run a small, safe checkpoint-generation smoke example (defaults: 2 x 128)
python examples/generate_1b_model.py

# Generate a very large dense artifact only after checking RAM and opting in
python examples/generate_1b_model.py --num-layers 15 --layer-size 8192 --confirm-large

# Load and benchmark the generated checkpoint
python examples/load_and_test.py saved_models/frozen_1b_model.pt

# Save visualizations under results/
python -m dna_dynimo.visualization
```

Large dense checkpoint generation is **not streaming**: the complete state dict is held in host RAM. The generator estimates required memory and refuses runs that appear unsafe, but the estimate cannot account for every process or system-level constraint. Use a smaller model or a purpose-built sharded checkpoint writer for constrained environments.

## Library example

```python
import torch
from dna_dynimo import FourierDNA

dna = FourierDNA(hidden=128, num_freqs=64)
with torch.inference_mode():
    weights = dna.generate_weights(1024, 1024)
print(weights.shape)
```

To train the shared generator through a transformer, construct it with `regenerate=True` and verify the cost on a small configuration first.

## Tests and security checks

GitHub Actions runs the test suite on Python 3.10, 3.11, and 3.12. The CodeQL workflow analyzes Python, but GitHub cannot accept its results while code scanning is disabled for this private repository. See [SECURITY.md](SECURITY.md). The examples load only local, trusted PyTorch checkpoints using restricted `weights_only=True` deserialization.

Contributor setup and pull-request guidance are in [CONTRIBUTING.md](CONTRIBUTING.md); project changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Known limitations

- The router’s rule-based score is only a rough heuristic and should not be used as a safety or quality gate.
- The transformer has not been evaluated on real language tasks.
- Dense generated matrices consume memory proportional to their full shape; chunking bounds intermediates, not final output storage.
- 1B-parameter generation requires multiple GiB of free RAM and is not suitable for ordinary laptops or CI.
- Checkpoint generation initializes the surrounding input/output projections randomly; it is an architecture experiment, not a trained pretrained model.

## License

MIT. See [LICENSE](LICENSE).
