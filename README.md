# DNA_DYNIMO

[![CI](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO/actions/workflows/ci.yml) [![CodeQL](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO/actions/workflows/codeql.yml) ![Python](https://img.shields.io/badge/python-3.10--3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c) ![License](https://img.shields.io/badge/license-MIT-green)

**DNA_DYNIMO is an experimental PyTorch research toolkit for exploring Fourier-feature neural fields as compact generators of dense weight matrices.** It also includes chunked matrix-generation utilities, a hard-budget demonstration router, and a small causal transformer. It is an architecture experiment—not a pretrained language model, a validated compression method, or a production inference server.

## What is included

### FourierDNA

`FourierDNA` maps normalized row/column coordinates through random Fourier features and a small multilayer perceptron to produce a dense matrix of shape `(out_dim, in_dim)`. Coordinate batches are chunked to bound intermediate activation memory, and gradients remain available when training the generator.

```python
import torch
from dna_dynimo import FourierDNA

dna = FourierDNA(hidden=128, num_freqs=64)
print(f"Trainable generator parameters: {dna.count_parameters():,}")

with torch.inference_mode():
    weight = dna.generate_weights(out_dim=256, in_dim=512)

print(weight.shape)  # torch.Size([256, 512])
```

Chunking limits temporary computation batches; it does **not** make the final dense matrix sparse or remove its output-memory cost. The default generator is roughly 50K trainable parameters, but that count by itself says nothing about accuracy, equivalent model capacity, or task performance.

### InfiniteParameter

`InfiniteParameter` adds domain and scale embeddings to a coordinate-based generator and exposes chunked matrix generation. It can evaluate a learned field at different resolutions. “Infinite” refers to the coordinate-field idea, not infinite compute, precision, memory, or output storage. Use `torch.inference_mode()` for inference-only generation.

### DNATransformer

`DNATransformer` is a compact causal transformer that uses matrices produced by the shared FourierDNA generator. It has two distinct modes:

- **Materialized mode (`regenerate=False`, the default):** generated attention and feed-forward matrices are stored as ordinary trainable dense parameters at initialization. Training updates these matrices and the transformer wrapper; it does not train the FourierDNA generator through those matrices.
- **Regeneration mode (`regenerate=True`):** matrices are generated during each forward pass, so gradients can flow into the shared FourierDNA generator. This is much slower and uses more computation and activation memory.

The included training example uses a synthetic copy task. It is a runnable functional demo, not a language-model benchmark or evidence of useful real-world language performance.

### ComplexityRouter and DNAEngine

`ComplexityRouter` estimates text complexity with deterministic, hand-written features based on input length, lexical diversity, and structural-symbol density. It is **not** a trained semantic classifier. The router allocates within the caller-provided parameter cap and asks `DNAEngine` to generate Q/K/V/O and feed-forward matrices. Routed weights are generated under inference mode and returned as dense tensors; memory use still scales with the generated weights.

The package also exports `KnowledgeClassifier`, a small neural module, but the router's current complexity score uses the deterministic heuristic above rather than this untrained module.

### Checkpoint generation

The checkpoint examples generate dense tensors initialized from FourierDNA and save model state dictionaries. They do not train a billion-parameter language model. Surrounding embedding/input/output components are initialized separately, and the artifacts are not pretrained checkpoints.

The generator estimates host RAM and free disk needs before writing. Generation is **not sharded or streaming**: the dense state is held in memory, and estimates cannot account for all system activity. Large runs require substantial free RAM and disk space. The default command is intentionally small; requests above 100M parameters require an explicit `--confirm-large` flag.

## Requirements

- Python 3.10, 3.11, or 3.12
- PyTorch 2.1 or newer
- NumPy, Matplotlib, and psutil (installed as package dependencies)
- A CPU is sufficient for the small tests and examples; CUDA is used when available by the training demos

## Install

Create and activate a virtual environment, then install the project and test tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For a CPU-only PyTorch install, install the CPU wheel first, then install the project:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

## Run the examples

Run commands from the repository root after installation:

```bash
# Budget-limited router demonstration
python -m dna_dynimo.complexity_router

# Fit a FourierDNA field to a synthetic smooth matrix (longer training demo)
python examples/train_dna.py

# Train the small transformer copy-task demo using materialized matrices
python examples/train_transformer.py

# Optional: train the shared generator through generated matrices (slower)
python examples/train_transformer.py --steps 10 --regenerate

# Generate and save a small dense checkpoint (safe demo defaults)
python examples/generate_1b_model.py

# Load and benchmark a checkpoint created by the generator
python examples/load_and_test.py saved_models/frozen_1b_model.pt

# Generate plots under results/
python -m dna_dynimo.visualization
```

Large dense checkpoint generation must be an intentional decision. Check available system resources first; the following example opts into a request above 100M parameters and can need many GiB of RAM:

```bash
python examples/generate_1b_model.py \
  --num-layers 15 \
  --layer-size 8192 \
  --confirm-large
```

The exact memory requirement depends on tensor shapes, PyTorch overhead, available disk, and other processes. Do not run this large example in ordinary CI or on a machine with limited free memory.

## Validate changes

Run tests and bytecode compilation locally:

```bash
python -m pytest -q
python -m compileall -q dna_dynimo examples tests
```

GitHub Actions runs the test suite on Python 3.10, 3.11, and 3.12. A separate CodeQL workflow scans the supported Python and GitHub Actions code paths and uploads findings to GitHub Code Scanning when the workflow completes. Check the linked workflow badges and [Actions runs](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO/actions) for the current result; a workflow badge represents CI/security checks, not a guarantee that the model is secure or production-validated.

Workflow dependencies are pinned to full commit SHAs. Dependabot checks Python and GitHub Actions dependencies weekly. See [SECURITY.md](SECURITY.md) for reporting guidance and [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and pull-request expectations.

## Repository layout

```text
DNA_DYNIMO/
├── dna_dynimo/                  # Installable Python package
│   ├── fourier_dna.py           # Fourier-feature matrix generator
│   ├── infinite_parameter.py    # Domain/scale-aware coordinate generator
│   ├── dna_transformer.py       # Compact causal transformer blocks
│   ├── complexity_router.py     # Heuristic routing and DNAEngine
│   ├── model_generator.py       # Dense checkpoint generation and estimates
│   ├── memory_safe_generator.py # Safety-oriented generation wrapper
│   └── visualization.py         # Research plots and demos
├── examples/                    # Training, inference, and generation demos
├── tests/                       # Bounded unit and smoke tests
├── .github/workflows/           # CI and CodeQL workflows
├── pyproject.toml               # Package metadata and test extra
└── requirements.txt             # Dependency list
```

## Limitations and interpretation

- This is experimental research software. It has not been validated as a production model-serving stack.
- The project does not provide a pretrained model or establish useful language-model quality.
- The router's heuristic score is not semantic understanding, a safety classifier, or a quality gate.
- Dense generated matrices consume memory proportional to their full dimensions. Chunking reduces intermediate activations, not final output storage.
- The transformer copy task is a smoke/demo workload and must not be treated as a language benchmark.
- Parameter-count comparisons do not establish equivalent capacity, compression quality, accuracy, or speed. Evaluate against a real task and a clearly specified baseline.
- Very large checkpoint generation is RAM- and disk-intensive and is unsuitable for ordinary laptops and CI runners.

## Security and responsible use

Do not commit credentials, local environment files, model checkpoints, or private datasets. The example checkpoint loaders use restricted `weights_only=True` deserialization and should still be used only with files you trust. Report suspected vulnerabilities privately rather than posting credentials or exploit details in a public issue; see [SECURITY.md](SECURITY.md).

## Contributing

Contributions that add tests, improve reproducibility, correct documentation, or make resource use safer are welcome. Please include the problem, implementation rationale, test evidence, and any compatibility or memory implications. Start with [CONTRIBUTING.md](CONTRIBUTING.md), and use the issue and pull-request templates.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for the full text.
