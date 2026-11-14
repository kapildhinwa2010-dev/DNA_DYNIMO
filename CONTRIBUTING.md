# Contributing

Thanks for considering a contribution to DNA_DYNIMO. The project is a research toolkit; please describe whether a change affects experimental behavior, model quality claims, or resource usage.

## Development setup

Use Python 3.10–3.12. From the repository root, create an isolated environment and install the package with its test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

For a CPU-only PyTorch installation, install the CPU wheel before the project:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

## Before opening a pull request

Run the test suite and compile checks:

```bash
python -m pytest -q
python -m compileall -q dna_dynimo examples tests
```

For inference-only generation, use `torch.inference_mode()` or `torch.no_grad()`. Keep CI tests small and deterministic; do not run large dense model generation in CI. Add tests for changed public behavior and document memory requirements when changing generation sizes.

Pull requests should explain the problem and the chosen fix, list relevant tests, and note any compatibility or resource-use changes. Do not commit generated model checkpoints, local credentials, or experiment outputs.
