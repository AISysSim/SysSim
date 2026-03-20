# Contributing to SysSim

Thank you for your interest in contributing to SysSim! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.6+ with CUDA support (for tracing features)
- Git

### Install

```bash
git clone https://github.com/AISysSim/SysSim.git
cd SysSim
pip install -e ".[dev]"
```

If you don't have a CUDA GPU, you can still work on non-tracing components (network simulator, packaging, docs, etc.). GPU-dependent tests are automatically skipped when CUDA is unavailable.

### Running Tests

```bash
# Run all tests (GPU-dependent tests skipped if no CUDA)
pytest

# Run a specific test file
pytest tests/test_network_simulator.py

# Run with coverage
pytest --cov=syssim
```

## How to Contribute

### Finding Work

- Check the [v0.1 Roadmap issue](https://github.com/AISysSim/SysSim/issues) for available tasks
- Look for issues labeled `good first issue` or `help wanted`
- Comment on an issue to claim it before starting work

### Workflow

1. **Fork** the repo and create a branch from `master`
2. **Make your changes** — keep PRs focused (one feature/fix per PR)
3. **Add tests** for new functionality
4. **Run the test suite** to make sure nothing is broken
5. **Open a PR** and reference the related issue (e.g., `Part of #1`)

### Code Style

- Follow existing code conventions in the file you're editing
- Use type hints for public API functions
- Add docstrings for public classes and functions (Google style)
- Keep imports organized: stdlib, third-party, local

### Commit Messages

- Use imperative mood: "Add feature" not "Added feature"
- Keep the first line under 72 characters
- Reference issues when relevant: "Fix FLOP counting for conv2d (#42)"

## What Makes a Good PR

- **Focused**: one logical change per PR
- **Tested**: new code has tests, existing tests still pass
- **Documented**: public API changes include docstring updates
- **Clean**: no unrelated formatting changes or refactors mixed in

## Reporting Bugs

Use the [bug report template](https://github.com/AISysSim/SysSim/issues/new?template=bug_report.md). Include:
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, PyTorch version, GPU model)

## Questions?

Open a [GitHub Discussion](https://github.com/AISysSim/SysSim/discussions) or comment on the relevant issue.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
