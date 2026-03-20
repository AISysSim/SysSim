---
name: "Roadmap: SysSim v0.1 — Foundation Release"
about: Tracks all work needed for a solid v0.1 open-source release
title: "Roadmap: SysSim v0.1 — Foundation Release"
labels: roadmap, v0.1
---

# SysSim v0.1 Roadmap — Foundation Release

This issue tracks the work needed to ship a solid v0.1 open-source release of SysSim. Each section below contains tasks that can be picked up independently. **Contributors: please link your PR to this issue when submitting work for any of these items.**

---

## Project Infrastructure

- [ ] Add a `LICENSE` file (propose: Apache 2.0 or MIT)
- [ ] Add `CONTRIBUTING.md` with development setup, coding standards, PR process
- [ ] Add `CODE_OF_CONDUCT.md`
- [ ] Add `pyproject.toml` with proper packaging metadata (replace raw `requirements.txt` for install)
- [ ] Add `CHANGELOG.md` (keep-a-changelog format)
- [ ] Add GitHub Issue templates (bug report, feature request)
- [ ] Add GitHub PR template

## CI / CD

- [ ] GitHub Actions: lint (ruff) on push & PR
- [ ] GitHub Actions: run test suite (`pytest`) on push & PR
- [ ] GitHub Actions: type checking (mypy or pyright) — at least on public API surface
- [ ] Decide on supported Python versions (3.10+?) and test matrix
- [ ] (Stretch) Publish to PyPI on tagged release

## Documentation

- [ ] Expand README: add badges (CI, license, PyPI), installation via pip, link to contributing guide
- [ ] Add inline API docstrings where missing (focus on `syssim/api.py`, `config.py`, `operator_graph.py`)
- [ ] (Stretch) Sphinx / MkDocs site with API reference + tutorials
- [ ] Document hardware requirements and limitations clearly (CUDA dependency)

## Testing & Quality

- [ ] Ensure full test suite passes in CI without a GPU (mock/skip GPU-dependent tests gracefully)
- [ ] Add test coverage reporting (e.g., `pytest-cov`)
- [ ] Review and tag tests that require CUDA so contributors without GPUs can still run the rest
- [ ] Add pre-commit hooks config (`.pre-commit-config.yaml`) for linting/formatting

## Core Features (v0.1 scope)

- [ ] Validate accuracy of roofline + efficiency model on at least 2 hardware targets (e.g., A100, GH200)
- [ ] Ensure `trace_model_for_inference` and `trace_model_for_training` work end-to-end on common architectures (transformer, CNN, MLP)
- [ ] Network simulator: validate collective communication estimates against NCCL benchmarks
- [ ] Stabilize public API surface — mark internal modules with `_` prefix or `__all__`

## Integrations

- [ ] HuggingFace integration: test with 3+ popular model families (Llama, Qwen, Mistral)
- [ ] Megatron integration: validate tensor parallel tracing
- [ ] (Stretch) Add vLLM or TensorRT-LLM integration example

## Packaging & Distribution

- [ ] Create `pyproject.toml` with `[project]` metadata, entry points, optional deps groups
- [ ] Separate core vs optional dependencies (e.g., `pip install syssim[huggingface]`, `pip install syssim[profiler]`)
- [ ] Ensure `data/` directory handling works after pip install (package data or download scripts)
- [ ] Add version string (`syssim.__version__`)

---

## How to Contribute

1. Comment on this issue to claim a task
2. Create a branch, do the work, open a PR
3. Reference this issue in your PR description (e.g., `Part of #<issue_number>`)
4. One task per PR preferred — keeps reviews fast

Questions? Open a discussion or comment below.
