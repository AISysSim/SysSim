# AGENTS.md

Guidance for coding agents working in this repository. Applies to the whole repo.

## ⚠️ Agent artifacts — keep them OUT of the repo

**Always write agent artifacts (brainstorming notes, scratch analysis, task
write-ups) under `agent_space/` — which is gitignored. NEVER put them in `docs/`,
the repo root, or anywhere else tracked by git.** Superpowers skill specs/plans go
under `agent_space/docs/superpowers/` (`specs/`, `plans/`). The only human-facing
docs in the tree are `README.md` (what / how-to-use) and `docs/DESIGN.md`
(architecture) — `docs/` holds curated docs, not agent scratch output.

## Project Overview

SysSim is a Python/PyTorch **operator-level performance and memory simulator** for
distributed LLM training. It traces a model with fake CUDA tensors to build an
operator DAG, estimates per-operator time through a **pluggable estimator** (GPU
roofline + learned efficiency by default), times collectives with a **flow-level
network simulator**, and replays the graph through a stream-queue discrete-event
simulator to get step time, MFU, and per-GPU peak memory.

Primary package: `syssim/`. Requires a CUDA-capable GPU (fake tensors still need
CUDA so PyTorch dispatches GPU kernel variants) and Megatron-Core.

## Public API

High-level (config-driven training simulator, `syssim.training`):

- `syssim.simulate(model, hardware, parallelism, training)` → `SimulationReport`
  (step time, fwd/bwd/optimizer, collective exposed, MFU/HFU, peak memory,
  per-PP-stage memory, OOM, bottlenecks).
- `syssim.estimate_memory(...)` → per-GPU peak memory via MemTracker (same memory
  path as `simulate`, but skips the runtime DES); returns a memory-only `SimulationReport`.
- `syssim.sweep(..., over={...})` → search a config axis; `.best(metric)`.
- Configs: `ModelConfig`, `ParallelismConfig`, `TrainingConfig`, `HardwareConfig`;
  model sources `HFModel` / `CustomModel`.
- CLI: `syssim run | memory | summary | sweep <model.yaml> --hardware <hw.yaml> ...`.

Low-level: `syssim.OperatorGraphTracer`, `HardwareInfo`, `OperatorGraph`.

Custom estimators (e.g. PLENA accelerator): `from syssim.external.plena import
PLENAEstimator, PLENAConfig` — attached via `HardwareConfig(estimator=...)`.

## Configuration: two YAML types

- **Model YAML** — architecture only (layers, hidden, heads, ffn, vocab, …).
- **Hardware YAML** — accelerator peaks + `gpus_per_node` + `gpu_memory_GB` + a
  `topology:` block (intra/inter-node bandwidth; `type`: simple | arbitrary |
  two_layer_multipath | fat_tree).
- Parallelism and training knobs are Python kwargs / CLI flags, **not** YAML.
  Bandwidth fields use neutral `intra_node_*` / `inter_node_*` names.

## Repository Map

- `syssim/` — core package (tracer, operator-graph IR, config, CLI) + subpackages:
  - `compute/` — per-operator cost: pluggable estimators (roofline default) + FLOP counting.
  - `network/` — flow-level network sim: topologies, collectives, max-min fair solver, ECMP load balancer.
  - `training/` — distributed training simulator: spec/configs, parallelism (incl. PP composition), memory (MemTracker), runtime DES, report.
  - `external/` — optional isolated integrations (e.g. `plena/` custom estimator). Core never imports `external/`.
- `examples/` — runnable examples + `configs/{models,hardware}/*.yaml`.
- `tests/` — pytest suite.
- `third_party/` — git submodules (e.g. `PLENA_Simulator`).
- `agent_space/` — gitignored agent workspace (see top of this file).
- `docs/DESIGN.md` — architecture reference. `pyproject.toml` — package metadata.

## Architecture Rules To Preserve

Tracing:
- The tracer requires CUDA even with fake tensors (PyTorch needs fake CUDA tensors
  to dispatch GPU kernel variants). It mutates params/buffers to fake tensors and
  must restore the model afterward. View ops aren't nodes (propagate storage
  aliases); creation ops are zero-time nodes. Real collectives must not run on fake
  tensors — they are patched to no-ops during tracing.

Estimation (pluggable, transparent):
- The **tracer and the simulator are transparent to the estimator**. The tracer's
  only estimation touchpoint is `compute_cost_predictor.estimate_runtime(...,
  hw_info, ...)`, which delegates to `hw_info.build_estimator().estimate_op(...)`.
  Do NOT make the tracer or DES reference an estimator type.
- Estimator selection lives on `hw_info` (`HardwareInfo.estimator`, default →
  `RooflineEstimator`). No `estimator=` kwarg on `simulate`/`trace`/the tracer.
- New backends implement the `Estimator` protocol (`syssim/compute/estimator.py`)
  and live under `syssim/external/`; core must not import them.
- Keep units explicit: peaks in TFLOP/s, bandwidth in GB/s, roofline internals ns,
  public op estimates ms.

Network:
- Collectives are decomposed into point-to-point `Op`s; topology + timing live in
  `simulate(ops, topology)`. The simulator re-solves **max-min fair** rates over
  the active flow set per event; per-rank contention is implicit via shared links.

Training simulator:
- Two YAML types only (model + hardware); parallelism/training are kwargs.
- PP runs each stage as a separate process (MPMD) composed with timed P2P; the
  schedule is **1F1B** only. Per-stage peak memory = single-microbatch footprint
  (MemTracker) scaled by the 1F1B in-flight count; earlier stages hold more.
- `HardwareConfig.topology` is required for the training path.

## Setup

```bash
pip install -e .
```

CUDA-enabled PyTorch (≥2.6) and Megatron-Core are required for tracing/simulation.

## Test Commands

```bash
python -m pytest tests/ -q                 # full suite
python -m pytest tests/compute -q          # estimator unit tests (no GPU/megatron)
```

Integration tests (`tests/training`, `tests/external/plena`) need Megatron-Core +
CUDA and skip cleanly when unavailable (PLENA tests need the `third_party/
PLENA_Simulator` submodule: `git submodule update --init`).

## Style And Maintenance

- Match the existing dataclass-heavy, explicit-type Python style; small focused
  changes with focused tests.
- Never reference external projects by name in source (identifiers, docstrings,
  comments, commit messages) — citations belong only in `agent_space/` design docs.
- Keep generated profiling outputs, trained models, logs, and caches out of commits
  unless explicitly requested.
