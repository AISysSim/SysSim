# AGENT.md

Instructions for AI coding agents working in this repository.

## Project Snapshot

SysSim is a Python/PyTorch operator-level performance simulator for LLM and
neural-network workloads. It traces PyTorch execution into an operator DAG,
estimates per-op runtime, and computes critical-path execution time.

Core concepts:
- Tracing uses `TorchDispatchMode` plus fake CUDA tensors in `syssim/tracer.py`.
- The DAG IR lives in `syssim/operator_graph.py`.
- Public entry points live in `syssim/api.py` and are exported from
  `syssim/__init__.py`.
- Hardware and network configuration live in `syssim/config.py`.
- Compute estimation lives under `syssim/compute/`.
- Collective/network simulation lives under `syssim/network/`.
- Hugging Face and PLENA integrations live under `syssim/integrations/`,
  `syssim/config_plena.py`, and `syssim/compute/plena_backend.py`.
- `PLENA_Simulator/` is a git submodule. Treat it as external unless the task
  explicitly asks for PLENA simulator changes.

Read `README.md` for user-facing behavior and `DESIGN.md` for architecture
before making non-trivial changes.


## Repository-Specific Guidance

### Units Matter

Be careful with performance units:
- Public hardware peaks use TFLOP/s.
- Public memory bandwidth uses GB/s.
- Internal roofline calculations often convert to FLOP/s, bytes/s, ns, or ms.
- Public operator times are reported in milliseconds.

Do not change units or conversions without targeted tests.

### CUDA and Fake Tensors

Tracing is CUDA-oriented. Many tracing tests are skipped without CUDA, and
`trace_model_for_inference`/`trace_model_for_training` should not be treated as
CPU-only execution paths. Avoid "fixes" that bypass CUDA requirements unless the
task explicitly asks for CPU tracing support.

### Compute vs Network Boundaries

Keep the two estimation paths distinct:
- GEMM, attention, math, and memory behavior belongs in `syssim/compute/` and
  tracing/operator classification code.
- Collectives, topology, LogGP parameters, device meshes, and congestion belong
  in `syssim/network/`.
- End-to-end distributed training support is not fully automatic; current docs
  describe manual use of the network simulator for communication estimates.

### Public API Changes

When changing user-facing behavior:
- Update `syssim/api.py` and `syssim/__init__.py` together when exports change.
- Update `README.md` examples or `DESIGN.md` if architecture or supported usage
  changes.
- Keep examples under `examples/` runnable from the repository root.

### PLENA Integration

PLENA support is split between SysSim glue code and the `PLENA_Simulator/`
submodule. Prefer changes in SysSim integration layers unless the user
explicitly asks to edit the submodule.

Relevant files:
- `syssim/config_plena.py`
- `syssim/compute/plena_backend.py`
- `docs/plena_syssim_requirements.md`
- `plena_integration_note.md`

## Common Commands

Install dependencies for this checkout:

```bash
python -m pip install -r requirements.txt
```

Run the full test suite:

```bash
python -m pytest tests -q
```

Run focused tests:

```bash
python -m pytest tests/test_network_collectives.py -q
python -m pytest tests/test_network_simulator.py -q
python -m pytest tests/test_tracing.py -q
```

Run NCCL tests only when CUDA and multiple GPUs are available:

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_nccl_backend.py
```

Run profiling tools from the repository root:

```bash
python -m syssim.compute.compute_cost_profiler --operator gemm --output data/profiling/example.csv
python -m syssim.compute.validate_profiler
```

Note: `README.md` mentions editable installs and optional extras, but this
checkout currently has `requirements.txt` at the root. Verify packaging metadata
exists before relying on `pip install -e .`.

## Testing Expectations

- For narrow fixes, run the smallest relevant test file or test node.
- For changes touching shared graph, tracing, config, or network contracts, run
  a broader subset and explain any skipped CUDA-dependent coverage.
- For unit conversion, performance model, or profiler changes, add assertions
  around dimensions, units, and expected monotonic behavior where possible.
- If tests cannot run because CUDA, multi-GPU, PLENA assets, or optional
  dependencies are unavailable, say exactly what was not verified.

## Documentation Expectations

Update docs when behavior changes:
- `README.md`: user-facing setup, examples, supported hardware, public API.
- `DESIGN.md`: architecture, core data flow, model boundaries.
- `docs/`: task notes and integration-specific details.
- `CHANGELOG.md`: externally visible changes intended for release notes.

Keep docs factual. Do not promise automatic distributed training estimates,
packaging extras, hardware support, or PLENA behavior unless the code in this
checkout actually supports it.


## Working Principles

### 1. Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

### 2. Simplicity First

Write the minimum code that solves the requested problem.

- No features beyond what was asked.
- No abstractions for single-use code.
- No speculative flexibility or configurability.
- No broad error handling for impossible scenarios.
- If a change starts getting large, look for the smaller path.

### 3. Surgical Changes

Touch only what the request requires.

- Do not improve adjacent code, comments, or formatting opportunistically.
- Do not refactor unrelated code.
- Match the existing local style even if you would choose differently.
- Remove imports, variables, or functions made unused by your own changes.
- Mention unrelated dead code or stale docs, but do not delete them unless asked.

Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Turn tasks into verifiable goals.

Examples:
- "Add validation" -> add or update tests for invalid inputs, then make them pass.
- "Fix a bug" -> reproduce it with a focused test, then make the test pass.
- "Refactor X" -> run relevant tests before and after when feasible.

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```