# SysSim Design

SysSim estimates the **step time** and **peak memory** of distributed LLM training
on hardware you may not have, without running real computation. It traces a model
once to build an operator graph, estimates each operator analytically, simulates
collectives over a network model, and replays the graph through a discrete-event
simulator to produce a step-time and memory report.

The guiding principle: **trace structure once, estimate cost analytically.** No
real kernels run; fake tensors carry only shapes/dtypes/devices, so a single CPU
host with one GPU visible can model multi-GPU, multi-node training.

---

## 1. Pipeline

```
model + configs
      │
      ▼
  ┌─────────┐   operator    ┌──────────────┐   per-op time   ┌───────────────┐
  │ tracer  │──── graph ───▶│   estimator  │──── + memory ──▶│   training     │
  │ (fake   │   (DAG, per   │  (per-op     │                 │   simulator    │
  │  CUDA)  │    stage)     │   cost)      │                 │  (compose +    │
  └─────────┘               └──────────────┘                 │   DES + report)│
                                                              └───────────────┘
```

1. **Trace** — intercept PyTorch dispatch on fake CUDA tensors → an `OperatorGraph`
   (nodes = operators with shapes; edges = data/stream dependencies).
2. **Estimate** — fill each node's `estimated_time_ms` via a pluggable estimator
   (the roofline bound by default; an optional per-device calibrated model for accuracy).
3. **Compose & simulate** — for distributed runs, compose per-stage graphs, inject
   collectives/optimizer, time them over the network model, and replay everything
   through a stream-aware discrete-event simulator to get the critical-path step
   time. Memory is estimated per stage and scaled by the pipeline schedule.

Layers map to packages: `syssim/{tracer.py, operator_graph.py}` (trace + IR),
`syssim/compute/` (per-op cost), `syssim/network/` (collective timing),
`syssim/training/` (distributed orchestration + memory + report),
`syssim/external/` (optional custom estimators).

---

## 2. Operator graph IR (`operator_graph.py`)

The common intermediate representation. An `OperatorGraph` is a DAG of
`OperatorNode`s; every estimator and the simulator operate on it.

- **Operator types:** `GEMM`, `ATTN`, `MATH` (compute), `COLLECTIVE` (communication),
  `MEMORY`, `BARRIER`, `STREAM_SYNC` (synchronization).
- **Node:** name, op type, shape/dtype metadata, `config` (op-specific fields, e.g.
  collective kind/bytes/ranks, training `phase`), `stream_id`, `predecessors`, and
  `estimated_time_ms` (filled by the estimator / network timing).
- **Dependencies:** intra-stream FIFO order plus explicit cross-stream edges from
  synchronization primitives. This lets the simulator overlap independent streams
  and find the true critical path rather than summing op times.

---

## 3. Tracing (`tracer.py`)

`OperatorGraphTracer` runs the model under `TorchDispatchMode` + `FakeTensorMode`
and records every dispatched operator as a graph node.

- **Fake CUDA tensors.** A CUDA device must be visible even though no kernels run:
  PyTorch needs fake CUDA tensors so dispatch selects the GPU kernel variants
  (e.g. flash attention). Tensors carry shape/stride/dtype/device only.
- **Two passes per stage.** A `num_microbatches=1` **memory pass** (under MemTracker
  + a fake optimizer step) captures a clean per-microbatch memory footprint, and a
  full **runtime pass** builds the operator graph. The memory pass runs first on the
  freshly built model so accumulated gradients don't inflate the footprint.
- **Model mutation + restore.** Parameters/buffers are swapped to fake tensors for
  the trace and restored afterward.
- **Op handling.** View ops propagate storage aliases (not nodes); creation ops are
  zero-time nodes; real distributed collectives are patched to no-ops (they cannot
  run on fake tensors) and recorded as `COLLECTIVE` nodes for later timing.
- **Phase tagging.** Each captured op is tagged forward/backward from the module
  tracker's backward flag, so the report can split forward vs. backward time.

The tracer is **transparent to cost**: it calls `estimate_runtime(...)` (Section 4)
and stores the result, knowing nothing about which estimator is used.

---

## 4. Per-operator estimation (`compute/`)

Per-operator runtime is **pluggable** behind a small protocol, so the cost model is
swappable without touching the tracer or the simulator.

```python
class Estimator(Protocol):
    def estimate_op(self, func_packet, args, kwargs, out, op_type,
                    execution_mode=None, cache_seq_len=0) -> float: ...   # ms
```

- **Selection lives on `hw_info`.** `HardwareInfo` carries an optional `estimator`;
  `hw_info.build_estimator()` returns it, or the default `RooflineEstimator`
  (`compute/roofline_estimator.py`) when unset. There is no `estimator=` argument on
  the public API — the hardware object is the single source.
- **Transparent boundary.** The tracer's only cost touchpoint is the module function
  `estimate_runtime(...)` in `compute/estimator.py` (alongside the `Estimator`
  protocol), a thin delegate: `return hw_info.build_estimator().estimate_op(…)`.
  Neither the tracer nor the discrete-event simulator references an estimator type.

### Default: the roofline bound (`RooflineEstimator`)

```
roofline_ns = max(tensor, fma, sfu, mem, launch)   # the single binding demand (ns)
```

The default is the **roofline**: the binding demand among Tensor (MMA), FMA (FP32 vector),
SFU (transcendentals) and memory, plus an optional kernel-launch floor — collapsed to one
`roofline_ns`. Each demand is blackbox — shapes + the op's math definition + device
constants: `tensor = MMA_FLOPs / tensor_peak`, `mem = bytes / peak_bandwidth`, etc., with
size-aware and per-dtype peaks (FP16/FP8/FP4) and FLOPs from `flop_counter.py`. The
`fma`/`sfu` terms come from the op's instruction mix (0 for GEMM — pure MMA — and nonzero
for softmax/gelu/norm), generalising the single-ceiling roofline (which has no SFU term and
is systematically too fast for those). Lives in `compute/predictor/roofline.py`; the
`RooflineEstimator` wrapper (`compute/roofline_estimator.py`) just returns `roofline_ns` in ms.

Units are explicit: peaks in TFLOP/s, bandwidth in GB/s, internals in nanoseconds, public
op estimates in milliseconds.

### Calibrated predictor (`TreeEstimator`, opt-in)

A per-device calibrated estimator: it multiplies the roofline `roofline_ns` by
`exp(residual)`, where the residual is one regularized LightGBM tree per operator family
(GEMM, attention, normalization, elementwise, reduction). The result is clamped to
`roofline_ns` as an out-of-distribution rail — a learned correction never predicts below
the physical floor, and this is the *same* rail calibration trains against. Any family
without a tree (or any prediction error) falls back to the bare roofline — it never
raises. Trained once per device and loaded from the committed per-device model:

```python
from syssim.compute.tree_estimator import TreeEstimator
hw = HardwareConfig(..., estimator=TreeEstimator.load("data/gh200", hw_info))
```

The trees (`TreeModel`, `compute/predictor/tree_model.py`) and their raw profiling data
live under `data/<device>/` (committed), produced by `syssim profile` (device-side
measurement) + `syssim calibrate` (CPU fit); see §10.

### Custom estimators (`syssim/external/`)

A custom backend (e.g. the PLENA accelerator's cycle model) implements `estimate_op`
and lives **outside the core library** under `syssim/external/`. The dependency is
one-way: `external` imports the core `Estimator` protocol; core never imports
`external`. A user opts in by attaching it to the hardware object:

```python
from syssim.external.plena import PLENAEstimator, PLENAConfig
hw = HardwareConfig(..., estimator=PLENAEstimator(PLENAConfig.from_plena_submodule()))
```

The estimator then rides `hw_info` through to the tracer; the rest of the pipeline is
unchanged.

---

## 5. Network simulation (`network/`)

Collective communication time comes from an **event-driven, flow-level simulator**
over a multipath topology — not a closed-form latency formula.

- **Topologies (`topology.py`):** a `Topology` is a graph of GPUs, switches, and links
  with a central adjacency. Builders: `simple` (per-node shared NIC, cloud-style),
  `arbitrary` (per-GPU NICs into a rack/root tree), `two_layer_multipath`
  (per-GPU uplinks + leaf↔spine mesh + intra-node mesh), and `fat_tree`. GPUs carry a
  physical node id and a routing leaf index so multi-node-per-rack and cross-pod paths
  resolve correctly. `build_topology_from_config(hardware)` dispatches on the YAML
  `topology.type`.
- **Routing (`load_balancer.py`):** multipath routes are chosen by an ECMP-style hash
  (deterministic), extensible via a registry.
- **Collectives (`collectives.py`):** ring all-reduce / all-gather / reduce-scatter and
  binary-tree broadcast, each decomposed into point-to-point `Op`s with inline
  data-causality dependencies.
- **Solver (`simulator.py`):** `simulate(ops, topology)` runs an event loop, resolving
  each flow's path and re-solving **max-min fair** rates across the active flow set on
  every event (flow start/activate-after-latency/complete). Per-rank contention emerges
  from shared-link sharing rather than explicit serialization, and the result is a
  makespan + per-op finish times.

---

## 6. Distributed training simulator (`training/`)

The high-level, config-driven layer that ties everything together.

### Configuration — two YAML types

- **Model YAML** — architecture only (layers, hidden size, heads, FFN, vocab, …).
- **Hardware YAML** — accelerator peaks, `gpus_per_node`, `gpu_memory_GB`, and a
  `topology:` block (intra/inter-node bandwidth + type). The number of nodes is derived
  from `world_size / gpus_per_node`.

Parallelism and training knobs (`ParallelismConfig`, `TrainingConfig`) are Python
kwargs / CLI flags, never YAML. Models may also be sourced from a `HardwareConfig`-
attached `HFModel` (resolved via Megatron-Bridge) instead of a YAML.

### Parallelism

Tensor (TP), sequence (SP), data (DP), context (CP), and pipeline (PP) parallelism,
combinable. Expert parallelism (EP) is in progress.

- **SPMD (no PP)** traces once in-process with a fake process group sized to the world.
- **PP is MPMD:** each pipeline stage is traced in its own OS process (`mp.spawn`),
  serialized to JSON, and recomposed by `compose_multi_rank_graph`, which namespaces
  per-stage streams and wires cross-stage P2P edges timed by `p2p_time_ms` over the
  topology. The schedule is **1F1B** (Megatron's default for `pp > 1`).
- TP/SP collectives captured during tracing, the injected DP gradient all-reduce, and
  PP send/recv are all timed through the network simulator (Section 5).

### Runtime estimation

`simulate(...)` injects the DP all-reduce + optimizer step into the composed graph,
times the captured collectives, then runs a **stream-queue discrete-event simulator**
(`runtime.py`): an overlap-aware, dependency-driven replay where pipeline bubbles
emerge naturally from cross-stage P2P edges. It yields the critical-path step time,
per-phase (forward/backward/optimizer) time, and exposed vs. total collective time;
MFU/HFU follow from a standard transformer FLOP budget.

---

## 7. Memory model

Memory is estimated from the trace, not a closed-form formula, and is correct by
construction for pipeline parallelism.

- **Per-stage footprint.** The `num_microbatches=1` memory pass yields a per-stage
  `MemoryProfile`: persistent categories (parameters, buffers, gradients, optimizer
  state) resident once, plus per-microbatch activation and per-backward temp.
- **Schedule scaling.** A stage's peak is `persistent + in_flight · activation + temp`,
  where `in_flight = min(pp − stage_rank, num_microbatches)` under 1F1B — so earlier
  stages, which hold more in-flight microbatches, correctly show higher peaks. For
  `pp = 1`, `in_flight = 1`.
- **Binding stage & OOM.** The report's `peak_memory_gb` is the heaviest stage;
  `pp_stage_memory_gb` lists every stage. When `gpu_memory_GB` is set, exceeding it
  flags OOM (with the offending stage/module) while still returning a finite step time.

`estimate_memory(...)` reuses this exact memory path but skips the runtime DES, for a
memory-only report.

---

## 8. Report & bottlenecks (`report.py`)

`simulate(...)` returns a `SimulationReport`: step time and per-phase breakdown,
collective total/exposed time, MFU/HFU, `peak_memory_gb`, `pp_stage_memory_gb`,
per-category memory bytes, and a `bottlenecks` summary (top ops by time, dominant op
type, longest collective, heaviest pipeline stage, peak module, and OOM capacity /
required / excess). `to_json` / `to_dataframe` support sweeps and tabulation;
`sweep(..., over={...})` runs a config grid and selects the best by a metric.

---

## 9. Extensibility

- **New cost backend** → implement the `Estimator` protocol under `syssim/external/`
  and attach it to `HardwareConfig.estimator`; nothing in the trace or simulator path
  changes. (A declarative `estimator:` YAML block + name registry is a planned
  follow-up; selection is already centralized on `hw_info`.)
- **New topology** → add a builder in `network/topology.py` (implement path resolution)
  and a `topology.type` case in `build_topology_from_config`.
- **New hardware** → supply peaks (incl. per-dtype where supported). The roofline
  default needs no per-device data; for higher accuracy, train the per-family residual
  model with `profile` + `calibrate` (§10) — the roofline itself needs no retraining.

---

## 10. Public API & CLI

```python
import syssim
report = syssim.simulate(model=..., hardware=..., parallelism=..., training=...)  # SimulationReport
mem    = syssim.estimate_memory(model=..., hardware=..., parallelism=..., training=...)  # memory-only report
result = syssim.sweep(..., over={"parallelism.tp": [1, 2, 4]})                    # grid search
```

Configs: `ModelConfig`, `ParallelismConfig`, `TrainingConfig`, `HardwareConfig`;
model sources `HFModel` / `CustomModel`. CLI: `syssim run | memory | summary | sweep
<model.yaml> --hardware <hw.yaml> …`, plus the predictor pipeline `syssim profile`
(device-side real-kernel measurement on the target GPU; `--num-workers N` shards the
sweep across GPUs) and `syssim calibrate` (CPU fit), writing the per-device model under
`data/<device>/`. The low-level `OperatorGraphTracer` / `OperatorGraph` / `HardwareInfo`
remain available for direct graph work.
