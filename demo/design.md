# SysSim ARIA Tutorial Colab — Design

**Status:** Draft, pending user review
**Author:** lexu (with Claude)
**Date:** 2026-05-27
**Target event:** ARIA Tutorial Meeting, 2026-05-22
**Branch:** `lexu/demo-notebook`

## 1. Goal

Produce a single Google Colab notebook that demonstrates SysSim's headline capabilities at the ARIA tutorial. The notebook must work in two use modes from the same artifact:

- **Live walkthrough** by Mike & Dayou during the meeting — cells run in minutes, prose is concise enough not to read aloud.
- **Self-paced handout** after the meeting — attendees open the notebook in their own Colab, follow it end-to-end, and understand without a presenter.

Five demo items, mapped to owners:

| # | Topic | Owner |
|---|---|---|
| 1 | Models — Qwen3-8B & Llama-3-8B (dense) | Mike |
| 2 | Configs — Batch / Seqlen / TP / PP | Mike |
| 3 | GPU vendor — AMD (roofline + profiling → predictor → run) | Mike |
| 4 | Precision — FP8 (profiling → predictor → run) | Dayou |
| 5 | Cost model — modifying `estimate_runtime()` (with PLENA as reference) | Dayou |

## 2. Non-Goals

- Real AMD or H100-FP8 profiling data — both are synthesized in-cell (workflow is real, numbers are illustrative).
- A parallel Markdown tutorial — notebook prose is the sole source of truth; a separate doc would drift within a week.
- Expert-parallel (EP) demo — marked work-in-progress in the README.
- Executing the PLENA cycle-level backend inside Colab — §5 references `syssim/external/plena/backend.py` as a reading example only.
- Promoting `demo/configs/*.yaml` into `examples/configs/` — by user direction, all demo artifacts stay under `demo/`.

## 3. File Layout

All new files under a single `demo/` directory at the SysSim repo root, on branch `lexu/demo-notebook`:

```
demo/
├── design.md                        # this document
├── aria_tutorial.ipynb              # the Colab notebook
├── smoke_test.py                    # offline regression check (see §7)
└── configs/
    ├── models/
    │   └── llama3-8b.yaml           # new asset: 32 layers, 4096 hidden, GQA 8/32
    └── hardware/
        └── mi300x.yaml              # new asset: AMD MI300X peaks + interconnect
```

Existing repo assets reused without modification:

- `examples/configs/models/qwen3-8b.yaml`
- `examples/configs/hardware/dgx_h100.yaml`
- `third_party/PLENA_Simulator` (submodule, reference reading only)
- The `syssim` package's public API: `simulate`, `estimate_memory`, `sweep`, `TrainingConfig`, `ParallelismConfig`, plus `compute.compute_cost_profiler.train_efficiency_model` and `compute.efficiency_models.set_efficiency_models_dir`.

Synthetic profiling CSVs and trained predictor artifacts produced during a notebook run live in `/tmp/`. Nothing synthetic is committed to the repo.

## 4. Colab Setup Cell (the fragile part)

### Runtime requirement

SysSim requires CUDA — `torch.cuda.is_available()` is asserted in `syssim/config.py:206`, `syssim/tracer.py:771`, and `syssim/training/runner.py:335`. The notebook header instructs the user to select **Runtime → Change runtime type → T4 GPU** before running any cell. Cell 1 hard-asserts CUDA availability and prints `torch.cuda.get_device_name(0)` so a missed runtime change fails loudly with a clear message.

### Install steps (single cell, ~3-5 min on a fresh runtime)

1. `git config --global url."https://github.com/".insteadOf "git@github.com:"` — neutralizes the SSH URL in `.gitmodules` so the PLENA submodule clones over HTTPS without keys.
2. `git clone -b lexu/demo-notebook --recurse-submodules https://github.com/AISysSim/SysSim.git` (drop `-b` flag once the branch is merged to `master`).
3. `%cd SysSim`
4. `pip install -q -e .` — pulls flashinfer-python, megatron-core, megatron-bridge, torch≥2.6, transformers, xgboost, scikit-learn.
5. `import syssim; print(syssim.__version__)` — install sanity check.

### Known fragility

- **`flashinfer-python` on T4 (sm_75)**: recent flashinfer wheels often target sm_80+. If T4 wheel install fails, the notebook prose instructs switching to Colab's L4 or A100 runtime (paid). Implementation must test this before claiming the notebook ships.
- **`megatron-bridge` install time**: can be slow. `-q` flag suppresses progress; prose tells the user to expect a multi-minute wait.
- **Colab PyTorch version**: current Colab images ship PyTorch ≥ 2.6, satisfying SysSim's pin. Verify during implementation; if Colab regresses, the install cell may need `pip install torch>=2.6 --upgrade` first.

### Open-in-Colab badge

Notebook cell 0 (markdown) contains:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AISysSim/SysSim/blob/lexu/demo-notebook/demo/aria_tutorial.ipynb)
```

Updated to `master` branch URL after merge.

## 5. Per-Section Content

Each section is independent — if one fails mid-meeting, the next still runs. No shared state across sections beyond the setup cell.

### §1 Models — Qwen3-8B vs. Llama-3-8B (Mike, ~5 min)

**Goal:** Show the simulator is architecture-aware.

Cells:
1. Markdown: explain that SysSim takes a model YAML + hardware YAML + parallelism/training kwargs.
2. Load `examples/configs/models/qwen3-8b.yaml` and `demo/configs/models/llama3-8b.yaml`, print their side-by-side architecture summary (layers, hidden, GQA groups, ffn ratio, RoPE).
3. Run `syssim.simulate(...)` on each with identical `HW=dgx_h100.yaml`, `ParallelismConfig(tp=2, dp=4)`, `TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16")`.
4. Print a side-by-side table: step time / forward / backward / MFU / peak memory.
5. One-line takeaway: differences are driven by arch alone — same HW, same parallelism.

**New asset needed:** `demo/configs/models/llama3-8b.yaml` — Llama-3-8B architecture (32 layers, 4096 hidden, 32 attn heads, 8 KV heads, 14336 ffn, RoPE, SwiGLU, no tied embeddings).

### §2 Configs — Batch / Seqlen / TP / PP (Mike, ~5 min)

**Goal:** Show parallelism and shape knobs are first-class.

Hold model = Qwen3-8B and HW = DGX H100 fixed. Run four `syssim.sweep(...)` calls, one per axis, each with a small matplotlib bar chart and a printed table:

- `micro_batch ∈ {1, 2, 4}` — memory-vs-throughput tradeoff.
- `seq_length ∈ {2048, 4096, 8192}` — attention cost growth.
- `parallelism.tp ∈ {1, 2, 4}` — MFU vs. exposed-collective tradeoff.
- `parallelism.pp ∈ {1, 2, 4}` — per-stage memory + bubble.

Chart x-axis = value of the swept knob; y-axis = step_time_ms (and peak_memory_gb on a second chart where relevant).

### §3 GPU vendor — AMD MI300X (Mike, ~8 min)

**Goal:** Show how to onboard a new vendor — roofline first, then trained-predictor refinement.

Three sub-cells building up:

**3a. Roofline-only.** Simulate Qwen3-8B on `demo/configs/hardware/mi300x.yaml` (peak FP16 ≈ 1307 TFLOPs, HBM 192 GB, peak HBM BW 5.3 TB/s, Infinity Fabric intra-node BW). Default `RooflineEstimator` runs with no efficiency model loaded — `efficiency_estimate` returns 1.0 (see `compute_cost_predictor.py:661`), so the report reflects pure roofline. Print the report.

**3b. Synthesize MI300X profiling data.** A cell generates three CSVs in `/tmp/mi300x_profiling/`:

- `gemm_mi300x_fp16_data.csv` with columns `M,N,K,t_measured_ms`
- `attn_mi300x_fp16_data.csv` with columns `bs,seq,nh,nkv,hd,t_measured_ms`
- `rmsnorm_mi300x_fp16_data.csv` with columns `seq,dim,t_measured_ms`

Values: `t_measured_ms = t_roofline_ms × (0.6 + 0.25·rng())`. Seeded RNG for reproducibility. Schema must exactly match what `compute_cost_profiler.train_efficiency_model` expects (verify against `data/profiling/gemm_pro6000_fp16_data.csv` etc.).

**3c. Train predictor → re-run.** Call `train_efficiency_model(...)` (XGBoost backend, training in seconds), point SysSim at the new model via `set_efficiency_models_dir(...)`, re-run `syssim.simulate(...)`. Print a 3-column table: Roofline ms / Trained ms / Δ%.

**New asset needed:** `demo/configs/hardware/mi300x.yaml`. Schema follows `examples/configs/hardware/dgx_h100.yaml`: `peak_tflops_mm`, `peak_tflops_math`, `peak_memory_bandwidth_GBps`, `gpus_per_node`, `gpu_memory_GB`, `inter_node_bandwidth_GBps`, `inter_node_latency_us`, plus `topology:` block.

### §4 Precision FP8 (Dayou, ~8 min)

**Goal:** Same workflow as §3, but the dimension is precision rather than vendor.

Three sub-cells symmetric to §3:

**4a. FP8 roofline.** `syssim.simulate(model=qwen3-8b, hardware=dgx_h100.yaml, training=TrainingConfig(dtype="fp8", ...), parallelism=ParallelismConfig(tp=2, dp=4))` — same parallelism as §1 so the bf16-vs-fp8 comparison is apples-to-apples. Default Roofline uses `peak_tflops_mm_fp8` (3958 in the H100 YAML). Print report; compare against the §1 bf16 number to show the expected ~2× throughput from FP8.

**4b. Synthesize H100-FP8 profiling data.** `gemm_h100_fp8_data.csv` in `/tmp/h100_fp8_profiling/`. Same schema as §3b. Values target `t_measured_ms = t_roofline_fp8_ms × (0.65 + 0.20·rng())` — slightly tighter band than §3 to suggest a more mature FP8 implementation.

**4c. Train predictor → re-run.** Same `train_efficiency_model` → `set_efficiency_models_dir` → simulate flow. Print Roofline-vs-trained Δ% table. Note: SysSim auto-detects FP8 output dtype to route to the FP8 efficiency model (`compute_cost_predictor.py:644-655`); the synthesized CSV must be trained with `dtype="fp8"` so the model registers under the right key. `SYSSIM_FORCE_DTYPE=fp8` env var is the override if auto-detection misses.

### §5 Cost model — modifying `estimate_runtime()` (Dayou, ~10 min)

**Goal:** Show how to plug a custom estimator into SysSim, using a toy example, with PLENA as the realistic reference.

Cells:
1. Markdown: explain SysSim's pluggable `Estimator` Protocol (`syssim/compute/estimator.py:17` — `estimate_op(...)` returns ms per operator). Default is `RooflineEstimator`; PLENA is the canonical custom backend.
2. Walk through `syssim/external/plena/backend.py:282` (`PLENAEstimator` class) at a high level: how it maps PyTorch ops → PLENA cycles → ms.
3. Define a `ConstantEstimator` in the notebook — returns `1.0` ms for every operator regardless of inputs. ~10 lines.
4. Plug it in via `HardwareInfo.build_estimator` (or whichever public hook implementation determines is cleanest — see §8 risk).
5. Run `syssim.simulate(...)` with the constant estimator, show step time is now `~num_ops × 1ms`, confirming the plumbing works.
6. Close with: "for a real implementation, see `syssim/external/plena/backend.py`."

User confirmed §5 stays toy (ConstantEstimator) — not a more involved demo estimator.

## 6. Notebook Style Decisions

- **Length budget:** ~30-40 cells total; ~35-40 min live runtime including narration.
- **Charts:** matplotlib bar charts in §2 (user-approved); text tables elsewhere.
- **Independence:** sections decoupled — each can be re-run from any point after setup without re-running prior sections.
- **Print verbosity:** `simulate(...).report.summary()` style output; not raw dicts.
- **No hidden state:** every cell that mutates global state (e.g., `set_efficiency_models_dir`) has a paired "reset" at section end.

## 7. Verification

**Before sharing the notebook:**

1. Run end-to-end in a fresh Colab T4 runtime. If T4 fails on flashinfer install, retest on L4 and document the runtime requirement.
2. Run `demo/smoke_test.py` locally — a small Python script that imports the same code paths the notebook uses (load YAMLs, simulate, train a tiny predictor, swap in ConstantEstimator) and asserts each returns non-empty. Target runtime <60 seconds. Used for regression checks after SysSim core changes without re-opening the notebook.

**In-notebook safety nets:**

- Cell 1 asserts `torch.cuda.is_available()` with a clear "Switch to GPU runtime" message on failure.
- Install cell uses `-q`; on failure, prose tells the user to re-run with `-v` for diagnostics.
- Each section independent — failure in §3 does not block §4.

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | `flashinfer-python` has no T4 wheel | flashinfer is a hard dep in `pyproject.toml` so removal isn't an option without a SysSim change. Realistic mitigations: (a) document L4/A100 fallback in prose; (b) pin to last sm_75-compatible flashinfer version in the install cell; (c) if neither works, escalate to a SysSim issue to soft-require flashinfer. Verify in implementation. |
| R2 | `§5` estimator-swap hook may not exist as a clean public API | The `Estimator` protocol exists; if no public swap mechanism, add a 1-line factory parameter to `HardwareConfig.build_estimator` as part of this work. |
| R3 | Synthesized data looks unrealistic to Mike/Dayou during dry-run | Formula is a constant + multiplier with seeded RNG; easy to tune. Spec leaves the scaling adjustable. |
| R4 | Llama-3-8B YAML schema mismatch with SysSim's `ModelConfig` expectations | Schema mirrored from existing `examples/configs/models/qwen3-8b.yaml`. Implementation validates by loading via `syssim.simulate(...)` in `smoke_test.py`. |
| R5 | MI300X hardware YAML field names don't map cleanly (SysSim is NVIDIA-centric) | The hardware YAML uses neutral `intra_node_bandwidth_GBps` / `inter_node_bandwidth_GBps` per README §Configuration. MI300X numbers slot in directly. |
| R6 | PLENA submodule clones but `_load_perf_model` fails (TOML config missing in public repo) | §5 only **references** the PLENA backend as reading material; it does not execute it. So this risk is contained even if it materializes. |

## 9. Ownership

- **lexu (this branch):** Produces the full first draft — notebook, two YAMLs, smoke test. Verifies it runs end-to-end in Colab.
- **Mike:** Polishes §1, §2, §3 — prose, numbers, any architecture nuances I got wrong.
- **Dayou:** Polishes §4, §5 — FP8 narrative, PLENA reference correctness.
- **Merge target:** `master` after Mike & Dayou sign off. Open-in-Colab badge URL gets updated to `master` in the merge commit.

## 10. Out of Scope

Explicit list, for clarity:

- Real AMD profiling data (use synthetic).
- Real H100 FP8 profiling data (use synthetic — although `data/profiling/gemm_pro6000_fp8_data.csv` exists for Blackwell, the design symmetry with §3 justifies synthesizing for H100).
- Promoting `demo/configs/*.yaml` into `examples/configs/` (per user direction).
- Markdown companion doc.
- Expert-parallel (EP) demo.
- Executing the PLENA cycle-level backend in Colab.
- Multi-notebook split / sub-tutorials.

## 11. Open Questions for Implementation

These are deferred to the implementation plan, not blocking the spec:

1. Exact public mechanism to swap `Estimator` — read `HardwareConfig.build_estimator` and adjacent code, decide whether to use an existing hook or add a small one.
2. Final MI300X peak numbers — confirm against AMD spec sheet at implementation time (FP16 peak, FP8 peak, HBM3 bandwidth, Infinity Fabric BW). Current draft uses public spec values.
3. Llama-3-8B exact arch numbers — confirm against Meta's published config at implementation time.
4. Whether `set_efficiency_models_dir(...)` is the cleanest hook for predictor swap, or whether a per-`simulate(...)` parameter exists.
