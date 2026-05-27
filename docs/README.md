# SysSim Docs

Supplementary documentation that doesn't belong in the top-level README or [DESIGN.md](../DESIGN.md). The repo-root docs cover *what SysSim does* and *how it's architected*; this directory holds engineering reports and task specs that capture *how a specific feature was built*.

## Layout

- `docs/` — Dated engineering reports written when a feature lands. Read these to understand the design choices, measured numbers, and known limitations of a specific subsystem.
- `tasks/` — Task specifications and planning docs. Read these to see the original scope, acceptance criteria, and step-by-step plan for in-flight or recently-shipped work.

## Reports (`docs/docs/`)

| Date | Report | Topic |
|---|---|---|
| 2026-04-30 | [`2026-04-30-pro6000-low-precision-profiling.md`](docs/2026-04-30-pro6000-low-precision-profiling.md) | FP16/FP8/FP4 profiling and per-`(op, dtype)` efficiency models on NVIDIA RTX PRO 6000 (Blackwell). Hardware peaks, profiling stats, XGBoost MAPE, end-to-end validation, known limits (FP8 attention on sm_120, FP4 attention in FlashInfer 0.6). |

## Task specs (`docs/tasks/`)

| Spec | Status |
|---|---|
| [`low_precision_profile.md`](tasks/low_precision_profile.md) | Shipped — see the 2026-04-30 report above. |

## Conventions

- **Report filenames:** `YYYY-MM-DD-<topic>.md` so they sort chronologically. Reports are append-only — corrections go in a later report, not by editing history.
- **Task filenames:** short kebab-case (`<topic>.md`). Drop status from the filename and track it in this index instead.
- **Cross-linking:** prefer relative paths (`../DESIGN.md`, `tasks/...`) so links survive a docs-site rebuild.
