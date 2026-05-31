# Megatron ↔ SysSim validation (GH200)

Validate SysSim's predicted **step time** and **per-GPU peak memory** against **real
megatron-core** training runs, on a single Grace-Hopper (GH200) node. Both sides train the
*same* megatron-core `GPTModel` (built from the same model YAML via `resolve_megatron_provider`
+ the local GPT layer spec), so the comparison is apples-to-apples.

The full results + an HTML report live in [`docs/megatron_gh200_validation/`](../../docs/megatron_gh200_validation/).

## The matrix

The case matrix is the single source of truth in [`run_syssim.sh`](run_syssim.sh):

| dimension | values |
|---|---|
| model | `llama3-8b`, `qwen3-8b` (the YAMLs here) |
| parallelism | `1gpu` (TP1·DP1), `tp2`, `tp4`, `dp4` (single node) |
| recompute | `none`, `full` (each case is reported both ways) |
| optimizer | DP runs use megatron's **distributed optimizer** (ZeRO-1) |

`micro_batch=1`, `global_batch = DP` (one micro-batch per rank, no gradient accumulation),
`bf16`, 8 train iters (3 warmup dropped, median of the rest).

## Files

| file | role |
|---|---|
| `llama3-8b.yaml`, `qwen3-8b.yaml` | model architecture (SysSim model-YAML schema), the single arch source |
| `run_megatron.py` | **real** per-case runner: a short megatron-core fwd+bwd+optimizer loop → `{step_time_ms, peak_memory_gb, oom}` |
| `run_megatron.sh` | thin per-case wrapper (`torchrun --nproc_per_node=TP*DP run_megatron.py …`) |
| `_sim_case.py` | **sim** per-case runner: `syssim.simulate(...)` → the same JSON |
| `run_syssim.sh` | runs SysSim over the whole matrix (defines the CASES) |
| `compare.py` | joins `*.real.json` + `*.sim.json` → `summary.json` (% error, pass/fail vs 10%) |
| `report.py` | renders `report.html` (tables + embedded sim-vs-real plots) |
| `../configs/hardware/gh200.yaml` | GH200 hardware config |

## Reproduce (no slurm)

Everything runs in a container that has megatron-core + transformer-engine + a CUDA PyTorch
(here: `localhost/mksit/syssim`) on a machine with GH200 GPUs. Standard run template:

```bash
RUN="podman-hpc run --rm --gpu --volume $PWD:$PWD --workdir $PWD \
  --env PYTHONPATH=$PWD --env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64 \
  localhost/mksit/syssim"

# 1. Real megatron-core, one case (nproc = TP*DP).  Args: model tp dp mbs gbs recompute distopt out
$RUN bash examples/megatron/run_megatron.sh examples/megatron/qwen3-8b.yaml 4 1 1 1 none 0 \
        docs/megatron_gh200_validation/results/qwen3-8b_tp4_norc.real.json
#    …repeat per case, or loop the matrix (see the CASES in run_syssim.sh).

# 2. SysSim over the whole matrix:
$RUN bash examples/megatron/run_syssim.sh

# 3. Compare + report:
$RUN python examples/megatron/compare.py --results docs/megatron_gh200_validation/results --tol 0.10
$RUN python examples/megatron/report.py        # -> docs/megatron_gh200_validation/report.html
```

Result files are named `<model>_<case>_<rc|norc>.{real,sim}.json`; `compare.py` joins them by that key.

> **slurm:** the cluster-specific batch wrapper (reserve a node, run the whole matrix) is
> intentionally *not* shipped here — it lives in `agent_space/` on the cluster. On Isambard-AI
> it's `sbatch agent_space/megatron_gh200_validation/submit.sbatch`; adapt to your scheduler.
