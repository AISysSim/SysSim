# GH200 calibration data

Reference cost-model data for the **NVIDIA GH200 (Hopper H100)** accelerator, used by SysSim's
roofline+residual estimator (`HardwareConfig.calibrated_model: data/gh200`).

## What is here

| File | Produced by | Contents |
|------|-------------|----------|
| `profile.parquet` | `syssim profile` | One row per unique `(operator, input shape, dtype)` seen when running real Megatron transformer layers, with the measured GPU self-time. Columns: `op`, `count`, `per_instance_ns`, `signature` (JSON of the op's args/kwargs/output). |
| `gemm_model.lgb`, `elementwise_model.lgb`, `reduction_model.lgb` | `syssim calibrate` | Per-family LightGBM residual model: `log(measured_time / roofline_anchor)` as a function of op-size features. |
| `manifest.json` | `syssim calibrate` | Per-family feature columns, categorical codes, launch-latency floor, and fit metrics. |

There is **no notion of a named model** in this data — the unit is the `(op, shape)` a layer
dispatches. The same shape produced by different LLMs is the same kernel and is stored once. Every
op a Megatron transformer layer (explicit-attention local spec) dispatches routes to one of the
three families above, so the three models cover the full layer.

## This snapshot

- **23,016** unique `(op, shape)` rows from **158** layer-config jobs (0 failures).
- Held-out median APE: **gemm 6.2%, elementwise 6.7%, reduction 5.5%**.

## How to reproduce

Both steps run inside the project container. Profiling needs the GPUs (real Megatron layers on
CUDA); calibration is CPU-only.

```bash
# 1. Profile real Megatron layers over the architecture/shape space in the profiling spec.
#    --num-workers N spawns N GPU worker processes (one pinned per GPU). On a 4-GPU GH200 node:
syssim profile --out data/gh200 --num-workers 4

# 2. Fit the per-family residual models from the profile (CPU).
syssim calibrate --data data/gh200 \
    --hardware examples/configs/hardware/isambard_gh200_4gpu.yaml
```

The architecture/shape space is defined entirely by the committed spec
`syssim/profiling/default_spec.yaml` (per-field lists of hidden size, FFN size, head count, query
groups, head dim, vocab; plus `seq_len_range` / `token_range` for the sequence/batch sweep and
`parallelism.tensor_parallel` for the per-rank tensor-parallel shapes). Edit that spec to change
coverage. Sequence/batch points whose attention scores or lm-head logits would exceed GPU memory
are skipped automatically; coverage above those caps is reached by the residual tree interpolating
on op size.

### On Isambard-AI (slurm + podman-hpc)

```bash
# Profiling — on a reserved 4-GPU node:
srun --jobid=<JID> --overlap \
  podman-hpc run --rm --gpu --volume "$PWD":"$PWD" --workdir "$PWD" \
    --env PYTHONPATH="$PWD" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64 \
    localhost/mksit/syssim \
    python -m syssim.cli profile --out data/gh200 --num-workers 4

# Calibration — CPU, no GPU needed:
srun --jobid=<JID> --overlap \
  podman-hpc run --rm --volume "$PWD":"$PWD" --workdir "$PWD" --env PYTHONPATH="$PWD" \
    localhost/mksit/syssim \
    python -m syssim.cli calibrate --data data/gh200 \
      --hardware examples/configs/hardware/isambard_gh200_4gpu.yaml
```

## Reproducibility notes

- The **shape space, op signatures, and data schema are fully determined** by the committed spec —
  re-running `profile` reproduces the same set of `(op, shape)` rows.
- `per_instance_ns` values are **hardware measurements**, so a re-profile is statistically
  equivalent, not bit-identical (kernel timing jitter; duplicate shapes are reduced by median).
- `calibrate` is **seeded and deterministic** given a fixed `profile.parquet`: the same parquet
  yields the same `.lgb` models.
