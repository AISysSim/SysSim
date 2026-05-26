# Plena Requirements for SysSim Roofline Estimation

This note lists the basic Plena hardware information SysSim needs for a roofline-only training-step estimate on one accelerator.

## 1. Roofline Model

SysSim can estimate forward and backward operator time without a trained ML efficiency model by using a roofline model. In this mode, each traced PyTorch op is classified as GEMM, attention, generic math, memory, or communication. For each op, SysSim computes:

```text
compute_time = FLOPs / peak_FLOP_s
memory_time  = bytes_read_written / peak_memory_bandwidth
op_time      = max(compute_time, memory_time)
```

For Plena, we need a `HardwareInfo` entry with these fields:

```python
HardwareInfo(
    peak_tflops_mm=...,              # matrix/GEMM peak
    peak_tflops_math=...,            # vector/scalar/math peak
    peak_memory_bandwidth_gbps=...,  # HBM bandwidth
    peak_tflops_mm_conservative=..., # lower peak for small/inefficient ops
)
```

Required values from Plena:

- `peak_tflops_mm`: sustained or theoretical matrix-unit peak in TFLOP/s for dense GEMM-like work. Please specify the datatype, for example BF16, FP16, FP8, or Plena native quantized format.
- `peak_tflops_math`: sustained or theoretical vector/scalar peak in TFLOP/s for elementwise math, reductions, normalization, activation functions, and scalar math.
- `peak_memory_bandwidth_gbps`: usable HBM bandwidth in GB/s. `HBM_WIDTH` alone is not enough; SysSim needs bandwidth after applying clock rate, channels, and practical efficiency assumptions.
- `peak_tflops_mm_conservative`: lower effective GEMM/attention peak for small or poorly tiled ops. SysSim uses this when matrix dimensions are below its large-op threshold. If Plena has tile utilization curves, this can start as a single representative value and later become a shape-aware model.

Please also clarify whether these numbers are theoretical peak, measured microbenchmark peak, or expected sustained application-level peak. SysSim can use any of them, but the interpretation of the final estimate depends on this choice.

## 2. Profiling Data for the Efficiency Model

SysSim can improve the roofline estimate with an ML efficiency model. To train that model for Plena, SysSim needs profiling data: one row per operator shape, with the latency measured or simulated on Plena.

The required CSV format is:

```text
operator_shape_columns...,t_measured_ms
```

The current SysSim profiling grids are defined in `syssim/compute/compute_cost_profiler.py`. Look for `COMPUTE_GRIDS` and these construction functions: `construct_dataset_gemm`, `construct_dataset_attn`, `construct_dataset_rmsnorm`, and `construct_dataset_silu`. Those functions define the shape ranges and sampling strategy used to generate the CSV rows.

For the first Plena target, the most important operator classes are:

- `gemm`: matrix multiplication / linear layers.
- `attn`: scaled dot-product attention.
- `rmsnorm`: RMS normalization used by Llama-style models.
- `silu`: SiLU activation used in SwiGLU/MLP blocks.

Example GEMM rows:

```csv
M,N,K,t_measured_ms
1024,4096,4096,0.082
4096,14336,4096,0.310
4096,4096,14336,0.295
```

Example attention rows:

```csv
bs,seq,nh,nkv,hd,t_measured_ms
1,2048,32,8,128,0.930
4,2048,32,8,128,3.410
```

Example RMSNorm or SiLU rows:

```csv
seq,dim,t_measured_ms
2048,4096,0.045
8192,4096,0.160
```

The latency can come from real Plena hardware, a validated Plena latency model, or the Plena simulator. The important requirement is that each row gives the expected Plena latency for that exact operator shape, in milliseconds.

For Llama training, the profiling data should include both forward-like and backward-like shapes. For GEMM, backward usually uses the same dimensions as the forward linear layer, but permuted:

```text
forward: M=batch*seq, K=in,        N=out
dX:      M=batch*seq, K=out,       N=in
dW:      M=in,        K=batch*seq, N=out
```

If Plena cannot provide measured data for all shapes, a simulated latency model is still useful as long as it can generate the same CSV fields for the target operator grid.

## 3. Multi-GPU Training Estimates

For a `2 x 8 H100` or similar multi-GPU training estimate, the single-GPU roofline inputs are not enough. The estimate has two parts:

1. **Per-rank compute time.** Trace the model shape that each GPU actually sees. For data parallelism, this means the local batch size. For tensor parallelism, GEMM and attention shapes are already sharded by the model-parallel framework.
2. **Communication time.** Add distributed collectives separately, such as all-reduce, reduce-scatter, all-gather, and broadcast, with their message sizes and participating ranks.

The user-facing information needed for this estimate is:

- Cluster shape: `num_nodes`, `gpus_per_node`, and total ranks.
- GPU hardware: compute peaks and memory bandwidth per GPU.
- Parallelism strategy: DP, TP, PP, FSDP/ZeRO, or a hybrid.
- Training shape: global batch size, local batch size, microbatch size, sequence length, and gradient accumulation.
- Network topology: intra-node NVLink/NVSwitch bandwidth and latency, plus inter-node InfiniBand bandwidth and latency.
- Communication schedule: which collectives occur, which ranks participate, message sizes in bytes, and whether communication overlaps with backward compute.

Current SysSim status:

- `trace_model_for_training(...)` estimates the traced compute graph, but it does not automatically add NCCL communication.
- The tracer patches `torch.distributed` collectives to no-op during tracing so fake tensors do not enter real process-group calls.
- `NetworkParams` exists on `HardwareInfo`, but it is not currently wired into the training tracer as an end-to-end multi-GPU estimator.
- Communication can be modeled manually with `syssim.network`, for example by building collectives with `allreduce(...)`, `reduce_scatter(...)`, or `allgather(...)`, then passing them to `simulate(...)` with a `HierarchicalTopology`.

In other words, SysSim has the compute tracer and the network simulator pieces, but it does not yet provide a single config file or CLI where a user can fill all `2 x 8 H100` training parameters and receive a complete training-step estimate automatically.

