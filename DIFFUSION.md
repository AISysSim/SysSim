# Diffusion Simulation

This document describes how SysSim models Wan2.2 diffusion inference. Diffusion
support intentionally reuses the existing PyTorch inference tracer rather than
adding a diffusion-specific execution mode or a public pipeline tracing API.

For the core tracing and runtime model, see [DESIGN.md](DESIGN.md).

---

## 1. API Shape

Diffusers support is a thin adapter around the existing inference path:

```python
graph = trace_diffusers_model_for_inference(
    pipeline,
    example_inputs,
    sim_cfg,
    component="transformer",
)
```

The adapter selects an `nn.Module` component from the object when `component` is
provided, validates that the selected object is traceable, and calls:

```python
trace_model_for_inference(model, example_inputs, sim_cfg, mode="prefill")
```

There is no `trace_diffusion_pipeline()` helper. Pipeline construction,
model-specific input helpers, stage selection, and aggregation all live in the
example script.

---

## 2. Wan2.2 Example

The reference example is [examples/diffusion/simulate_wan2_2.py](examples/diffusion/simulate_wan2_2.py).
It builds a diffusers `WanPipeline` from component configs on the `meta` device,
so it does not download or allocate Wan weights.

The example traces these stages:

- `text_encoder`: `UMT5EncoderModel`
- `transformer`: high-noise `WanTransformer3DModel`
- `transformer_2`: low-noise `WanTransformer3DModel`
- `vae.decode`: wrapped in an example-local module because `decode()` is not the
  VAE module's `forward()`

The scheduler is instantiated from the Wan2.2 scheduler config and used for
timestep accounting. Scheduler tensor updates are not traced.

---

## 3. Stage Inputs

The example creates synthetic inputs with Wan2.2-style dimensions:

```text
video frames            81
height x width          480 x 832
latent channels         16
VAE spatial compression 8x
VAE temporal compression 4x
prompt length           512
text hidden size        4096
```

The denoiser latent shape is:

```text
latent_T = (num_frames - 1) // temporal_compression + 1
latent_H = height // spatial_compression
latent_W = width // spatial_compression
```

For the default 81-frame, 480x832 run this gives:

```text
hidden_states          (B, 16, 21, 60, 104)
timestep               (B,)
encoder_hidden_states  (B, 512, 4096)
```

Text encoder inputs are synthetic token IDs and attention masks shaped
`(B, 512)`. VAE decode inputs reuse the same latent shape; the example applies
Wan's latent mean/std transform before calling `vae.decode()`, matching the
diffusers pipeline path.

---

## 4. Aggregation

The example traces one call for each stage, then aggregates using the pipeline's
scheduler and boundary ratio. For Wan2.2 T2V-A14B, `boundary_ratio=0.875`; with
50 inference steps this produces 16 high-noise steps and 34 low-noise steps.

Classifier-free guidance is modeled as a pass multiplier:

```text
cfg_passes_per_step = 2 if guidance_scale > 1.0 else 1
```

The total is:

```text
T_text     = T_text_call * text_encoder_calls
T_high     = T_high_step * high_noise_steps * cfg_passes_per_step
T_low      = T_low_step * low_noise_steps * cfg_passes_per_step
T_vae      = T_vae_decode
T_pipeline = T_text + T_high + T_low + T_vae
```

`text_encoder_calls` is also `2` when classifier-free guidance is enabled,
representing prompt and negative-prompt encoding. CFG batch folding is not
modeled; the example counts separate conditional and unconditional passes.

---

## 5. Attention Semantics

Diffusion stages are traced in `prefill` mode because a denoising step is a
full latent forward pass, not an autoregressive token decode with a growing KV
cache.

Self-attention and cross-attention are both captured as SDPA or flash-attention
operators. Runtime estimation uses the actual traced query/key/value shapes, so
cross-attention is naturally different from self-attention:

```text
self-attention  S_q == S_kv  video latent tokens
cross-attention S_q != S_kv  video latent queries over text tokens
```

For the default Wan2.2 run, each transformer stage contains 40 self-attention
ops with `q_len=32760, kv_len=32760` and 40 cross-attention ops with
`q_len=32760, kv_len=512`.

`mode="decode"` is reserved for autoregressive LLM inference. In decode mode,
SysSim overrides attention K/V length with `SimulatorConfig.cache_seq_len` to
approximate KV-cache reads. That path is not used for Wan diffusion.

---

## 6. Current Limits

- Scheduler tensor updates are counted as steps but not traced.
- CFG batching is not modeled; CFG is represented as separate passes.
- Tokenization and final video post-processing are not traced.
- Text encoder and VAE inputs are synthetic; no prompt text or image/video data
  is loaded.
- Multi-GPU diffusion execution is not connected to the network simulator.
