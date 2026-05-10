"""
Simulate Wan2.2 video diffusion inference on GH200 using syssim.

Wan2.2 is a video generation diffusion model from Wan-AI. It uses:
  - UMT5-XXL text encoder (cross_attention_dim=4096)
  - 3D DiT (Diffusion Transformer) denoiser operating on video latents
  - 3D VAE for encoding/decoding video frames

This example constructs a diffusers WanPipeline, extracts the major pipeline
stages, traces each stage with synthetic inputs matching Wan2.2 T2V-A14B, then
aggregates stage times using the pipeline scheduler and boundary ratio.

Wan2.2 T2V-A14B published specs:
  - Text encoder: UMT5-XXL (hidden_size=4096)
  - Transformer: 40 layers, 40 attention heads, head dim 128
  - Transformer stages: high-noise transformer + low-noise transformer
  - Latent channels: 16
  - VAE spatial compression: 8x, temporal compression: 4x
  - Default video: 480p (480x832), 81 frames (5s @ 16fps + 1)

Run:
    srun -N 1 --gpus 1 python examples/diffusion/simulate_wan2_2.py

Note:
    This example requires diffusers, but does not download Wan2.2 weights.
    It constructs the real diffusers model class on the meta device using
    the published transformer config and traces synthetic inputs.
"""

import os
import sys
from dataclasses import dataclass

# Ensure repo root is on path when invoked via srun without pip install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
from transformers import UMT5Config, UMT5EncoderModel

from syssim import HardwareInfo, SimulatorConfig, trace_diffusers_model_for_inference, trace_model_for_inference
from syssim.operator_graph import OperatorGraph, OperatorType

# ---- Wan2.2 architecture parameters ----
WAN2_2_T2V_A14B = dict(
    model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    boundary_ratio=0.875,
    latent_channels=16,
    vae_spatial_compression=8,
    vae_temporal_compression=4,
    cross_attention_dim=4096,
    text_encoder=dict(
        vocab_size=256384,
        d_model=4096,
        d_kv=64,
        d_ff=10240,
        num_layers=24,
        num_decoder_layers=24,
        num_heads=64,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu",
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        classifier_dropout=0.0,
        is_encoder_decoder=True,
    ),
    transformer=dict(
        patch_size=(1, 2, 2),
        num_attention_heads=40,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=13824,
        num_layers=40,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        image_dim=None,
        added_kv_proj_dim=None,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
    ),
    transformer_2=dict(
        patch_size=(1, 2, 2),
        num_attention_heads=40,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=13824,
        num_layers=40,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        image_dim=None,
        added_kv_proj_dim=None,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
    ),
    vae=dict(
        base_dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        scale_factor_temporal=4,
        scale_factor_spatial=8,
    ),
    scheduler=dict(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        solver_order=2,
        prediction_type="flow_prediction",
        thresholding=False,
        dynamic_thresholding_ratio=0.995,
        sample_max_value=1.0,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        disable_corrector=[],
        solver_p=None,
        use_karras_sigmas=False,
        use_exponential_sigmas=False,
        use_beta_sigmas=False,
        use_flow_sigmas=True,
        flow_shift=3.0,
        timestep_spacing="linspace",
        steps_offset=0,
        final_sigmas_type="zero",
        rescale_betas_zero_snr=False,
        use_dynamic_shifting=False,
        time_shift_type="exponential",
    ),
)

# ---- Video generation parameters ----
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 81       # 5 seconds @ 16fps + 1
NUM_STEPS = 50
GUIDANCE_SCALE = 7.5
PROMPT_LENGTH = 512   # UMT5-XXL token length


@dataclass
class StageTrace:
    name: str
    calls: int
    graph: OperatorGraph | None

    @property
    def ms_per_call(self) -> float:
        return self.graph.compute_critical_path() if self.graph is not None else 0.0

    @property
    def total_ms(self) -> float:
        return self.ms_per_call * self.calls


class WanVaeDecodeStage(nn.Module):
    """Wrap the exact WanPipeline VAE decode stage."""

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor, return_dict: bool = False):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        return self.vae.decode(latents, return_dict=return_dict)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_wan2_2_pipeline(arch: dict) -> WanPipeline:
    """Build the full WanPipeline from component configs without loading weights."""
    text_cfg = UMT5Config(**arch["text_encoder"])

    with torch.device("meta"):
        text_encoder = UMT5EncoderModel(text_cfg).to(dtype=torch.bfloat16)
        transformer = WanTransformer3DModel(**arch["transformer"]).to(dtype=torch.bfloat16)
        transformer_2 = WanTransformer3DModel(**arch["transformer_2"]).to(dtype=torch.bfloat16)
        vae = AutoencoderKLWan(**arch["vae"]).to(dtype=torch.bfloat16)

    scheduler = UniPCMultistepScheduler(**arch["scheduler"])
    pipeline = WanPipeline(
        tokenizer=None,
        text_encoder=text_encoder.eval(),
        vae=vae.eval(),
        scheduler=scheduler,
        transformer=transformer.eval(),
        transformer_2=transformer_2.eval(),
        boundary_ratio=arch["boundary_ratio"],
    )
    return pipeline


def build_text_encoder_inputs(
    *,
    batch_size: int,
    prompt_length: int,
    vocab_size: int,
) -> dict[str, torch.Tensor | bool]:
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, prompt_length)),
        "attention_mask": torch.ones((batch_size, prompt_length), dtype=torch.long),
        "return_dict": False,
    }


def build_wan2_2_inputs(
    *,
    height: int,
    width: int,
    num_frames: int,
    prompt_length: int,
    latent_channels: int,
    vae_scale_factor: int,
    temporal_compression: int,
    cross_attention_dim: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | bool]:
    """Build synthetic Wan2.2 denoiser inputs for this example."""
    latent_t = (num_frames - 1) // temporal_compression + 1
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    return {
        "hidden_states": torch.randn(
            batch_size, latent_channels, latent_t, latent_h, latent_w, dtype=dtype,
        ),
        "timestep": torch.full((batch_size,), 500, dtype=torch.long),
        "encoder_hidden_states": torch.randn(
            batch_size, prompt_length, cross_attention_dim, dtype=dtype,
        ),
        "return_dict": False,
    }


def build_vae_decode_inputs(
    *,
    height: int,
    width: int,
    num_frames: int,
    latent_channels: int,
    vae_scale_factor: int,
    temporal_compression: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | bool]:
    latent_t = (num_frames - 1) // temporal_compression + 1
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    return {
        "latents": torch.randn(batch_size, latent_channels, latent_t, latent_h, latent_w, dtype=dtype),
        "return_dict": False,
    }


def denoising_step_counts(pipeline: WanPipeline, num_steps: int) -> tuple[int, int]:
    """Return high-noise and low-noise transformer step counts from the scheduler."""
    pipeline.scheduler.set_timesteps(num_steps)

    if pipeline.config.boundary_ratio is None or pipeline.transformer_2 is None:
        return num_steps, 0

    boundary_timestep = pipeline.config.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
    high_noise_steps = int((pipeline.scheduler.timesteps >= boundary_timestep).sum().item())
    low_noise_steps = int((pipeline.scheduler.timesteps < boundary_timestep).sum().item())
    return high_noise_steps, low_noise_steps


def print_operator_breakdown(label: str, graph: OperatorGraph) -> None:
    type_counts: dict[OperatorType, int] = {}
    type_times: dict[OperatorType, float] = {}
    for op in graph.operators.values():
        type_counts[op.op_type] = type_counts.get(op.op_type, 0) + 1
        type_times[op.op_type] = type_times.get(op.op_type, 0.0) + op.estimated_time_ms

    print(f"Operator breakdown ({label}):")
    print(f"  {'Type':<12} {'Count':>6} {'Time (ms)':>12} {'% of total':>10}")
    total_time = sum(type_times.values())
    for op_type in OperatorType:
        count = type_counts.get(op_type, 0)
        time = type_times.get(op_type, 0.0)
        if count:
            pct = 100.0 * time / total_time if total_time > 0 else 0.0
            print(f"  {op_type.name:<12} {count:>6} {time:>12.4f} {pct:>9.1f}%")


def main():
    # --- Hardware ---
    # Use GH200 specs directly (works without CUDA device present)
    hw = HardwareInfo(
        peak_tflops_mm=989.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )
    print("Hardware: GH200 (Grace Hopper)")
    print(f"  Peak MM TFLOP/s : {hw.peak_tflops_mm:.1f}")
    print(f"  Peak BW GB/s    : {hw.peak_memory_bandwidth_gbps:.1f}")
    print()

    # --- Model ---
    arch = WAN2_2_T2V_A14B
    transformer_cfg = arch["transformer"]
    temporal_compression = arch["vae_temporal_compression"]
    spatial_compression = arch["vae_spatial_compression"]
    hidden_size = transformer_cfg["num_attention_heads"] * transformer_cfg["attention_head_dim"]

    latent_t = (NUM_FRAMES - 1) // temporal_compression + 1
    latent_h = HEIGHT // spatial_compression
    latent_w = WIDTH // spatial_compression

    print("Wan2.2 T2V-A14B Pipeline (diffusers WanPipeline)")
    print(f"  Video: {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames")
    print(f"  Latent: {arch['latent_channels']}x{latent_t}x{latent_h}x{latent_w}")
    print(f"  Text encoder hidden size: {arch['text_encoder']['d_model']}")
    print(f"  Text encoder layers: {arch['text_encoder']['num_layers']}")
    print(f"  Layers: {transformer_cfg['num_layers']}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Attention heads: {transformer_cfg['num_attention_heads']}")
    print(f"  FFN size: {transformer_cfg['ffn_dim']}")
    print(f"  Boundary ratio: {arch['boundary_ratio']}")
    print(f"  Prompt length: {PROMPT_LENGTH}")
    print()

    print("Building diffusers WanPipeline components (meta device)...")
    pipeline = build_wan2_2_pipeline(arch)
    print(f"  Text encoder parameters: {param_count(pipeline.text_encoder) / 1e9:.2f}B")
    print(f"  Transformer parameters : {param_count(pipeline.transformer) / 1e9:.2f}B")
    print(f"  Transformer_2 params   : {param_count(pipeline.transformer_2) / 1e9:.2f}B")
    print(f"  VAE parameters         : {param_count(pipeline.vae) / 1e9:.2f}B")
    print()

    # --- Inputs ---
    text_inputs = build_text_encoder_inputs(
        batch_size=1,
        prompt_length=PROMPT_LENGTH,
        vocab_size=arch["text_encoder"]["vocab_size"],
    )
    denoise_inputs = build_wan2_2_inputs(
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        prompt_length=PROMPT_LENGTH,
        latent_channels=arch["latent_channels"],
        vae_scale_factor=spatial_compression,
        temporal_compression=temporal_compression,
        cross_attention_dim=arch["cross_attention_dim"],
    )
    vae_inputs = build_vae_decode_inputs(
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        latent_channels=arch["latent_channels"],
        vae_scale_factor=spatial_compression,
        temporal_compression=temporal_compression,
    )

    # --- Trace ---
    sim_cfg = SimulatorConfig(hw_info=hw)
    cfg_multiplier = 2 if GUIDANCE_SCALE > 1.0 else 1
    text_encoder_calls = cfg_multiplier
    high_noise_steps, low_noise_steps = denoising_step_counts(pipeline, NUM_STEPS)
    high_noise_calls = high_noise_steps * cfg_multiplier
    low_noise_calls = low_noise_steps * cfg_multiplier

    print("Tracing text encoder stage...")
    text_graph = trace_diffusers_model_for_inference(
        pipeline,
        text_inputs,
        sim_cfg,
        component="text_encoder",
    )

    print("Tracing high-noise transformer stage...")
    transformer_graph = trace_diffusers_model_for_inference(
        pipeline,
        denoise_inputs,
        sim_cfg,
        component="transformer",
    )

    print("Tracing low-noise transformer stage...")
    transformer_2_graph = trace_diffusers_model_for_inference(
        pipeline,
        denoise_inputs,
        sim_cfg,
        component="transformer_2",
    )

    print("Tracing VAE decode stage...")
    vae_graph = trace_model_for_inference(
        WanVaeDecodeStage(pipeline.vae).eval(),
        vae_inputs,
        sim_cfg,
    )
    print()

    stages = [
        StageTrace("Text encoder", text_encoder_calls, text_graph),
        StageTrace("High-noise transformer", high_noise_calls, transformer_graph),
        StageTrace("Low-noise transformer", low_noise_calls, transformer_2_graph),
        StageTrace("VAE decoder", 1, vae_graph),
        StageTrace("Scheduler step (not traced)", NUM_STEPS, None),
    ]
    total_pipeline_ms = sum(stage.total_ms for stage in stages)

    # --- Report ---
    print("=== Wan2.2 Diffusion Simulation ===")
    print(f"Denoising steps     : {NUM_STEPS}")
    print(f"Guidance scale      : {GUIDANCE_SCALE} (CFG passes/step: {cfg_multiplier})")
    print(f"High-noise steps    : {high_noise_steps}")
    print(f"Low-noise steps     : {low_noise_steps}")
    print(f"Video frames        : {NUM_FRAMES}")
    print()
    print(f"{'Stage':<30} {'Calls':>7} {'ms/call':>12} {'Total (ms)':>14}")
    for stage in stages:
        print(f"{stage.name:<30} {stage.calls:>7} {stage.ms_per_call:>12.2f} {stage.total_ms:>14.2f}")
    print()
    print(f"Total pipeline      : {total_pipeline_ms:.2f} ms")
    print()
    print("--- Stage graph summaries ---")
    print(text_graph.summary())
    print()
    print(transformer_graph.summary())
    print()
    print(transformer_2_graph.summary())
    print()
    print(vae_graph.summary())
    print()
    print_operator_breakdown("text encoder", text_graph)
    print()
    print_operator_breakdown("high-noise transformer step", transformer_graph)
    print()
    print_operator_breakdown("low-noise transformer step", transformer_2_graph)
    print()
    print_operator_breakdown("VAE decode", vae_graph)


if __name__ == "__main__":
    main()
