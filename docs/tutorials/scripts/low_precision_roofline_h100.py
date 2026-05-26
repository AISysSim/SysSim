"""Run Tutorial 4 SysSim roofline predictions for FP16 and FP8 on H100."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, Qwen3Config

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from syssim import HardwareInfo, SimulatorConfig
from syssim.integrations.huggingface import trace_hf_model_for_training


RESULT_PATH = REPO_ROOT / "docs/tasks/results/low_precision_roofline_h100.json"
MODEL_NAME = "Qwen/Qwen3.5-9B"

QWEN35_TEXT_CONFIG = {
    "vocab_size": 248320,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}

REDUCED_TEXT_CONFIG = {
    "vocab_size": 32000,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "max_position_embeddings": 8192,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def environment_summary() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def qwen_config(model_scale: str) -> Qwen3Config:
    if model_scale == "full":
        return Qwen3Config(**QWEN35_TEXT_CONFIG)
    if model_scale == "reduced":
        return Qwen3Config(**REDUCED_TEXT_CONFIG)
    raise ValueError(f"unknown model scale: {model_scale}")


def h100_fp16_config() -> HardwareInfo:
    return HardwareInfo(
        peak_tflops_mm=1979.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
        peak_tflops_mm_conservative=535.0,
        peak_tflops_mm_fp8=3958.0,
    )


def h100_fp8_roofline_config() -> HardwareInfo:
    return HardwareInfo(
        peak_tflops_mm=3958.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
        peak_tflops_mm_conservative=1070.0,
        peak_tflops_mm_fp8=3958.0,
    )


def model_state_memory_gb(parameter_count: int, parameter_bytes: float, gradient_bytes: float) -> dict[str, float]:
    parameters_gb = parameter_count * parameter_bytes / 1e9
    gradients_gb = parameter_count * gradient_bytes / 1e9
    return {
        "parameters_gb": parameters_gb,
        "gradients_gb": gradients_gb,
        "optimizer_states_gb": 0.0,
        "total_model_state_gb": parameters_gb + gradients_gb,
    }


def trace_once(
    *,
    label: str,
    model_scale: str,
    batch_size: int,
    seq_len: int,
    hw_info: HardwareInfo,
    parameter_bytes: float,
    gradient_bytes: float,
) -> dict[str, object]:
    cfg = qwen_config(model_scale)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16)
    model.train()
    model.config.use_cache = False

    parameter_count = sum(p.numel() for p in model.parameters())
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    inputs = {"input_ids": input_ids, "labels": input_ids.clone()}

    graph = trace_hf_model_for_training(
        model,
        inputs,
        SimulatorConfig(hw_info=hw_info),
    )
    return {
        "label": label,
        "wall_time_ms": graph.compute_critical_path(),
        "operators_traced": len(graph),
        "parameter_count": int(parameter_count),
        "memory": model_state_memory_gb(
            parameter_count,
            parameter_bytes=parameter_bytes,
            gradient_bytes=gradient_bytes,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-scale", choices=["full", "reduced"], default="full")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SysSim tracing requires a CUDA-capable device")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.environ.pop("RLSYSIM_MODEL_DIR", None)

    predictions = [
        trace_once(
            label="roofline_fp16",
            model_scale=args.model_scale,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hw_info=h100_fp16_config(),
            parameter_bytes=2.0,
            gradient_bytes=2.0,
        ),
        trace_once(
            label="roofline_fp8",
            model_scale=args.model_scale,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hw_info=h100_fp8_roofline_config(),
            parameter_bytes=1.0,
            gradient_bytes=2.0,
        ),
    ]
    payload = {
        "status": "ok",
        "model": MODEL_NAME,
        "model_scale": args.model_scale,
        "model_config": qwen_config(args.model_scale).to_dict(),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "hardware": "1 x H100",
        "step_scope": "forward_backward",
        "environment": environment_summary(),
        "predictions": predictions,
        "notes": [
            "FP8 roofline uses H100 FP8 tensor peak for GEMM/attention.",
            "Gradients are counted as FP16 because typical FP8 training keeps higher-precision gradients.",
        ],
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {RESULT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
