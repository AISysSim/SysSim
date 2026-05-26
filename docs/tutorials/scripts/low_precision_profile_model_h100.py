"""Run a Tutorial 4 FP8 profiling-model workflow on one H100.

The default profile grid is reduced and each kernel shape is measured once so
the tutorial can produce real H100 JSON and local chart artifacts as a smoke
workflow. Use ``--profile-scale tutorial`` and increase ``--num-runs`` for a
larger calibration grid.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, Qwen3Config

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import syssim.compute.compute_cost_profiler as profiler
from syssim import HardwareInfo, SimulatorConfig, set_efficiency_model_dir
from syssim.compute.compute_cost_profiler import profile_operator, train_efficiency_model
from syssim.integrations.huggingface import trace_hf_model_for_training


PROFILE_DIR = REPO_ROOT / "data/profiling"
MODEL_DIR = REPO_ROOT / "data/trained_models"
RESULT_PATH = REPO_ROOT / "docs/tasks/results/low_precision_profile_model_h100.json"
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
    import pandas
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def qwen_config(model_scale: str) -> Qwen3Config:
    if model_scale == "full":
        return Qwen3Config(**QWEN35_TEXT_CONFIG)
    if model_scale == "reduced":
        return Qwen3Config(**REDUCED_TEXT_CONFIG)
    raise ValueError(f"unknown model scale: {model_scale}")


def h100_fp8_config() -> HardwareInfo:
    return HardwareInfo(
        peak_tflops_mm=3958.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
        peak_tflops_mm_conservative=1070.0,
        peak_tflops_mm_fp8=3958.0,
    )


def set_profile_grids(profile_scale: str) -> dict[str, object]:
    if profile_scale == "tutorial":
        profiler.COMPUTE_GRIDS["gemm"] = {
            "M": [512, 1024, 2048, 4096],
            "N": [4096, 8192, 12288, 24832],
            "K": [1024, 4096, 8192, 12288],
        }
        profiler.COMPUTE_GRIDS["attn"] = {
            "bs": [1],
            "seq": [512, 1024, 2048, 4096],
            "nh": [8, 16, 32],
            "nkv": [1, 2, 4],
            "hd": [128, 256],
        }
        return {"scale": "tutorial", "gemm_configs": 64, "attn_configs": 72}

    if profile_scale == "reduced":
        profiler.COMPUTE_GRIDS["gemm"] = {
            "M": [512, 1024],
            "N": [1024, 4096, 8192],
            "K": [1024, 4096],
        }
        profiler.COMPUTE_GRIDS["attn"] = {
            "bs": [1],
            "seq": [512, 1024, 2048],
            "nh": [8, 16],
            "nkv": [1, 2],
            "hd": [128],
        }
        return {"scale": "reduced", "gemm_configs": 12, "attn_configs": 12}

    raise ValueError(f"unknown profile scale: {profile_scale}")


def train_operator(operator: str, csv_path: Path, dtype: str) -> dict[str, object]:
    backend_suffix = "xgb"
    output_path = MODEL_DIR / f"{operator}_h100_{dtype}_{backend_suffix}.pth"
    metrics, eff_mape, time_mape = train_efficiency_model(
        operator=operator,
        csv_path=csv_path,
        output_path=str(output_path),
        backend="xgboost",
        dtype=dtype,
    )
    return {
        "operator": operator,
        "model_path": str(output_path.relative_to(REPO_ROOT)),
        "cv_metrics": metrics,
        "eff_mape": float(eff_mape),
        "time_mape": float(time_mape),
    }


def profile_and_train(num_runs: int, profile_scale: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grid_summary = set_profile_grids(profile_scale)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    profile_results: list[dict[str, object]] = []
    model_results: list[dict[str, object]] = []
    for operator in ("gemm", "attn"):
        csv_path = profile_operator(
            operator,
            str(PROFILE_DIR),
            num_runs=num_runs,
            dtype="fp8",
        )
        import pandas as pd

        df = pd.read_csv(csv_path)
        valid_rows = int((df["t_measured_ms"] > 0).sum())
        profile_entry = {
            "operator": operator,
            "csv_path": str(csv_path.relative_to(REPO_ROOT)),
            "rows": int(len(df)),
            "valid_rows": valid_rows,
            "invalid_rows": int(len(df) - valid_rows),
        }
        profile_results.append(profile_entry)
        if valid_rows >= 5:
            model_results.append(train_operator(operator, csv_path, "fp8"))
        else:
            profile_entry["training_skipped"] = "fewer than 5 valid rows for 5-fold cross validation"

    for entry in profile_results:
        entry["profile_grid"] = grid_summary
    return profile_results, model_results


def predict_qwen_step(model_scale: str, batch_size: int, seq_len: int) -> dict[str, object]:
    cfg = qwen_config(model_scale)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16)
    model.train()
    model.config.use_cache = False

    parameter_count = sum(p.numel() for p in model.parameters())
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    inputs = {"input_ids": input_ids, "labels": input_ids.clone()}

    set_efficiency_model_dir(str(MODEL_DIR.resolve()))
    os.environ["SYSSIM_FORCE_DTYPE"] = "fp8"
    graph = trace_hf_model_for_training(
        model,
        inputs,
        SimulatorConfig(hw_info=h100_fp8_config()),
    )
    return {
        "label": "profiling_model_fp8",
        "wall_time_ms": graph.compute_critical_path(),
        "operators_traced": len(graph),
        "parameter_count": int(parameter_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-scale", choices=["full", "reduced"], default="full")
    parser.add_argument("--profile-scale", choices=["tutorial", "reduced"], default="reduced")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-runs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP8 profiling")

    started = time.time()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile_results, model_results = profile_and_train(args.num_runs, args.profile_scale)
    prediction = predict_qwen_step(args.model_scale, args.batch_size, args.seq_len)

    payload = {
        "status": "ok",
        "model": MODEL_NAME,
        "model_scale": args.model_scale,
        "model_config": qwen_config(args.model_scale).to_dict(),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "hardware": torch.cuda.get_device_name(0),
        "profile_scale": args.profile_scale,
        "num_runs": args.num_runs,
        "environment": environment_summary(),
        "profiles": profile_results,
        "trained_models": model_results,
        "predictions": [prediction],
        "elapsed_s": time.time() - started,
        "limitations": [
            "The default profiling grid is a one-run reduced smoke workflow; use --profile-scale tutorial and increase --num-runs for calibration.",
            "Only operators with at least 5 valid FP8 rows are trained because train_efficiency_model uses 5-fold CV.",
        ],
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {RESULT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
