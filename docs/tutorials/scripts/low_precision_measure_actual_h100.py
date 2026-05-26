"""Measure Tutorial 4 FP16/FP8 forward+backward runtime on one H100.

The default target is the Qwen3.5-9B-shaped configuration from the tutorial.
Use ``--model-scale reduced`` for an explicit smaller fallback when the full
configuration is too slow or runs out of memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, Qwen3Config


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = REPO_ROOT / "docs/tasks/results/low_precision_actual_h100.json"
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
    try:
        import torchao
    except Exception as exc:  # pragma: no cover - diagnostic path
        torchao_version = f"unavailable: {exc!r}"
    else:
        torchao_version = torchao.__version__

    payload: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torchao": torchao_version,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        payload.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return payload


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the actual H100 benchmark")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name and "Hopper" not in name:
        print(f"WARNING: detected {name}; this tutorial expects 1 x H100", file=sys.stderr)


def qwen_config(model_scale: str) -> Qwen3Config:
    if model_scale == "full":
        return Qwen3Config(**QWEN35_TEXT_CONFIG)
    if model_scale == "reduced":
        return Qwen3Config(**REDUCED_TEXT_CONFIG)
    raise ValueError(f"unknown model scale: {model_scale}")


def build_model(dtype: torch.dtype, model_scale: str) -> tuple[torch.nn.Module, Qwen3Config]:
    cfg = qwen_config(model_scale)
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, cfg


def make_batch(cfg: Qwen3Config, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(
        0,
        cfg.vocab_size,
        (batch_size, seq_len),
        device="cuda",
    )
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss if hasattr(out, "loss") else out[0]
    loss.backward()
    return loss


def convert_model_to_fp8_training(model: torch.nn.Module) -> torch.nn.Module:
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    return convert_to_float8_training(model, config=Float8LinearConfig())


def benchmark(
    *,
    label: str,
    model_scale: str,
    batch_size: int,
    seq_len: int,
    convert_to_fp8: bool,
    warmups: int,
    runs: int,
) -> dict[str, object]:
    model, cfg = build_model(torch.float16, model_scale)
    parameter_count = sum(p.numel() for p in model.parameters())
    if convert_to_fp8:
        model = convert_model_to_fp8_training(model)

    batch = make_batch(cfg, batch_size, seq_len)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    loss: torch.Tensor | None = None
    for _ in range(warmups):
        loss = one_step(model, batch)
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)

    timings_ms: list[float] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = one_step(model, batch)
        end.record()
        torch.cuda.synchronize()
        timings_ms.append(start.elapsed_time(end))
        model.zero_grad(set_to_none=True)

    result = {
        "label": label,
        "wall_time_ms": float(sum(timings_ms) / len(timings_ms)),
        "timings_ms": [float(t) for t in timings_ms],
        "peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1e9),
        "loss": float(loss.detach().float().cpu()) if loss is not None else None,
        "parameter_count": int(parameter_count),
        "converted_to_fp8_training": convert_to_fp8,
    }

    del batch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-scale", choices=["full", "reduced"], default="full")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    status = "ok"
    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for label, use_fp8 in (("actual_fp16", False), ("actual_fp8", True)):
        try:
            results.append(
                benchmark(
                    label=label,
                    model_scale=args.model_scale,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    convert_to_fp8=use_fp8,
                    warmups=args.warmups,
                    runs=args.runs,
                )
            )
        except Exception as exc:
            status = "partial"
            errors.append({"label": label, "error": repr(exc)})
            torch.cuda.empty_cache()

    payload = {
        "status": status,
        "model": MODEL_NAME,
        "model_scale": args.model_scale,
        "model_config": qwen_config(args.model_scale).to_dict(),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "hardware": torch.cuda.get_device_name(0),
        "step_scope": "forward_backward",
        "warmups": args.warmups,
        "runs": args.runs,
        "measurement_scope": "one warmup step followed by the average of three measured forward+backward repetitions when warmups=1 and runs=3",
        "environment": environment_summary(),
        "elapsed_s": time.time() - started,
        "results": results,
        "errors": errors,
        "notes": [
            "The step excludes optimizer updates.",
            "FP8 uses torchao.float8.convert_to_float8_training on Linear modules.",
        ],
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {RESULT_PATH.relative_to(REPO_ROOT)}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
