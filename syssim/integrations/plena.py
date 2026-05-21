"""High-level PLENA model estimation API.

Provides layer-level estimation using PLENA's native LLaMAModel
for optimized LLM performance prediction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class PLENAModelConfig:
    """Model configuration for PLENA layer-level estimation.

    Holds the architectural parameters needed by PLENA's LLaMAModel.

    Attributes:
        hidden_size: Hidden dimension of the model
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        num_hidden_layers: Number of transformer layers
        intermediate_size: FFN intermediate dimension
        vocab_size: Vocabulary size
        head_dim: Head dimension (default: hidden_size // num_attention_heads)
    """

    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    head_dim: int = field(default=0)

    def __post_init__(self):
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "PLENAModelConfig":
        """Create PLENAModelConfig from HuggingFace model config.

        Args:
            hf_config: HuggingFace model config (e.g., LlamaConfig)

        Returns:
            PLENAModelConfig with extracted parameters

        Example:
            >>> from transformers import AutoConfig
            >>> hf_config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
            >>> plena_config = PLENAModelConfig.from_hf_config(hf_config)
        """
        # Handle different config attribute names
        num_kv_heads = getattr(
            hf_config,
            "num_key_value_heads",
            getattr(hf_config, "num_attention_heads", 32),
        )

        return cls(
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            num_hidden_layers=hf_config.num_hidden_layers,
            intermediate_size=hf_config.intermediate_size,
            vocab_size=hf_config.vocab_size,
        )

    def to_model_param_dict(self) -> dict:
        """Convert to dict format expected by PLENA LLaMAModel."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_kv_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
        }


@dataclass
class PLENAInferenceResult:
    """Results from PLENA inference estimation.

    Attributes:
        prefill_ms: Prefill phase time in milliseconds
        decode_ms: Decode phase time in milliseconds
        ttft_ms: Time to first token in milliseconds
        tps: Tokens per second (throughput)
        total_cycles: Total cycle count (for debugging)
    """

    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    tps: float
    total_cycles: int = 0

    @property
    def prefill_s(self) -> float:
        """Prefill time in seconds."""
        return self.prefill_ms / 1000.0

    @property
    def decode_s(self) -> float:
        """Decode time in seconds."""
        return self.decode_ms / 1000.0


def _load_plena_llama_model(
    model_config: PLENAModelConfig,
    settings_path: str,
    isa_lib_path: str,
    batch_size: int = 1,
    input_seq_len: int = 2048,
    output_seq_len: int = 128,
    device_num: int = 1,
    frequency_hz: float = 1e9,
):
    """Load PLENA LLaMAModel with given configuration.

    Returns a wrapped LLaMAModel instance.
    """
    import json
    import tempfile

    # Add PLENA_Simulator to path
    plena_base = Path(settings_path).parent
    plena_perf_dir = plena_base / "analytic_models" / "performance"
    if str(plena_perf_dir) not in sys.path:
        sys.path.insert(0, str(plena_perf_dir))

    from llama_model import LLaMAModel
    from perf_model import load_hardware_config_from_toml

    # Create temporary model config file
    model_params = model_config.to_model_param_dict()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(model_params, f)
        model_config_path = f.name

    hardware_config = load_hardware_config_from_toml(settings_path)

    return LLaMAModel(
        model_config_path=model_config_path,
        hardware_config=hardware_config,
        custom_isa_path=isa_lib_path,
        batch_size=batch_size,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        device_num=device_num,
        frequency_hz=frequency_hz,
    )


def estimate_plena_inference(
    model_config: PLENAModelConfig,
    settings_path: str,
    isa_lib_path: str,
    batch_size: int = 1,
    input_seq_len: int = 2048,
    output_seq_len: int = 128,
    device_num: int = 1,
    frequency_hz: float = 1e9,
    verbose: bool = False,
) -> PLENAInferenceResult:
    """Estimate LLM inference performance on PLENA using layer-level model.

    Uses PLENA's native LLaMAModel for accurate cycle-level estimation
    of prefill and decode phases.

    Args:
        model_config: PLENAModelConfig with model architecture
        settings_path: Path to plena_settings.toml
        isa_lib_path: Path to customISA_lib.json
        batch_size: Batch size for inference
        input_seq_len: Input sequence length (prompt length)
        output_seq_len: Output sequence length (tokens to generate)
        device_num: Number of devices (for parallel inference)
        frequency_hz: Clock frequency in Hz (default: 1 GHz)
        verbose: Print detailed execution breakdown

    Returns:
        PLENAInferenceResult with timing metrics

    Example:
        >>> config = PLENAModelConfig(
        ...     hidden_size=4096, num_attention_heads=32, num_kv_heads=8,
        ...     num_hidden_layers=32, intermediate_size=14336, vocab_size=128256,
        ... )
        >>> result = estimate_plena_inference(
        ...     config, "plena_settings.toml", "customISA_lib.json"
        ... )
        >>> print(f"TPS: {result.tps:.2f}")
    """
    llama = _load_plena_llama_model(
        model_config,
        settings_path,
        isa_lib_path,
        batch_size=batch_size,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        device_num=device_num,
        frequency_hz=frequency_hz,
    )

    # Compute performance using PLENA's model
    ttft_s, tps = llama.compute_performance(verbose=verbose)

    prefill_s = llama.compute_prefill_time(verbose=False)
    decode_s = llama.compute_decode_time(output_seq_len, verbose=False)

    return PLENAInferenceResult(
        prefill_ms=prefill_s * 1000.0,
        decode_ms=decode_s * 1000.0,
        ttft_ms=ttft_s * 1000.0,
        tps=tps,
    )


def estimate_plena_inference_from_config(
    model_config: PLENAModelConfig,
    plena_config: "PLENAConfig",
    batch_size: int = 1,
    input_seq_len: int = 2048,
    output_seq_len: int = 128,
    device_num: int = 1,
    verbose: bool = False,
) -> PLENAInferenceResult:
    """Estimate LLM inference performance using PLENAConfig.

    Convenience wrapper that uses PLENAConfig for path management.

    Args:
        model_config: PLENAModelConfig with model architecture
        plena_config: PLENAConfig with paths to config files
        batch_size: Batch size for inference
        input_seq_len: Input sequence length
        output_seq_len: Output sequence length
        device_num: Number of devices
        verbose: Print detailed breakdown

    Returns:
        PLENAInferenceResult with timing metrics
    """
    from ..config_plena import PLENAConfig

    return estimate_plena_inference(
        model_config=model_config,
        settings_path=plena_config.settings_path,
        isa_lib_path=plena_config.isa_lib_path,
        batch_size=batch_size,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        device_num=device_num,
        frequency_hz=plena_config.frequency_hz,
        verbose=verbose,
    )


def trace_hf_model_for_plena(
    hf_model: "nn.Module",
    batch_size: int = 1,
    input_seq_len: int = 2048,
    output_seq_len: int = 128,
    plena_config: Optional["PLENAConfig"] = None,
    device_num: int = 1,
    verbose: bool = False,
) -> PLENAInferenceResult:
    """One-liner for HuggingFace model PLENA estimation.

    Extracts model config from HF model and runs PLENA layer-level estimation.

    Args:
        hf_model: HuggingFace PreTrainedModel (e.g., LlamaForCausalLM)
        batch_size: Batch size for inference
        input_seq_len: Input sequence length
        output_seq_len: Output sequence length
        plena_config: PLENAConfig (auto-detected if None)
        device_num: Number of devices
        verbose: Print detailed breakdown

    Returns:
        PLENAInferenceResult with timing metrics

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> result = trace_hf_model_for_plena(model, input_seq_len=2048)
        >>> print(f"TTFT: {result.ttft_ms:.2f} ms, TPS: {result.tps:.2f}")
    """
    from ..config_plena import PLENAConfig

    # Extract config from HF model
    if hasattr(hf_model, "config"):
        model_config = PLENAModelConfig.from_hf_config(hf_model.config)
    else:
        raise ValueError("HuggingFace model must have a .config attribute")

    # Auto-detect PLENA config if not provided
    if plena_config is None:
        plena_config = PLENAConfig.from_plena_submodule()

    return estimate_plena_inference_from_config(
        model_config=model_config,
        plena_config=plena_config,
        batch_size=batch_size,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        device_num=device_num,
        verbose=verbose,
    )
