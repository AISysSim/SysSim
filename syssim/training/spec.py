"""Config dataclasses and YAML loaders for the high-level training simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    """Model architecture. Either Megatron fields OR `huggingface` discriminator.

    For Megatron fields branch: all the architecture fields are required.
    For HuggingFace branch: only `huggingface` is required; `overrides` is optional.
    Validation that exactly one branch is populated happens in `ModelConfig.from_dict`.
    """
    # Megatron fields (all optional at dataclass level; loader enforces presence)
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_query_groups: Optional[int] = None
    kv_channels: Optional[int] = None   # head_dim; defaults to hidden_size // heads when unset
    ffn_hidden_size: Optional[int] = None
    seq_length: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    vocab_size: Optional[int] = None
    swiglu: bool = True
    rope: bool = True
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    rms_norm_eps: float = 1e-6

    # HuggingFace branch
    huggingface: Optional[str] = None
    overrides: dict = field(default_factory=dict)


@dataclass
class ParallelismConfig:
    """Parallelism dimensions. Short kwargs (`tp`, `dp`, `sp`, `cp`, `pp`, `vpp`) map to Megatron names."""

    tensor_model_parallel_size: int = 1
    data_parallel_size: int = 1
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None

    def __init__(
        self,
        *,
        tp: Optional[int] = None,
        dp: Optional[int] = None,
        sp: Optional[bool] = None,
        cp: Optional[int] = None,
        pp: Optional[int] = None,
        vpp: Optional[int] = None,
        tensor_model_parallel_size: Optional[int] = None,
        data_parallel_size: Optional[int] = None,
        sequence_parallel: Optional[bool] = None,
        context_parallel_size: Optional[int] = None,
        pipeline_model_parallel_size: Optional[int] = None,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
    ):
        # Resolve short or long names; long name wins if both given
        self.tensor_model_parallel_size = (
            tensor_model_parallel_size if tensor_model_parallel_size is not None
            else (tp if tp is not None else 1)
        )
        self.data_parallel_size = (
            data_parallel_size if data_parallel_size is not None
            else (dp if dp is not None else 1)
        )
        self.sequence_parallel = (
            sequence_parallel if sequence_parallel is not None
            else (sp if sp is not None else False)
        )
        self.context_parallel_size = (
            context_parallel_size if context_parallel_size is not None
            else (cp if cp is not None else 1)
        )
        self.pipeline_model_parallel_size = (
            pipeline_model_parallel_size if pipeline_model_parallel_size is not None
            else (pp if pp is not None else 1)
        )
        self.virtual_pipeline_model_parallel_size = (
            virtual_pipeline_model_parallel_size
            if virtual_pipeline_model_parallel_size is not None
            else vpp
        )

        for name, val in (
            ("tensor_model_parallel_size", self.tensor_model_parallel_size),
            ("data_parallel_size", self.data_parallel_size),
            ("context_parallel_size", self.context_parallel_size),
            ("pipeline_model_parallel_size", self.pipeline_model_parallel_size),
        ):
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")
        if (self.virtual_pipeline_model_parallel_size is not None
                and self.virtual_pipeline_model_parallel_size < 1):
            raise ValueError(
                f"virtual_pipeline_model_parallel_size must be >= 1, "
                f"got {self.virtual_pipeline_model_parallel_size}"
            )

    @property
    def world_size(self) -> int:
        return (
            self.tensor_model_parallel_size
            * self.data_parallel_size
            * self.context_parallel_size
            * self.pipeline_model_parallel_size
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters. Short kwargs map to Megatron names."""

    micro_batch_size: int = 0
    global_batch_size: int = 0
    fp16: bool = False
    bf16: bool = True
    fp8: bool = False
    recompute_granularity: Optional[str] = None
    use_distributed_optimizer: bool = False

    def __init__(
        self,
        *,
        micro_batch: Optional[int] = None,
        global_batch: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        global_batch_size: Optional[int] = None,
        dtype: Optional[str] = None,
        fp16: Optional[bool] = None,
        bf16: Optional[bool] = None,
        fp8: Optional[bool] = None,
        recompute: Optional[str] = None,
        recompute_granularity: Optional[str] = None,
        use_distributed_optimizer: bool = False,
    ):
        self.micro_batch_size = (
            micro_batch_size if micro_batch_size is not None
            else (micro_batch if micro_batch is not None else 0)
        )
        self.global_batch_size = (
            global_batch_size if global_batch_size is not None
            else (global_batch if global_batch is not None else 0)
        )
        if self.micro_batch_size < 1 or self.global_batch_size < 1:
            raise ValueError("micro_batch_size and global_batch_size are required and must be >= 1")

        if dtype is not None:
            if fp16 is not None or bf16 is not None or fp8 is not None:
                raise ValueError("pass exactly one of `dtype=` or the flag kwargs")
            if dtype not in ("fp16", "bf16", "fp8"):
                raise ValueError(f"dtype must be 'fp16'/'bf16'/'fp8', got {dtype!r}")
            self.fp16 = dtype == "fp16"
            self.bf16 = dtype == "bf16"
            self.fp8 = dtype == "fp8"
        else:
            self.fp16 = fp16 if fp16 is not None else False
            self.bf16 = bf16 if bf16 is not None else True
            self.fp8 = fp8 if fp8 is not None else False
        active = sum(int(x) for x in (self.fp16, self.bf16, self.fp8))
        if active != 1:
            raise ValueError(f"exactly one of fp16/bf16/fp8 must be true; got {active}")

        rg = recompute_granularity if recompute_granularity is not None else recompute
        if rg not in (None, "selective", "full"):
            raise ValueError(
                f"recompute_granularity must be null/'selective'/'full', got {rg!r}"
            )
        self.recompute_granularity = rg
        self.use_distributed_optimizer = use_distributed_optimizer


@dataclass
class HardwareConfig:
    """Self-describing hardware spec — compute + topology.

    Compute fields (required):
        peak_tflops_mm, peak_tflops_math, peak_memory_bandwidth_GBps, gpus_per_node
    Network params (required when derived num_nodes > 1):
        inter_node_bandwidth_GBps
    Optional:
        peak_tflops_mm_fp8, peak_tflops_mm_fp4, inter_node_latency_us
        gpu_memory_GB — per-GPU HBM capacity; enables OOM detection when set.
        topology — dict with intra-node + inter-node network parameters
                   (see syssim.network.build_topology_from_config).
    """
    peak_tflops_mm: float
    peak_tflops_math: float
    peak_memory_bandwidth_GBps: float
    gpus_per_node: int

    peak_tflops_mm_fp8: Optional[float] = None
    peak_tflops_mm_fp4: Optional[float] = None
    sfu_peak: Optional[float] = None

    gpu_memory_GB: Optional[float] = None

    inter_node_bandwidth_GBps: Optional[float] = None
    inter_node_latency_us: float = 0.0
    topology: Optional[dict] = None
    estimator: Optional[Any] = None   # custom per-op Estimator (Python-only; not from YAML)
    calibrated_model: Optional[str] = None  # path to a calibrated TreeEstimator model dir (.lgb files)

    def __post_init__(self) -> None:
        for name, val in (
            ("peak_tflops_mm", self.peak_tflops_mm),
            ("peak_tflops_math", self.peak_tflops_math),
            ("peak_memory_bandwidth_GBps", self.peak_memory_bandwidth_GBps),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        if self.gpus_per_node < 1:
            raise ValueError(f"gpus_per_node must be >= 1, got {self.gpus_per_node}")


_MODEL_MEGATRON_FIELDS = frozenset({
    "num_layers", "hidden_size", "num_attention_heads", "num_query_groups", "kv_channels",
    "ffn_hidden_size", "seq_length", "max_position_embeddings", "vocab_size",
    "swiglu", "rope", "rope_theta", "tie_word_embeddings", "rms_norm_eps",
})
_MODEL_HF_FIELDS = frozenset({"huggingface", "overrides"})
_MODEL_ALLOWED = _MODEL_MEGATRON_FIELDS | _MODEL_HF_FIELDS


def load_model_yaml(path: str) -> ModelConfig:
    """Load and validate a model YAML.

    Raises ValueError on any disallowed top-level key, or if neither/both
    of (Megatron fields, huggingface) branches are populated.
    """
    import yaml as _yaml
    with open(path) as f:
        data = _yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"model YAML must be a mapping at top level: {path}")
    disallowed = set(data) - _MODEL_ALLOWED
    if disallowed:
        names = ", ".join(sorted(disallowed))
        raise ValueError(
            f"model YAML has disallowed key(s): {names}. "
            f"Per-run state (parallelism, training, hardware, network, output) "
            f"goes via Python kwargs or CLI flags."
        )
    has_megatron = bool(set(data) & _MODEL_MEGATRON_FIELDS)
    has_hf = "huggingface" in data
    if has_megatron and has_hf:
        raise ValueError(
            "model YAML has both `huggingface` and architecture fields; choose one"
        )
    if not has_megatron and not has_hf:
        raise ValueError(
            "model YAML must set either `huggingface` or architecture fields"
        )
    if "overrides" in data and not has_hf:
        raise ValueError("`overrides` is only valid with `huggingface`")
    return ModelConfig(**data)


_HARDWARE_ALLOWED = frozenset({
    "peak_tflops_mm", "peak_tflops_math", "peak_memory_bandwidth_GBps",
    "peak_tflops_mm_fp8", "peak_tflops_mm_fp4", "sfu_peak",
    "gpus_per_node",
    "gpu_memory_GB",
    "inter_node_bandwidth_GBps", "inter_node_latency_us",
    "topology",
    "calibrated_model",
})


def load_hardware_yaml(path: str) -> HardwareConfig:
    """Load and validate a hardware YAML.

    Raises ValueError on disallowed keys or required-field constraints.
    Cross-field check (`inter_node_bandwidth_GBps` required when `num_nodes > 1`)
    happens at simulate-time when world size is known.
    """
    import yaml as _yaml
    with open(path) as f:
        data = _yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"hardware YAML must be a mapping at top level: {path}")
    disallowed = set(data) - _HARDWARE_ALLOWED
    if disallowed:
        names = ", ".join(sorted(disallowed))
        raise ValueError(
            f"hardware YAML has disallowed key(s): {names}. "
            f"Per-run state (parallelism, training, num_nodes) goes via kwargs/flags."
        )
    return HardwareConfig(**data)


def derive_num_nodes(parallelism: "ParallelismConfig", hardware: "HardwareConfig") -> int:
    """Compute num_nodes from world_size and gpus_per_node. Raises on invalid combinations."""
    world_size = parallelism.world_size
    gpn = hardware.gpus_per_node
    if world_size <= gpn:
        return 1  # fits on a single node (possibly partial); all collectives are intra-node
    if world_size % gpn != 0:
        raise ValueError(
            f"world_size ({world_size}) is not divisible by gpus_per_node ({gpn})"
        )
    num_nodes = world_size // gpn
    if num_nodes > 1 and hardware.topology is None and hardware.inter_node_bandwidth_GBps is None:
        raise ValueError(
            f"inter-node network spec required when num_nodes > 1 (derived num_nodes = {num_nodes}): "
            f"provide a `topology` block (preferred; it specifies inter-node bandwidth) or the "
            f"legacy inter_node_bandwidth_GBps"
        )
    return num_nodes


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply dotted-key overrides to a config dict.

    `data` is a dict of section -> kwargs (e.g. {"parallelism": {"tp": 1}}).
    Each override is `section.field=value`. Values are parsed as YAML scalars.
    """
    import copy
    import yaml as _yaml
    out = copy.deepcopy(data)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be section.field=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        parts = key.split(".")
        if len(parts) < 2:
            raise ValueError(f"override key must contain a section, got {key!r}")
        cursor = out
        for p in parts[:-1]:
            if not isinstance(cursor, dict) or p not in cursor:
                raise KeyError(key)
            cursor = cursor[p]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise KeyError(key)
        cursor[leaf] = _yaml.safe_load(raw)
    return out
