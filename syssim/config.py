from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class ExecutionMode(Enum):
    TRAINING = "training"    # Forward + backward, standard shapes
    PREFILL = "prefill"      # Forward only, full sequence length
    DECODE = "decode"        # Forward only, seq_len=1, KV cache read


@dataclass
class NetworkParams:
    """Network hardware parameters for communication simulation.

    Used by the network simulator (syssim.network) to model collective
    communication performance on multi-node clusters.

    Single-node parameters:
        nvlink_bandwidth: NVLink bandwidth per link (bytes/second)
        nvlink_count: Number of NVLinks per GPU pair
        ib_bandwidth: InfiniBand bandwidth per NIC (bytes/second)

    Multi-node parameters:
        num_nodes: Number of nodes in cluster
        gpus_per_node: GPUs per node (e.g., 8 for DGX)

    LogGP parameters:
        loggp_nvlink_L: NVLink latency (seconds)
        loggp_nvlink_o: NVLink CPU overhead per message (seconds)
        loggp_ib_L: InfiniBand latency (seconds)
        loggp_ib_o: InfiniBand CPU overhead per message (seconds)

    Example (single DGX H100 node):
        >>> net = NetworkParams(
        ...     nvlink_bandwidth=25e9,  # 25 GB/s
        ...     nvlink_count=18,        # 18 NVLinks per pair on H100
        ...     loggp_nvlink_L=1e-6,    # 1 μs latency
        ...     loggp_nvlink_o=5e-6,    # 5 μs overhead
        ... )

    Example (4-node DGX cluster):
        >>> net = NetworkParams(
        ...     num_nodes=4,
        ...     gpus_per_node=8,
        ...     nvlink_bandwidth=25e9,
        ...     nvlink_count=18,
        ...     ib_bandwidth=25e9,      # 200 Gb/s IB
        ...     loggp_nvlink_L=1e-6,
        ...     loggp_nvlink_o=5e-6,
        ...     loggp_ib_L=5e-6,        # Higher latency than NVLink
        ...     loggp_ib_o=10e-6,       # Higher overhead than NVLink
        ... )
    """
    # Single-node parameters
    nvlink_bandwidth: float = 25e9  # bytes/second per NVLink
    nvlink_count: int = 12          # NVLinks per GPU pair (DGX A100)
    ib_bandwidth: float = 25e9      # bytes/second (200 Gb/s = 25 GB/s)

    # Multi-node parameters
    num_nodes: int = 1
    gpus_per_node: int = 8

    # LogGP parameters - NVLink (intra-node)
    loggp_nvlink_L: float = 1e-6    # latency (seconds)
    loggp_nvlink_o: float = 5e-6    # CPU overhead (seconds)

    # LogGP parameters - InfiniBand (inter-node)
    loggp_ib_L: float = 5e-6        # latency (seconds)
    loggp_ib_o: float = 10e-6       # CPU overhead (seconds)


class HardwareInfo:
    """Hardware specifications for performance modeling.

    Stores peak throughput rates used by the roofline model to estimate
    operator execution times.

    Args:
        peak_tflops_mm: Peak throughput for matrix multiply operations (TFLOP/s).
                        Used for large GEMM and ATTN operators. IMPORTANT: Must be in
                        TFLOP/s (10^12 FLOP/s), NOT PFLOP/s or GFLOP/s.
                        Example: H100 FP16 tensor unit peak = 1979.0 TFLOP/s
        peak_tflops_math: Peak throughput for vector unit operations (TFLOP/s).
                          Used for all non-GEMM/ATTN operators.
                          Example: H100 FP16 = 989.0 TFLOP/s
        peak_memory_bandwidth_gbps: Peak memory bandwidth (GB/s).
                                    Used for memory-bound roofline calculations.
                                    IMPORTANT: Must be in GB/s (10^9 bytes/s).
                                    Example: H100 = 3350.0 GB/s
        peak_tflops_mm_conservative: Conservative peak for small matrix operations (TFLOP/s).
                                     Used for small GEMM/ATTN ops where launch overhead dominates.
                                     Defaults to peak_tflops_mm if not specified.
                                     Example: H100 conservative = 535.0 TFLOP/s

    Unit System:
        - Peak FLOP rates: TFLOP/s (tera = 10^12)
        - Memory bandwidth: GB/s (giga = 10^9)
        - Internal calculations use nanoseconds (ns)
        - External API returns milliseconds (ms)
        - Conversion: 1 TFLOP/s × 1e12 = 1e12 FLOP/s
                     1 GB/s × 1e9 = 1e9 bytes/s
    """

    def __init__(
        self,
        peak_tflops_mm: float,
        peak_tflops_math: float,
        peak_memory_bandwidth_gbps: float,
        peak_tflops_mm_conservative: float | None = None,
        network: Optional[NetworkParams] = None,
    ):
        self.peak_tflops_mm = peak_tflops_mm
        self.peak_tflops_math = peak_tflops_math
        self.peak_memory_bandwidth_gbps = peak_memory_bandwidth_gbps
        # Backward compatibility: if not specified, use same peak for both
        self.peak_tflops_mm_conservative = (
            peak_tflops_mm_conservative if peak_tflops_mm_conservative is not None else peak_tflops_mm
        )
        # Network parameters (for network simulator)
        self.network = network if network is not None else NetworkParams()

    def get_peak_tflops(self, op_type, dtype: torch.dtype, is_large_op: bool = False) -> float:
        """Select peak FLOP/s based on operator type and size.

        Args:
            op_type: OperatorType (GEMM/ATTN use peak_tflops_mm, others use peak_tflops_math)
            dtype: Data type (currently unused, for future multi-precision support)
            is_large_op: Whether this is a large tensor unit operation (≥512 in all dims)
                        True: use tensor unit peak (peak_tflops_mm)
                        False: use conservative peak (peak_tflops_mm_conservative)

        Returns:
            Peak FLOP/s in TFLOP/s
        """
        from .operator_graph import OperatorType
        if op_type in (OperatorType.GEMM, OperatorType.ATTN):
            return self.peak_tflops_mm if is_large_op else self.peak_tflops_mm_conservative
        else:
            return self.peak_tflops_math

    def get_peak_memory_bandwidth_gbps(self) -> float:
        return self.peak_memory_bandwidth_gbps


def get_hardware_info() -> tuple[HardwareInfo, str]:
    """Auto-detect current hardware and return HardwareInfo.

    Returns:
        Tuple of (HardwareInfo object, hardware name string)

    Raises:
        RuntimeError: If CUDA is not available or hardware cannot be identified.

    Example:
        >>> hw_info, hw_name = get_hardware_info()
        >>> print(f"Detected {hw_name}: {hw_info.peak_tflops_mm} TFLOP/s")
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot auto-detect hardware.")

    device_name = torch.cuda.get_device_name(0).lower()

    # Hardware specifications lookup table
    # Format: (pattern, hw_name, peak_tflops_mm_fp16, peak_tflops_math_fp16, peak_bw_gb_s)
    hw_database = [
        # NVIDIA GH200 (Grace Hopper) - uses H100 GPU specs
        ("gh200", "gh200", 989.0, 989.0, 3350.0),
        ("grace hopper", "gh200", 989.0, 989.0, 3350.0),

        # NVIDIA H100
        ("h100", "h100", 1979.0, 989.0, 3350.0),

        # NVIDIA A100
        ("a100", "a100", 312.0, 156.0, 1935.0),

        # NVIDIA V100
        ("v100", "v100", 125.0, 62.5, 900.0),

        # NVIDIA A40
        ("a40", "a40", 149.0, 74.5, 696.0),

        # NVIDIA RTX 4090
        ("rtx 4090", "rtx4090", 330.0, 165.0, 1008.0),
        ("geforce rtx 4090", "rtx4090", 330.0, 165.0, 1008.0),

        # AMD MI250
        ("mi250", "mi250", 362.0, 181.0, 1600.0),

        # AMD MI300
        ("mi300", "mi300", 653.0, 326.5, 5200.0),

        # Tenstorrent Wormhole N300 (per-chip: 64 Tensix cores @ 1 GHz)
        # BF16 matrix engine 74 TFLOP/s, FP32 SFPU 2 TFLOP/s, 12 GB GDDR6 288 GB/s
        ("wormhole", "tt_wh_n300", 74.0, 2.0, 288.0),
        ("n300", "tt_wh_n300", 74.0, 2.0, 288.0),
    ]

    # Check device name against known patterns
    for pattern, hw_name, peak_mm, peak_math, peak_bw in hw_database:
        if pattern in device_name:
            hw_info = HardwareInfo(
                peak_tflops_mm=peak_mm,
                peak_tflops_math=peak_math,
                peak_memory_bandwidth_gbps=peak_bw,
            )
            return hw_info, hw_name

    # If no match found
    raise RuntimeError(
        f"Unknown hardware: {device_name}. "
        f"Please add hardware specs to get_hardware_info() in config.py"
    )


@dataclass
class SimulatorConfig:
    hw_info: HardwareInfo
    cache_seq_len: int = 0
