"""Load saved LogGP parameters from profiled models.

This module provides utilities for loading LogGP parameters that were saved
by the profiler CLI tool. It supports both:
- Auto-resolution from topology name (e.g., "nvlink" → "data/network_models/nvlink_loggp.json")
- Explicit file paths

Example:
    >>> from syssim.network import load_loggp_params
    >>>
    >>> # Load profiled NVLink parameters
    >>> loggp = load_loggp_params("nvlink")
    >>> print(f"L={loggp.L*1e6:.2f}μs, G={loggp.G*1e9:.2f}ns/byte")
    L=1.52μs, G=0.04ns/byte
    >>>
    >>> # Load from custom path
    >>> loggp = load_loggp_params("data/network_models/custom_nvlink.json")
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

from .loggp import LogGPParams


def load_loggp_params(topology: Union[str, Path]) -> LogGPParams:
    """Load primary LogGP parameters from saved model.

    The primary protocol is the first protocol in the saved model (typically
    for small messages, corresponding to eager protocol).

    Args:
        topology: Either a topology name (e.g., "nvlink") for auto-resolution,
                  or an explicit path to a JSON file

    Returns:
        LogGPParams with primary protocol parameters

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If JSON is malformed or missing required fields

    Example:
        >>> # Auto-resolve from topology name
        >>> loggp = load_loggp_params("nvlink")
        >>> loggp.L
        1.5e-06
        >>>
        >>> # Explicit path
        >>> loggp = load_loggp_params("data/network_models/nvlink_loggp.json")
    """
    # Resolve path
    if isinstance(topology, str) and not topology.endswith(".json"):
        # Auto-resolve from topology name
        project_root = Path(__file__).parent.parent.parent
        path = project_root / "data" / "network_models" / f"{topology}_loggp.json"
    else:
        path = Path(topology)

    # Check existence
    if not path.exists():
        raise FileNotFoundError(
            f"LogGP model not found: {path}\n"
            f"Run profiler to generate: torchrun --nproc_per_node=2 -m syssim.network.profiler --topology {topology}"
        )

    # Load JSON
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {e}")

    # Extract primary protocol
    if "primary" not in data:
        raise ValueError(f"Missing 'primary' field in {path}")

    primary = data["primary"]

    # Validate required fields
    required_fields = ["L", "o", "G"]
    for field in required_fields:
        if field not in primary:
            raise ValueError(f"Missing '{field}' in primary protocol in {path}")

    # Create LogGPParams (g is optional, defaults to 0.0)
    return LogGPParams(
        L=primary["L"],
        o=primary["o"],
        G=primary["G"],
        g=primary.get("g", 0.0),  # Backward compatibility
    )


def load_all_protocols(topology: Union[str, Path]) -> List[Tuple[Tuple[int, int], LogGPParams]]:
    """Load all protocol ranges from saved model.

    This is useful for size-dependent simulation where different message sizes
    may use different protocols (eager vs rendezvous).

    Args:
        topology: Either a topology name or explicit path to JSON file

    Returns:
        List of ((min_size, max_size), LogGPParams) tuples, one per protocol

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If JSON is malformed

    Example:
        >>> protocols = load_all_protocols("nvlink")
        >>> for (min_size, max_size), params in protocols:
        ...     print(f"{min_size}-{max_size} bytes: G={params.G*1e9:.2f}ns/byte")
        1-12288 bytes: G=0.04ns/byte
        12289-65536 bytes: G=0.04ns/byte
    """
    # Resolve path (same logic as load_loggp_params)
    if isinstance(topology, str) and not topology.endswith(".json"):
        project_root = Path(__file__).parent.parent.parent
        path = project_root / "data" / "network_models" / f"{topology}_loggp.json"
    else:
        path = Path(topology)

    if not path.exists():
        raise FileNotFoundError(
            f"LogGP model not found: {path}\n"
            f"Run profiler to generate: torchrun --nproc_per_node=2 -m syssim.network.profiler --topology {topology}"
        )

    # Load JSON
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {e}")

    # Extract all protocols
    if "protocols" not in data:
        raise ValueError(f"Missing 'protocols' field in {path}")

    protocols = data["protocols"]

    # Convert to list of ((min_size, max_size), LogGPParams)
    result = []
    for protocol in protocols:
        # Validate fields
        if "size_range" not in protocol:
            raise ValueError(f"Missing 'size_range' in protocol: {protocol}")

        size_range = protocol["size_range"]
        if not isinstance(size_range, list) or len(size_range) != 2:
            raise ValueError(f"Invalid size_range format: {size_range}")

        min_size, max_size = size_range

        # Create LogGPParams
        params = LogGPParams(
            L=protocol.get("L", 0.0), o=protocol.get("o", 0.0), G=protocol.get("G", 0.0), g=protocol.get("g", 0.0)
        )

        result.append(((min_size, max_size), params))

    return result


def get_protocol_for_size(protocols: List[Tuple[Tuple[int, int], LogGPParams]], size: int) -> LogGPParams:
    """Get appropriate LogGP parameters for a given message size.

    Args:
        protocols: List of protocol ranges from load_all_protocols()
        size: Message size in bytes

    Returns:
        LogGPParams for the protocol covering this size

    Raises:
        ValueError: If no protocol covers the requested size

    Example:
        >>> protocols = load_all_protocols("nvlink")
        >>> params_small = get_protocol_for_size(protocols, 1024)  # Eager
        >>> params_large = get_protocol_for_size(protocols, 32768)  # Rendezvous
    """
    for (min_size, max_size), params in protocols:
        if min_size <= size <= max_size:
            return params

    # No matching protocol
    raise ValueError(
        f"No protocol found for size {size} bytes. "
        f"Available ranges: {[(min_s, max_s) for (min_s, max_s), _ in protocols]}"
    )


def is_hierarchical_model(topology: Union[str, Path]) -> bool:
    """Check if saved model is hierarchical or single-layer.

    Args:
        topology: Either a topology name or explicit path to JSON file

    Returns:
        True if hierarchical format, False if single-layer

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If JSON is malformed

    Example:
        >>> if is_hierarchical_model("perlmutter"):
        ...     params = load_hierarchical_loggp("perlmutter")
        ... else:
        ...     params = load_loggp_params("perlmutter")
    """
    # Resolve path
    if isinstance(topology, str) and not topology.endswith(".json"):
        project_root = Path(__file__).parent.parent.parent
        path = project_root / "data" / "network_models" / f"{topology}_loggp.json"
    else:
        path = Path(topology)

    if not path.exists():
        raise FileNotFoundError(f"LogGP model not found: {path}")

    # Load JSON
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {e}")

    # Check if hierarchical (has "layers" dict) or single-layer (has "topology" str)
    return "layers" in data and isinstance(data["layers"], dict)


def load_hierarchical_loggp(topology: Union[str, Path]) -> Dict[str, LogGPParams]:
    """Load hierarchical LogGP parameters from saved model.

    Args:
        topology: Either a topology name (e.g., "perlmutter") or path to hierarchical JSON

    Returns:
        Dict mapping layer name to LogGPParams

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If JSON is malformed or not hierarchical format

    Example:
        >>> params = load_hierarchical_loggp("perlmutter")
        >>> params["intra_node_nvlink"].G
        1.59e-11
        >>> params["inter_node_ib"].G
        7.78e-10
    """
    # Resolve path
    if isinstance(topology, str) and not topology.endswith(".json"):
        project_root = Path(__file__).parent.parent.parent
        path = project_root / "data" / "network_models" / f"{topology}_loggp.json"
    else:
        path = Path(topology)

    if not path.exists():
        raise FileNotFoundError(
            f"LogGP model not found: {path}\n"
            f"Run profiler to generate: see examples/configs/ for hierarchy config examples"
        )

    # Load JSON
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {e}")

    # Check if hierarchical
    if "layers" not in data or not isinstance(data["layers"], dict):
        raise ValueError(
            "Expected hierarchical format (with 'layers' dict), got single-layer. Use load_loggp_params() instead."
        )

    # Extract parameters for each layer
    result = {}
    for layer_name, layer_data in data["layers"].items():
        if "primary" not in layer_data:
            raise ValueError(f"Missing 'primary' field in layer '{layer_name}'")

        primary = layer_data["primary"]

        # Validate required fields
        required_fields = ["L", "o", "G"]
        for field in required_fields:
            if field not in primary:
                raise ValueError(f"Missing '{field}' in primary protocol of layer '{layer_name}'")

        result[layer_name] = LogGPParams(
            L=primary["L"],
            o=primary["o"],
            G=primary["G"],
            g=primary.get("g", 0.0),  # Backward compatibility
        )

    return result


def get_layer_params(
    hierarchical_params: Dict[str, LogGPParams],
    src_rank: int,
    dst_rank: int,
    topology_map: Dict[str, Callable[[int, int], bool]],
) -> LogGPParams:
    """Get LogGP parameters for communication between src and dst ranks.

    This function uses a topology map (predicates) to determine which layer
    handles communication between two ranks.

    Args:
        hierarchical_params: Dict from load_hierarchical_loggp()
        src_rank: Source rank
        dst_rank: Destination rank
        topology_map: Dict mapping layer name to predicate function that returns
            True if this layer handles (src, dst) communication

    Returns:
        LogGPParams for the appropriate layer

    Raises:
        ValueError: If no layer found for the given rank pair

    Example:
        >>> params = load_hierarchical_loggp("perlmutter")
        >>> # 4 GPUs per node
        >>> topology_map = {
        ...     "intra_node_nvlink": lambda s, d: s // 4 == d // 4,
        ...     "inter_node_ib": lambda s, d: s // 4 != d // 4
        ... }
        >>> # Ranks 0 and 1 are on same node (node 0)
        >>> loggp_intra = get_layer_params(params, 0, 1, topology_map)
        >>> # Ranks 0 and 4 are on different nodes
        >>> loggp_inter = get_layer_params(params, 0, 4, topology_map)
    """
    for layer_name, predicate in topology_map.items():
        if predicate(src_rank, dst_rank):
            if layer_name not in hierarchical_params:
                raise ValueError(
                    f"Layer '{layer_name}' matched by predicate but not found in hierarchical params. "
                    f"Available layers: {list(hierarchical_params.keys())}"
                )
            return hierarchical_params[layer_name]

    # No matching layer
    raise ValueError(
        f"No layer found for ranks {src_rank} -> {dst_rank}. Available layers: {list(hierarchical_params.keys())}"
    )
