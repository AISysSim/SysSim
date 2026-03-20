__version__ = "0.1.0"

from .config import ExecutionMode, HardwareInfo, SimulatorConfig, NetworkParams, get_hardware_info
from .operator_graph import OperatorType, OperatorNode, OperatorGraph, TensorMeta
from .api import trace_model_for_training, trace_model_for_inference, set_efficiency_model_dir

# Hugging Face integration (syssim.integrations.huggingface)
from .integrations.huggingface import (
    trace_hf_model_for_training,
    trace_hf_training_step,
)

# Network simulator (syssim.network)
from .network import (
    # Core types
    LogGPParams, Topology, Resource, Op, Step, SimulationResult,
    # Topologies
    FullyConnectedTopology, RingTopology, SwitchTopology,
    NVLinkMeshTopology, HierarchicalTopology,
    # Collectives
    allreduce, broadcast, reduce, reduce_scatter, allgather,
    alltoall, scatter, gather,
    # Simulation
    simulate, build_dag,
)
