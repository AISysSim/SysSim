__version__ = "0.1.0"

from .api import (
    set_efficiency_model_dir,
    trace_model_for_inference,
    trace_model_for_plena,
    trace_model_for_training,
)
from .config import (
    ExecutionMode,
    HardwareInfo,
    NetworkParams,
    SimulatorConfig,
    get_hardware_info,
)

# PLENA integration (Op-Level via syssim.config_plena)
from .config_plena import PLENAConfig, is_plena_hardware

# Diffusers integration (syssim.integrations.diffusers)
from .integrations.diffusers import trace_diffusers_model_for_inference

# Hugging Face integration (syssim.integrations.huggingface)
from .integrations.huggingface import (
    trace_hf_model_for_training,
    trace_hf_training_step,
)

# Network simulator (syssim.network)
from .network import (
    # Topologies
    FullyConnectedTopology,
    HierarchicalTopology,
    # Core types
    LogGPParams,
    NVLinkMeshTopology,
    Op,
    Resource,
    RingTopology,
    SimulationResult,
    Step,
    SwitchTopology,
    Topology,
    allgather,
    # Collectives
    allreduce,
    alltoall,
    broadcast,
    build_dag,
    gather,
    reduce,
    reduce_scatter,
    scatter,
    # Simulation
    simulate,
)
from .operator_graph import OperatorGraph, OperatorNode, OperatorType, TensorMeta
