"""
Simulate Qwen3-30B-A3B single-GPU training on GH200 using syssim.

Constructs the Qwen3-30B-A3B MoE architecture from its published specs using
the Qwen3MoeForCausalLM model class. Creates synthetic token inputs, traces a
full training step (forward + backward), and reports operator breakdown and
critical path time.

Qwen3-30B-A3B published specs (Mixture of Experts):
  - 48 hidden layers (all sparse MoE)
  - Hidden size: 2048
  - Expert intermediate size: 768 (per expert MLP)
  - Dense intermediate size: 6144
  - Attention heads: 32
  - KV heads: 4 (8x GQA compression)
  - Vocab size: 151936
  - Total experts: 128
  - Active experts per token: 8
  - Total parameters: ~30B
  - Active parameters per token: ~3B

Run:
    srun -N 1 --gpus 1 python examples/huggingface/train_qwen3_moe_single.py
"""

import os
import sys

# Ensure repo root is on path when invoked via srun without pip install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from syssim import SimulatorConfig, get_hardware_info, trace_hf_model_for_training
from syssim.operator_graph import OperatorType

# Qwen3-30B-A3B published architecture dimensions
QWEN3_MOE_CONFIG = dict(
    num_hidden_layers=48,
    hidden_size=2048,
    intermediate_size=6144,
    moe_intermediate_size=768,
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=128,
    vocab_size=151936,
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    hidden_act="silu",
    num_experts=128,
    num_experts_per_tok=8,
    decoder_sparse_step=1,
    norm_topk_prob=True,
    router_aux_loss_coef=0.001,
    attention_bias=False,
    tie_word_embeddings=False,
)

BATCH_SIZE = 1
SEQ_LEN = 2048


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def active_param_estimate(config_dict):
    """Estimate active parameters per token (excluding inactive experts)."""
    hidden = config_dict["hidden_size"]
    moe_intermediate = config_dict["moe_intermediate_size"]
    num_experts_per_tok = config_dict["num_experts_per_tok"]
    num_layers = config_dict["num_hidden_layers"]
    num_heads = config_dict["num_attention_heads"]
    num_kv_heads = config_dict["num_key_value_heads"]
    head_dim = config_dict["head_dim"]
    vocab_size = config_dict["vocab_size"]

    # Per-layer attention: Q + K + V projections + O projection
    attn_params = (num_heads * head_dim * hidden  # Q
                   + num_kv_heads * head_dim * hidden  # K
                   + num_kv_heads * head_dim * hidden  # V
                   + num_heads * head_dim * hidden)  # O

    # Per-layer MoE: only top_k experts active
    # Each expert: gate_proj + up_proj + down_proj (SwiGLU)
    expert_params = 3 * hidden * moe_intermediate
    active_moe_params = num_experts_per_tok * expert_params

    # Router (gate)
    router_params = hidden * config_dict["num_experts"]

    # Per-layer active params
    per_layer = attn_params + active_moe_params + router_params

    # Embedding + LM head
    embed_params = vocab_size * hidden * 2  # input + output embeddings

    return num_layers * per_layer + embed_params


def main():
    # --- Hardware ---
    hw, hw_name = get_hardware_info()
    print(f"Detected hardware: {hw_name}")
    print(f"  Peak MM TFLOP/s : {hw.peak_tflops_mm:.1f}")
    print(f"  Peak BW GB/s    : {hw.peak_memory_bandwidth_gbps:.1f}")
    print()

    # --- Model (meta device — no real tensor allocation) ---
    print("Building Qwen3-30B-A3B MoE architecture (meta device, no memory allocation)...")
    model_cfg = Qwen3MoeConfig(**QWEN3_MOE_CONFIG)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_cfg, torch_dtype=torch.bfloat16)
    model.train()

    n_params = param_count(model)
    n_active = active_param_estimate(QWEN3_MOE_CONFIG)
    print(f"Model: Qwen3-30B-A3B (MoE)")
    print(f"  Layers           : {QWEN3_MOE_CONFIG['num_hidden_layers']}")
    print(f"  Hidden size      : {QWEN3_MOE_CONFIG['hidden_size']}")
    print(f"  Expert FFN size  : {QWEN3_MOE_CONFIG['moe_intermediate_size']}")
    print(f"  Attn heads       : {QWEN3_MOE_CONFIG['num_attention_heads']}")
    print(f"  KV heads         : {QWEN3_MOE_CONFIG['num_key_value_heads']}")
    print(f"  Total experts    : {QWEN3_MOE_CONFIG['num_experts']}")
    print(f"  Active experts   : {QWEN3_MOE_CONFIG['num_experts_per_tok']}")
    print(f"  Total parameters : {n_params / 1e9:.2f}B")
    print(f"  Active parameters: ~{n_active / 1e9:.2f}B (estimate)")
    print()

    # --- Synthetic inputs ---
    print(f"Input: batch={BATCH_SIZE}, seq_len={SEQ_LEN}")
    input_ids = torch.randint(0, QWEN3_MOE_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN))
    inputs = {"input_ids": input_ids, "labels": input_ids.clone()}
    print()

    # --- Trace ---
    sim_cfg = SimulatorConfig(hw_info=hw)
    print("Tracing training step (forward + backward)...")
    graph = trace_hf_model_for_training(model, inputs, sim_cfg)
    print()

    # --- Report ---
    type_counts: dict[OperatorType, int] = {}
    for op in graph.operators.values():
        type_counts[op.op_type] = type_counts.get(op.op_type, 0) + 1

    print("Operator counts by type:")
    for op_type in OperatorType:
        count = type_counts.get(op_type, 0)
        if count:
            print(f"  {op_type.name:<12}: {count}")
    print()

    critical_path_ms = graph.compute_critical_path()
    print(f"Critical path time : {critical_path_ms:.2f} ms")
    print()

    print(graph.summary())


if __name__ == "__main__":
    main()
