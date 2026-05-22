"""PLENA performance estimation example: trace models and estimate cycle-level timing.

This example demonstrates the Op-Level PLENA integration (Approach A) which maps
traced PyTorch operators to PLENA instructions for cycle-accurate estimation.

Requirements:
    - CUDA-capable device
    - PLENA_Simulator submodule initialized: `git submodule update --init`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from syssim import trace_model_for_plena, PLENAConfig


class TransformerBlock(nn.Module):
    """A single transformer block with self-attention and FFN.

    Operators exercised:
    - GEMM: Q/K/V/O projections, FFN up/down projections
    - ATTN: scaled_dot_product_attention
    - MATH: LayerNorm, SiLU activation, residual adds
    """

    def __init__(self, hidden_size=512, num_heads=8, ffn_size=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Self-attention projections (GEMM)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Feed-forward network (GEMM)
        self.up_proj = nn.Linear(hidden_size, ffn_size)
        self.down_proj = nn.Linear(ffn_size, hidden_size)
        self.act = nn.SiLU()

        # Layer norms (MATH)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        B, S, H = x.shape

        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (ATTN)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, H)

        x = self.o_proj(attn) + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.down_proj(self.act(self.up_proj(x))) + residual

        return x


def print_table(graph, title):
    """Print operator nodes in a tabular format."""
    graph.compute_critical_path()

    w_name = max(len(op.name) for op in graph.operators.values())
    w_name = max(w_name, 4)
    header = (
        f"{'Name':<{w_name}}  {'Type':<10}  {'Time (ms)':>12}  "
        f"{'Start (ms)':>12}  {'Finish (ms)':>12}"
    )
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f" {title}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for name in graph.topological_sort():
        op = graph.operators[name]
        print(
            f"{op.name:<{w_name}}  {op.op_type.value:<10}  "
            f"{op.estimated_time_ms:>12.6f}  {op.earliest_start:>12.6f}  "
            f"{op.earliest_finish:>12.6f}"
        )

    print(sep)
    total = sum(op.estimated_time_ms for op in graph.operators.values())
    cp = max(op.earliest_finish for op in graph.operators.values())
    print(f"Total time: {total:.6f} ms | Critical path: {cp:.6f} ms | Ops: {len(graph)}")
    print()


def print_breakdown(graph, title):
    """Print timing breakdown by operator type."""
    op_times = defaultdict(float)
    op_counts = defaultdict(int)
    for op in graph.operators.values():
        op_times[op.op_type.name] += op.estimated_time_ms
        op_counts[op.op_type.name] += 1

    print(f"\n{title}")
    print("-" * 40)
    for op_type, time in sorted(op_times.items(), key=lambda x: -x[1]):
        count = op_counts[op_type]
        print(f"  {op_type:8s}: {time:10.6f} ms  ({count} ops)")
    print()


def example_simple_mlp():
    """Example 1: Simple MLP model."""
    print("\n" + "=" * 70)
    print(" Example 1: Simple MLP")
    print("=" * 70)

    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
    )

    inputs = torch.randn(32, 256).cuda()
    config = PLENAConfig.from_plena_submodule()

    graph = trace_model_for_plena(model, inputs, config, mode="prefill")

    print(f"\nModel: Linear(256->512) -> ReLU -> Linear(512->128)")
    print(f"Input: batch=32, dim=256")
    print(f"\nCritical path: {graph.compute_critical_path():.6f} ms")
    print(graph.summary())


def example_transformer_block():
    """Example 2: Transformer block."""
    print("\n" + "=" * 70)
    print(" Example 2: 4-Layer Transformer Block")
    print("=" * 70)

    model = nn.Sequential(
        *[TransformerBlock(hidden_size=512, num_heads=8, ffn_size=2048) for _ in range(4)]
    )

    inputs = torch.randn(4, 512, 512).cuda()
    config = PLENAConfig.from_plena_submodule()

    graph = trace_model_for_plena(model, inputs, config, mode="prefill")

    print(f"\nModel: 4x TransformerBlock (hidden=512, heads=8, ffn=2048)")
    print(f"Input: batch=4, seq_len=512, hidden=512")
    print(f"\nCritical path: {graph.compute_critical_path():.6f} ms")

    print_breakdown(graph, "Breakdown by Operator Type:")
    print(graph.summary())

    # Show top operators
    print("Top 5 Operators by Time:")
    ops_sorted = sorted(graph.operators.values(), key=lambda x: -x.estimated_time_ms)
    for i, op in enumerate(ops_sorted[:5]):
        print(f"  {i+1}. {op.op_type.name:8s} | {op.estimated_time_ms:.6f} ms | {op.name[:40]}")


def example_prefill_vs_decode():
    """Example 3: Compare prefill vs decode modes."""
    print("\n" + "=" * 70)
    print(" Example 3: Prefill vs Decode Comparison")
    print("=" * 70)

    model = TransformerBlock(hidden_size=1024, num_heads=16, ffn_size=4096)
    config = PLENAConfig.from_plena_submodule()

    # Prefill: full sequence processing
    prefill_input = torch.randn(1, 2048, 1024).cuda()
    graph_prefill = trace_model_for_plena(model, prefill_input, config, mode="prefill")

    # Decode: single token generation
    decode_input = torch.randn(1, 1, 1024).cuda()
    graph_decode = trace_model_for_plena(model, decode_input, config, mode="decode")

    prefill_time = graph_prefill.compute_critical_path()
    decode_time = graph_decode.compute_critical_path()

    print(f"\nModel: TransformerBlock (hidden=1024, heads=16, ffn=4096)")
    print()
    print(f"[PREFILL MODE] Input: batch=1, seq_len=2048, hidden=1024")
    print(f"  Critical path: {prefill_time:.6f} ms")
    print(f"  Operators:     {len(graph_prefill)}")

    print(f"\n[DECODE MODE] Input: batch=1, seq_len=1, hidden=1024")
    print(f"  Critical path: {decode_time:.6f} ms")
    print(f"  Operators:     {len(graph_decode)}")

    print(f"\n[COMPARISON]")
    print(f"  Prefill / Decode ratio: {prefill_time / decode_time:.1f}x")
    print(f"  (Prefill processes 2048 tokens, Decode processes 1 token)")

    print_breakdown(graph_prefill, "Prefill Breakdown:")
    print_breakdown(graph_decode, "Decode Breakdown:")


def example_detailed_trace():
    """Example 4: Detailed operator trace table."""
    print("\n" + "=" * 70)
    print(" Example 4: Detailed Operator Trace")
    print("=" * 70)

    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Linear(512, 256),
    )

    inputs = torch.randn(8, 64, 256).cuda()
    config = PLENAConfig.from_plena_submodule()

    graph = trace_model_for_plena(model, inputs, config, mode="prefill")

    print(f"\nModel: Linear -> LayerNorm -> SiLU -> Linear")
    print(f"Input: batch=8, seq_len=64, dim=256")

    print_table(graph, "PLENA Cycle-Level Estimation")


def main():
    if not torch.cuda.is_available():
        print("ERROR: PLENA tracing requires a CUDA-capable device.")
        return

    # Check PLENA submodule
    try:
        config = PLENAConfig.from_plena_submodule()
        print(f"PLENA Config loaded:")
        print(f"  Settings: {config.settings_path}")
        print(f"  ISA Lib:  {config.isa_lib_path}")
        print(f"  Frequency: {config.frequency_hz / 1e9:.1f} GHz")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please initialize the PLENA submodule: git submodule update --init")
        return

    # Run examples
    example_simple_mlp()
    example_transformer_block()
    example_prefill_vs_decode()
    example_detailed_trace()

    print("\n" + "=" * 70)
    print(" All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
