"""Build the GEMM work-list from a profiling spec.

MVP: synthesize the canonical transformer weight GEMMs analytically (CPU-only,
deterministic), apply TP sharding and fwd/bwd orientations, dedupe. A separate
trace-based enumerator (needs CUDA) is used by the real sweep; not unit-tested.
"""
from __future__ import annotations

from .spec import ProfilingSpec


def gemm_worklist(spec: ProfilingSpec, token_points: list[int]) -> list[dict]:
    """Return deduped GEMM work items: {M, K, N, dtype, direction, tp}.

    Weight templates per (hidden h, ffn f, heads, kv groups, head_dim d):
      qkv:   K=h,  N=(heads + 2*kv)*d        (column-parallel -> N/tp)
      proj:  K=heads*d, N=h                  (row-parallel    -> K/tp)
      gate/up: K=h, N=f                      (column-parallel -> N/tp)
      down:  K=f, N=h                        (row-parallel    -> K/tp)
    M = token count. fwd uses these; wgrad swaps M into the contraction (M<->K).
    """
    tps = spec.parallelism.get("tensor_parallel", [1])
    items: dict[tuple, dict] = {}
    for h in spec.hidden_sizes:
        for f in spec.ffn_hidden_sizes:
            for heads in spec.num_attention_heads:
                for kv in spec.num_query_groups:
                    for d in spec.head_dims:
                        if heads * d > 4 * h or heads * d < h // 4:
                            continue   # head_dim sanity filter
                        templates = [
                            ("qkv",  h, (heads + 2 * kv) * d, "col"),
                            ("proj", heads * d, h, "row"),
                            ("gate", h, f, "col"),
                            ("up",   h, f, "col"),
                            ("down", f, h, "row"),
                        ]
                        for dtype in spec.dtypes:
                            for tp in tps:
                                for _name, K, N, shard in templates:
                                    Ks, Ns = (K, N // tp) if shard == "col" else (K // tp, N)
                                    if Ks < 1 or Ns < 1:
                                        continue
                                    for M in token_points:
                                        for direction in spec.sweep.get("directions", ["fwd"]):
                                            mm = (M, Ks, Ns) if direction == "fwd" else (Ks, M, Ns)
                                            key = (mm[0], mm[1], mm[2], dtype, direction, tp)
                                            items[key] = {
                                                "M": mm[0], "K": mm[1], "N": mm[2],
                                                "dtype": dtype, "direction": direction, "tp": tp,
                                            }
    return list(items.values())
