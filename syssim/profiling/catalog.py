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


def bmm_worklist(spec: ProfilingSpec, seq_points: list[int]) -> list[dict]:
    """Batched attention score matmuls for the EXPLICIT attention path: QK^T, A@V and
    their backward orientations, batched over (micro_batch * heads/tp). These are O(S^2)
    memory-bound and shaped unlike the 2D weight GEMMs, so the gemm family must profile
    them directly or it extrapolates (over-predicts ~2x). Items carry `batch`>1, which
    routes them to measure_bmm; the gemm featurizer already exposes batched_dim."""
    tps = spec.parallelism.get("tensor_parallel", [1])
    items: dict[tuple, dict] = {}
    for heads in spec.num_attention_heads:
        for d in spec.head_dims:
            for dtype in spec.dtypes:
                for tp in tps:
                    hq = heads // tp
                    if hq < 1:
                        continue
                    batch = hq                      # micro_batch=1 (the validation matrix)
                    for S in seq_points:
                        if batch * S * S * 2 > 8_000_000_000:    # cap working set (~16GB live)
                            continue
                        # (M,K,N): QK^T=(S,d,S); A@V=(S,S,d); backward=(d,S,S)
                        for (M, K, N) in ((S, d, S), (S, S, d), (d, S, S)):
                            key = (batch, M, K, N, dtype, tp)
                            items[key] = {"batch": batch, "M": M, "K": K, "N": N,
                                          "dtype": dtype, "tp": tp, "direction": "fwd"}
    return list(items.values())


def attention_worklist(spec: ProfilingSpec, seq_points: list[int]) -> list[dict]:
    """Deduped attention items {family,B,S,H_q,H_kv,D,causal,dtype}; H_kv<=H_q (GQA)."""
    causals = spec.sweep.get("causal", [True, False])
    items: dict[tuple, dict] = {}
    for heads in spec.num_attention_heads:
        for kv in spec.num_query_groups:
            if kv > heads:
                continue
            for d in spec.head_dims:
                for dtype in spec.dtypes:
                    for B in (1, 2):
                        for S in seq_points:
                            for causal in causals:
                                key = (B, S, heads, kv, d, causal, dtype)
                                items[key] = {"family": "attention", "B": B, "S": S,
                                              "H_q": heads, "H_kv": kv, "D": d,
                                              "causal": bool(causal), "dtype": dtype}
    return list(items.values())


def norm_worklist(spec: ProfilingSpec, token_points: list[int]) -> list[dict]:
    """Deduped normalization items {family,tokens,hidden,op_subtype,dtype}."""
    items: dict[tuple, dict] = {}
    for hidden in spec.hidden_sizes:
        for dtype in spec.dtypes:
            for sub in ("layernorm", "rmsnorm"):
                for tokens in token_points:
                    key = (tokens, hidden, sub, dtype)
                    items[key] = {"family": "normalization", "tokens": tokens,
                                  "hidden": hidden, "op_subtype": sub, "dtype": dtype}
    return list(items.values())


def elementwise_worklist(spec: ProfilingSpec, token_points: list[int]) -> list[dict]:
    """Deduped elementwise/activation items {family,total_elements,op_subtype,dtype}.

    Covers activations (gelu/silu) and pure mem ops (add/mul) over [tokens,hidden], AND
    copy_/masked_fill — which the explicit-attention path runs over the O(S^2) attention
    scores. Those score-sized copies/masks were previously uncalibrated (the residual
    extrapolated -> ~2x under), so profile them at the real [heads/tp, S, S] sizes too."""
    items: dict[tuple, dict] = {}
    subtypes = ("gelu", "silu", "add", "mul", "copy_", "masked_fill", "masked_fill_")
    for hidden in spec.hidden_sizes:
        for dtype in spec.dtypes:
            for sub in subtypes:
                for tokens in token_points:
                    total = tokens * hidden
                    key = (total, sub, dtype)
                    items[key] = {"family": "elementwise", "total_elements": total,
                                  "op_subtype": sub, "dtype": dtype}
    # copy_/masked_fill over attention-scores-sized tensors [b=1, heads/tp, S, S].
    tps = spec.parallelism.get("tensor_parallel", [1])
    for heads in spec.num_attention_heads:
        for dtype in spec.dtypes:
            for tp in tps:
                H = max(1, heads // tp)
                for S in token_points:
                    total = H * S * S
                    if total * 2 > 8_000_000_000:
                        continue
                    for sub in ("copy_", "masked_fill", "masked_fill_"):
                        key = (total, sub, dtype)
                        items[key] = {"family": "elementwise", "total_elements": total,
                                      "op_subtype": sub, "dtype": dtype}
    return list(items.values())


def reduction_worklist(spec: ProfilingSpec, seq_points: list[int]) -> list[dict]:
    """Deduped reduction/softmax items over attention scores {family,B,H,S,op_subtype,dtype}.

    H is the PER-RANK head count (heads/tp): the explicit-attention softmax runs on the
    TP-sharded scores [b, heads/tp, S, S], so profiling at full head count made the
    in-step softmax out-of-distribution (-> residual extrapolated, ~2x under). Cap the
    working set by tensor bytes (not S) so large-S / low-H shapes are still covered."""
    tps = spec.parallelism.get("tensor_parallel", [1])
    items: dict[tuple, dict] = {}
    for heads in spec.num_attention_heads:
        for dtype in spec.dtypes:
            for tp in tps:
                H = max(1, heads // tp)
                for B in (1, 2):
                    for S in seq_points:
                        if B * H * S * S * 2 > 8_000_000_000:    # cap scores tensor ~8GB
                            continue
                        key = (B, H, S, "_softmax", dtype)
                        items[key] = {"family": "reduction", "B": B, "H": H, "S": S,
                                      "op_subtype": "_softmax", "dtype": dtype}
    return list(items.values())


def worklist(spec: ProfilingSpec, families: list[str], token_points: list[int]) -> list[dict]:
    """Concatenated, family-tagged work-list across the requested families."""
    fams = set(families)
    out: list[dict] = []
    if "gemm" in fams:
        for it in gemm_worklist(spec, token_points):
            it = dict(it); it["family"] = "gemm"; out.append(it)
        for it in bmm_worklist(spec, token_points):
            it = dict(it); it["family"] = "gemm"; out.append(it)
    if "attention" in fams:
        out += attention_worklist(spec, token_points)
    if "normalization" in fams:
        out += norm_worklist(spec, token_points)
    if "elementwise" in fams:
        out += elementwise_worklist(spec, token_points)
    if "reduction" in fams:
        out += reduction_worklist(spec, token_points)
    return out
