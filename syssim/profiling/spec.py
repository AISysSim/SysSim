"""Profiling-spec YAML: the architectural space to cover (profiling-time only)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_SPEC_PATH = os.path.join(os.path.dirname(__file__), "default_spec.yaml")

_ALLOWED = frozenset({
    "hidden_sizes", "ffn_hidden_sizes", "num_attention_heads", "num_query_groups",
    "head_dims", "vocab_sizes", "dtypes", "parallelism",
    "token_range", "seq_len_range", "elem_range", "sweep",
})


@dataclass
class ProfilingSpec:
    hidden_sizes: list
    ffn_hidden_sizes: list
    num_attention_heads: list
    num_query_groups: list
    head_dims: list
    vocab_sizes: list
    dtypes: list
    parallelism: dict
    token_range: dict
    seq_len_range: dict
    elem_range: dict
    sweep: dict = field(default_factory=dict)


def load_profiling_spec(path: str) -> ProfilingSpec:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"profiling spec must be a mapping: {path}")
    bad = set(data) - _ALLOWED
    if bad:
        raise ValueError(f"profiling spec has disallowed key(s): {', '.join(sorted(bad))}")
    return ProfilingSpec(**{k: data[k] for k in _ALLOWED if k in data})
