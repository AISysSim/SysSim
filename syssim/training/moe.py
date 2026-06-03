"""Tracer-scoped patches that make Megatron's MoE expert path traceable under
fake tensors.

`SequentialMLP.forward` computes its per-expert token split from
`tokens_per_expert.tolist()`. Under FakeTensorMode fake tensors carry shape but
not data, so `.tolist()` returns zeros and `torch.split` produces empty chunks —
the expert GEMMs never run. With the trace-safe MoE config (capacity_factor=1.0,
pad_expert_input_to_capacity=True, force_load_balancing=True) the permuted buffer
has exactly `capacity * num_local_experts` rows, so an EQUAL split derived from the
static row count (a SHAPE, which fake tensors DO carry) reproduces the real shapes.

The context manager is gated on `provider.num_moe_experts`: for a dense provider it
is a no-op and the dense trace stays byte-identical.
"""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def moe_fake_trace_patches(provider):
    """Patch `SequentialMLP.forward` for the duration of a fake-tensor trace.

    No-op when `provider.num_moe_experts` is None (dense path). Restores the
    original method on exit.
    """
    if getattr(provider, "num_moe_experts", None) is None:
        yield
        return

    import torch
    from megatron.core.transformer.moe.experts import SequentialMLP

    original_forward = SequentialMLP.forward

    def forward_with_static_split(self, permuted_local_hidden_states, tokens_per_expert,
                                  permuted_probs):
        # Only the multi-local-expert case hits the broken `.tolist()` split. A
        # single local expert (num_local_experts == 1) runs the original forward,
        # which doesn't depend on per-expert token counts.
        if self.num_local_experts <= 1:
            return original_forward(
                self, permuted_local_hidden_states, tokens_per_expert, permuted_probs)

        total_rows = permuted_local_hidden_states.shape[0]
        per_expert = total_rows // self.num_local_experts
        split_sizes = [per_expert] * self.num_local_experts
        remainder = total_rows - per_expert * self.num_local_experts
        if remainder:
            split_sizes[-1] += remainder

        token_chunks = torch.split(permuted_local_hidden_states, split_sizes)
        prob_chunks = torch.split(permuted_probs, split_sizes)
        expert_outputs = []
        for expert, tokens, probs in zip(self.local_experts, token_chunks, prob_chunks):
            output, _ = expert(tokens, probs)
            expert_outputs.append(output)
        return torch.cat(expert_outputs, dim=0), None

    SequentialMLP.forward = forward_with_static_split
    try:
        yield
    finally:
        SequentialMLP.forward = original_forward
