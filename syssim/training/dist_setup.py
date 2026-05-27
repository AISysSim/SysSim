"""Fake process group helpers for single-process SPMD tracing.

Uses PyTorch's "fake" backend so `dist.get_world_size()` and
`dist.get_rank()` return the requested values without any inter-process
negotiation. Sub-groups created via `dist.new_group()` are also fake.
"""

from __future__ import annotations

import torch.distributed as dist


def init_fake_process_group(world_size: int, rank: int) -> None:
    """Initialize the default process group with the "fake" backend.

    Idempotent: if already initialized, this is a no-op.
    """
    if dist.is_initialized():
        return
    try:
        from torch.distributed.fake_pg import FakeStore
    except ImportError:
        from torch.testing._internal.distributed.fake_pg import FakeStore
    store = FakeStore()
    dist.init_process_group(backend="fake", store=store, rank=rank, world_size=world_size)


def destroy_process_group() -> None:
    """Tear down the default process group if one exists."""
    if dist.is_initialized():
        dist.destroy_process_group()
