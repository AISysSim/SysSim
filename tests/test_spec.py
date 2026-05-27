"""Tests for syssim.training.spec."""

import pytest
from syssim.training.spec import ParallelismConfig


def test_parallelism_allows_pp_greater_than_one():
    p = ParallelismConfig(pp=4)
    assert p.pipeline_model_parallel_size == 4
    assert p.world_size == 4  # tp=dp=cp=1, pp=4


def test_parallelism_world_size_includes_pp():
    p = ParallelismConfig(tp=4, dp=2, cp=2, pp=8)
    assert p.world_size == 4 * 2 * 2 * 8


def test_parallelism_vpp_kwarg():
    p = ParallelismConfig(pp=4, vpp=2)
    assert p.virtual_pipeline_model_parallel_size == 2


def test_parallelism_vpp_default_none():
    p = ParallelismConfig(pp=4)
    assert p.virtual_pipeline_model_parallel_size is None


def test_parallelism_pp_rejects_zero():
    with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
        ParallelismConfig(pp=0)
