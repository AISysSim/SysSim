"""Shared pytest configuration.

`requires_cuda` marks tests that need a real CUDA GPU (the tracer requires
`torch.cuda.is_available()`, and the integration tests build real Megatron models). On a
CPU-only host (e.g. a login node) those tests are skipped so the suite stays green; the
CUDA path runs on a GPU node. Mark a module with `pytestmark = pytest.mark.requires_cuda`
or an individual test with `@pytest.mark.requires_cuda`. The GPU-only test modules below are
also skipped automatically on CPU-only hosts.
"""
import os

import pytest

try:
    import torch
    _HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    _HAS_CUDA = False

# Whole modules that exercise the tracer / a real Megatron model and therefore need a GPU.
_CUDA_MODULES = frozenset({
    "test_tracing.py",
    "test_report.py",
    "test_spmd_regression.py",
    "test_pipeline_parallel.py",
    "test_pro6000_hw_detect.py",
    "test_cli.py",
    "test_end_to_end.py",
    "test_mem_tracker_integration.py",
    "test_mem_profile.py",
    "test_runner.py",
    "test_runner_topology_wiring.py",
    "test_sweep.py",
    "test_tracer_capture.py",
    "test_tree_integration.py",
    "test_measure.py",
    "test_measure_families.py",
    "test_calibrate_families.py",
})


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_cuda: needs a CUDA GPU; skipped on CPU-only hosts")


def pytest_collection_modifyitems(config, items):
    if _HAS_CUDA:
        return
    skip = pytest.mark.skip(reason="requires a CUDA GPU; run on a GPU node")
    for item in items:
        if "requires_cuda" in item.keywords or os.path.basename(str(item.fspath)) in _CUDA_MODULES:
            item.add_marker(skip)
