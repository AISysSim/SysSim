"""Smoke tests for Tenstorrent Blackhole hardware specs.

Confirms the Blackhole entries added to ``TT_HW_DATABASE`` are wired through
``HardwareInfo`` and the auto-detect helper. These tests do not require a
real Tenstorrent device — they patch the ``ttnn`` device handle.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from syssim.config import HardwareInfo, get_hardware_info
from syssim.compute.compute_cost_profiler import (
    TT_HW_DATABASE,
    _auto_detect_tt_platform,
)


BLACKHOLE_VARIANTS = ("tt_bh_p150b", "tt_bh_p150a", "tt_bh_p100a")


class TestBlackholeDatabase:
    """Sanity checks on the Blackhole entries in ``TT_HW_DATABASE``."""

    @pytest.mark.parametrize("name", BLACKHOLE_VARIANTS)
    def test_entry_present(self, name):
        assert name in TT_HW_DATABASE, f"missing TT_HW_DATABASE entry: {name}"

    @pytest.mark.parametrize("name", BLACKHOLE_VARIANTS)
    def test_entry_shape_and_positivity(self, name):
        mm, mm_cons, math_t, bw = TT_HW_DATABASE[name]
        assert mm > 0 and mm_cons > 0 and math_t > 0 and bw > 0
        # Conservative peak should not exceed nominal peak.
        assert mm_cons <= mm
        # Matrix peak should dominate the vector unit on these chips.
        assert mm > math_t

    def test_p150_faster_than_p100(self):
        """P150 is the larger Blackhole variant — confirm specs reflect that."""
        p150 = TT_HW_DATABASE["tt_bh_p150b"]
        p100 = TT_HW_DATABASE["tt_bh_p100a"]
        assert p150[0] >= p100[0], "P150 BF16 MM peak should be >= P100"
        assert p150[3] >= p100[3], "P150 memory bandwidth should be >= P100"

    @pytest.mark.parametrize("name", BLACKHOLE_VARIANTS)
    def test_hardware_info_constructable(self, name):
        mm, mm_cons, math_t, bw = TT_HW_DATABASE[name]
        hw = HardwareInfo(
            peak_tflops_mm=mm,
            peak_tflops_math=math_t,
            peak_memory_bandwidth_gbps=bw,
            peak_tflops_mm_conservative=mm_cons,
        )
        assert hw.peak_tflops_mm == mm
        assert hw.peak_tflops_mm_conservative == mm_cons
        assert hw.peak_tflops_math == math_t
        assert hw.peak_memory_bandwidth_gbps == bw


class TestAutoDetect:
    """Patch ttnn.get_arch_name / device repr to cover both code paths."""

    @pytest.mark.parametrize(
        "arch_name,expected",
        [
            ("Blackhole", "tt_bh_p150b"),
            ("blackhole", "tt_bh_p150b"),
            ("Wormhole_b0", "tt_wh_n300"),
            ("wormhole_b0", "tt_wh_n300"),
        ],
    )
    def test_get_arch_name_path(self, arch_name, expected):
        """When ``ttnn.get_arch_name()`` is available it should be used directly."""
        fake_ttnn = types.SimpleNamespace(get_arch_name=lambda: arch_name)
        with mock.patch.dict(sys.modules, {"ttnn": fake_ttnn}):
            assert _auto_detect_tt_platform() == expected

    @pytest.mark.parametrize(
        "device_repr,expected",
        [
            ("Device(arch=Blackhole, P150B chip0)", "tt_bh_p150b"),
            ("BlackHole P150A device", "tt_bh_p150a"),
            ("BH P100A", "tt_bh_p100a"),
            ("Wormhole N300 chip", "tt_wh_n300"),
            ("Device(arch=N150)", "tt_wh_n150"),
        ],
    )
    def test_device_repr_fallback(self, device_repr, expected):
        """Fallback path: ttnn.get_arch_name() raises, scrape str(device)."""
        fake_ttnn = types.SimpleNamespace(
            get_arch_name=mock.Mock(side_effect=RuntimeError("no get_arch_name")),
        )
        fake_device = mock.MagicMock()
        fake_device.__str__ = lambda self: device_repr
        with mock.patch.dict(sys.modules, {"ttnn": fake_ttnn}), \
             mock.patch(
                 "syssim.compute.compute_cost_profiler._get_tt_device",
                 return_value=fake_device,
             ):
            assert _auto_detect_tt_platform() == expected

    def test_unknown_falls_back_to_n300(self):
        fake_ttnn = types.SimpleNamespace(get_arch_name=lambda: "unknown_arch")
        with mock.patch.dict(sys.modules, {"ttnn": fake_ttnn}):
            assert _auto_detect_tt_platform() == "tt_wh_n300"


class TestBackendManagerHwOverride:
    """The BackendManager must honour an explicit hw_name without calling
    the CUDA-only auto-detect path. This is what lets a non-CUDA host load
    Blackhole models from disk."""

    def test_explicit_hw_name_skips_cuda_probe(self, tmp_path):
        from syssim.compute.efficiency_models import BackendManager
        # No model files exist — BackendManager should still construct and
        # report no models, but importantly should NOT raise from the
        # missing CUDA device.
        with mock.patch("syssim.config.torch") as torch_mock:
            torch_mock.cuda.is_available.return_value = False
            mgr = BackendManager(str(tmp_path), hw_name="tt_bh_p150b")
        assert mgr.hw_name == "tt_bh_p150b"
        assert all(v is None for v in mgr._models.values()) or mgr._models == {}

    def test_env_var_picks_up_hw_name(self, tmp_path, monkeypatch):
        from syssim.compute.efficiency_models import BackendManager
        monkeypatch.setenv("SYSSIM_HW_NAME", "tt_bh_p100a")
        with mock.patch("syssim.config.torch") as torch_mock:
            torch_mock.cuda.is_available.return_value = False
            mgr = BackendManager(str(tmp_path))
        assert mgr.hw_name == "tt_bh_p100a"


class TestGetHardwareInfoLookup:
    """Patch ``torch.cuda`` so ``get_hardware_info`` matches Blackhole names."""

    @pytest.mark.parametrize(
        "device_name,expected_hw",
        [
            ("Tenstorrent Blackhole P150B", "tt_bh_p150b"),
            ("Tenstorrent BH p150a engine", "tt_bh_p150a"),
            ("blackhole p100a", "tt_bh_p100a"),
        ],
    )
    def test_blackhole_pattern_matches(self, device_name, expected_hw):
        with mock.patch("syssim.config.torch") as torch_mock:
            torch_mock.cuda.is_available.return_value = True
            torch_mock.cuda.get_device_name.return_value = device_name
            hw, hw_name = get_hardware_info()
        assert hw_name == expected_hw
        assert hw.peak_tflops_mm > 0
        assert hw.peak_memory_bandwidth_gbps > 0
