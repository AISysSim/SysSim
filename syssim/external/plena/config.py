"""PLENA configuration helpers for SysSim integration.

Provides PLENAConfig dataclass for locating PLENA configuration files
and utilities for type checking.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PLENAConfig:
    """Configuration for PLENA accelerator simulation.

    Holds paths to the PLENA configuration files needed to initialize
    the PerfModel from PLENA_Simulator.

    Attributes:
        settings_path: Path to plena_settings.toml
        isa_lib_path: Path to customISA_lib.json
        frequency_hz: Clock frequency in Hz (default: 1 GHz)
    """

    settings_path: str
    isa_lib_path: str
    frequency_hz: float = 1e9

    def __post_init__(self) -> None:
        """Validate that config files exist."""
        settings = Path(self.settings_path)
        isa_lib = Path(self.isa_lib_path)

        if not settings.exists():
            raise FileNotFoundError(f"PLENA settings file not found: {self.settings_path}")
        if not isa_lib.exists():
            raise FileNotFoundError(f"PLENA ISA library not found: {self.isa_lib_path}")

    @classmethod
    def from_plena_submodule(
        cls,
        submodule_path: str | Path | None = None,
        frequency_hz: float = 1e9,
    ) -> "PLENAConfig":
        """Create PLENAConfig by auto-locating files in PLENA_Simulator submodule.

        Args:
            submodule_path: Path to PLENA_Simulator directory.
                           If None, searches relative to SysSim root.
            frequency_hz: Clock frequency in Hz (default: 1 GHz)

        Returns:
            PLENAConfig with paths to config files

        Raises:
            FileNotFoundError: If PLENA_Simulator or config files not found

        Example:
            >>> config = PLENAConfig.from_plena_submodule()
            >>> print(config.settings_path)
        """
        if submodule_path is None:
            # Locate the submodule relative to this file:
            # syssim/external/plena/config.py -> <repo root>/third_party/PLENA_Simulator
            repo_root = Path(__file__).resolve().parents[3]
            submodule_path = repo_root / "third_party" / "PLENA_Simulator"

        submodule_path = Path(submodule_path)
        if not submodule_path.exists():
            raise FileNotFoundError(
                f"PLENA_Simulator submodule not found at {submodule_path}. "
                "Please run: git submodule update --init"
            )

        settings_path = submodule_path / "plena_settings.toml"
        isa_lib_path = submodule_path / "analytic_models" / "performance" / "customISA_lib.json"

        return cls(
            settings_path=str(settings_path),
            isa_lib_path=str(isa_lib_path),
            frequency_hz=frequency_hz,
        )


def get_plena_perf_model(config: PLENAConfig):
    """Load PLENA PerfModel from configuration.

    Args:
        config: PLENAConfig with paths to config files

    Returns:
        PerfModel instance from PLENA_Simulator

    Raises:
        ImportError: If PLENA_Simulator is not importable
    """
    # Add PLENA_Simulator to path if needed
    plena_perf_dir = Path(config.settings_path).parent / "analytic_models" / "performance"
    if str(plena_perf_dir) not in sys.path:
        sys.path.insert(0, str(plena_perf_dir))

    from perf_model import PerfModel, load_hardware_config_from_toml

    hardware_config = load_hardware_config_from_toml(config.settings_path)
    return PerfModel(hardware_config, config.isa_lib_path)
