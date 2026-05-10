"""ML efficiency model management and inference."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class EfficiencyFeatures:
    """Features for ML efficiency model."""

    # Roofline envelope features
    constraint_times: dict[tuple[str, str], float]  # T_{k,l}
    constraint_ratios: dict[tuple[str, str], float]  # r_{k,l}
    dominant_constraint: tuple[str, str]

    # Operator parameters (log-scaled)
    op_params: dict[str, float]

    # Hardware descriptors
    hw_params: dict[str, float]

    def to_array(self, feature_order: list[str]) -> np.ndarray:
        """Convert to flat array for ML model input."""
        feature_dict = {}

        # Add constraint times
        for key, val in self.constraint_times.items():
            feature_dict[f"T_{key[0]}_{key[1]}"] = val

        # Add constraint ratios
        for key, val in self.constraint_ratios.items():
            feature_dict[f"r_{key[0]}_{key[1]}"] = val

        # Add dominant constraint (one-hot)
        feature_dict[f"dom_{self.dominant_constraint[0]}_{self.dominant_constraint[1]}"] = 1.0

        # Add operator params
        feature_dict.update(self.op_params)

        # Add hardware params
        feature_dict.update(self.hw_params)

        # Convert to array in specified order
        values = [feature_dict.get(name, 0.0) for name in feature_order]
        return np.array(values, dtype=np.float32)


class EfficiencyModel(ABC):
    """Base class for efficiency models."""

    @abstractmethod
    def predict(self, features: EfficiencyFeatures) -> float:
        """Predict efficiency from features."""
        pass


class MLPEfficiencyModel(EfficiencyModel):
    """MLP-based efficiency model."""

    def __init__(self, model_path: str, feature_order: list[str]):
        self.feature_order = feature_order
        self.device = torch.device("cpu")

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = self._build_model(checkpoint["input_dim"], checkpoint["hidden_dims"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _build_model(self, input_dim: int, hidden_dims: list[int]) -> nn.Module:
        """Build MLP architecture with Sigmoid output layer.

        Note: Includes dropout layers to match training architecture.
        Dropout is automatically disabled in eval mode.
        Sigmoid output layer constrains predictions to (0, 1) range.
        """
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            # Add dropout after all but the last hidden layer
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output in (0, 1)
        return nn.Sequential(*layers)

    def predict(self, features: EfficiencyFeatures) -> float:
        """Predict efficiency in [0, 1] using Sigmoid-activated model.

        Returns:
            Predicted efficiency η ∈ [0.01, 1.0]
            - Sigmoid automatically constrains output to (0, 1)
            - Additional clamp ensures minimum 0.01 for numerical stability
        """
        x = features.to_array(self.feature_order)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            eta = self.model(x_tensor).item()  # Sigmoid already applied

        # Sigmoid guarantees (0, 1), but add safety clamp for edge cases
        return float(np.clip(eta, 0.01, 1.0))


class XGBoostEfficiencyModel(EfficiencyModel):
    """XGBoost-based efficiency model."""

    def __init__(self, model_path: str, feature_order: list[str]):
        import pickle

        self.feature_order = feature_order

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Reconstruct XGBoost model from pickle
        model_bytes = checkpoint["model_state_dict"]
        self.model = pickle.loads(model_bytes)

    def predict(self, features: EfficiencyFeatures) -> float:
        """Predict efficiency using XGBoost.

        Returns:
            Predicted efficiency η ∈ [0.01, 1.0]
            - XGBoost may predict outside [0,1], so we clip to valid range
        """
        x = features.to_array(self.feature_order)

        # XGBRegressor (sklearn API) expects 2D numpy array directly
        # NOT DMatrix (that's for the raw Booster API)
        prediction = self.model.predict(x.reshape(1, -1))[0]

        # XGBoost may predict outside [0, 1], clip to valid efficiency range
        return float(np.clip(prediction, 0.01, 1.0))


class BackendManager:
    """Manages efficiency models per (operator, dtype) class.

    Files are looked up as ``{op}_{hw}_{dtype}_xgb.pth`` first, then ``_mlp.pth``.
    Legacy files without a ``_{dtype}_`` segment (e.g. existing GH200 models) are
    treated as fp16 for backwards compatibility.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self.hw_name: Optional[str] = None
        self._models: dict[tuple[Any, str], Optional[EfficiencyModel]] = {}

        if model_dir is not None:
            self._load_models()

    def _load_models(self):
        """Load all available models (XGBoost or MLP) from model_dir for this hw."""
        if not os.path.isdir(self.model_dir):
            log.warning(f"Model directory not found: {self.model_dir}")
            return

        # Import OperatorType and get_hardware_info here to avoid circular import
        from ..operator_graph import OperatorType
        from ..config import get_hardware_info

        # Get hardware name
        try:
            _, hw_name = get_hardware_info()
        except RuntimeError as e:
            log.warning(f"Could not identify hardware: {e}")
            return
        self.hw_name = hw_name

        dtypes = ("fp16", "fp8", "fp4")

        def _try_load_xgb(path: str):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            feature_order = checkpoint["feature_order"]
            return XGBoostEfficiencyModel(path, feature_order)

        def _try_load_mlp(path: str):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            feature_order = checkpoint["feature_order"]
            return MLPEfficiencyModel(path, feature_order)

        for op_type in OperatorType:
            for dtype in dtypes:
                # Per-dtype filenames first ({op}_{hw}_{dtype}_xgb.pth)
                xgb_path = os.path.join(self.model_dir, f"{op_type.value}_{hw_name}_{dtype}_xgb.pth")
                mlp_path = os.path.join(self.model_dir, f"{op_type.value}_{hw_name}_{dtype}_mlp.pth")

                # Legacy fallback for FP16 only (no dtype in filename, e.g. gh200)
                legacy_xgb = os.path.join(self.model_dir, f"{op_type.value}_{hw_name}_xgb.pth")
                legacy_mlp = os.path.join(self.model_dir, f"{op_type.value}_{hw_name}_mlp.pth")

                model: Optional[EfficiencyModel] = None
                try:
                    if os.path.exists(xgb_path):
                        model = _try_load_xgb(xgb_path)
                        log.info(f"Loaded XGBoost {op_type.value} {dtype} ({hw_name})")
                    elif os.path.exists(mlp_path):
                        model = _try_load_mlp(mlp_path)
                        log.info(f"Loaded MLP {op_type.value} {dtype} ({hw_name})")
                    elif dtype == "fp16" and os.path.exists(legacy_xgb):
                        model = _try_load_xgb(legacy_xgb)
                        log.info(f"Loaded legacy XGBoost {op_type.value} ({hw_name}) as fp16")
                    elif dtype == "fp16" and os.path.exists(legacy_mlp):
                        model = _try_load_mlp(legacy_mlp)
                        log.info(f"Loaded legacy MLP {op_type.value} ({hw_name}) as fp16")
                except Exception as e:
                    log.warning(f"Failed to load {op_type.value} {dtype}: {e}")
                    model = None

                self._models[(op_type, dtype)] = model

    def _resolve_model_path(self, op_type: Any, dtype: str) -> str:
        """Build the canonical model file path for testing helpers."""
        return os.path.join(
            self.model_dir or "",
            f"{op_type.value}_{self.hw_name}_{dtype}_xgb.pth",
        )

    def get_model(self, op_type: Any, dtype: str = "fp16") -> Optional[EfficiencyModel]:
        """Get model for (operator type, dtype), or None if unavailable.

        Falls back to fp16 when the requested dtype has no model loaded; this
        keeps the predictor responsive on hardware without per-dtype training.
        """
        m = self._models.get((op_type, dtype))
        if m is not None:
            return m
        if dtype != "fp16":
            return self._models.get((op_type, "fp16"))
        return None


# Singleton instance
_backend_manager: Optional[BackendManager] = None


def get_backend_manager() -> BackendManager:
    """Get global model manager instance."""
    global _backend_manager
    if _backend_manager is None:
        # Check environment variable for model directory
        model_dir = os.environ.get("RLSYSIM_MODEL_DIR")
        _backend_manager = BackendManager(model_dir)
    return _backend_manager


def set_backend_dir(model_dir: str):
    """Set the directory containing trained efficiency models."""
    global _backend_manager
    _backend_manager = BackendManager(model_dir)
