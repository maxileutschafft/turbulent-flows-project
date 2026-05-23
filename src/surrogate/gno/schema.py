"""Data schema helpers for the GNO surrogate (inference subset)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

POS_COLS = ["x", "y"]
GNO_INPUT_COLS = ["x", "y", "sdf", "u_init", "v_init", "angle_of_attack", "reynolds"]
GNO_TARGET_COLS = ["u", "v", "p", "k", "omega", "nut"]

SCENARIO_SCALAR_COLS = {"reynolds", "angle_of_attack"}

# Target channels that are strictly positive and span many orders of magnitude.
# Training applied log10(x + TARGET_LOG_EPS) BEFORE z-score normalization, so
# inference inverts: 10**x - TARGET_LOG_EPS.
LOG_TARGET_COLS = ("k", "omega", "nut")
TARGET_LOG_EPS = 1.0e-10


def _log_target_indices() -> tuple[int, ...]:
    return tuple(GNO_TARGET_COLS.index(c) for c in LOG_TARGET_COLS)


def apply_target_log_inverse(target: torch.Tensor) -> torch.Tensor:
    """Inverse of the training-time log transform. Returns a new tensor."""
    out = target.clone()
    for idx in _log_target_indices():
        out[..., idx] = (
            torch.pow(10.0, target[..., idx].to(torch.float64)) - TARGET_LOG_EPS
        ).to(out.dtype)
    return out


def _float32_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _column(raw: Mapping[str, Any], name: str, n_nodes: int) -> np.ndarray:
    if name in SCENARIO_SCALAR_COLS:
        return np.full(n_nodes, float(raw[name]), dtype=np.float32)
    return np.asarray(raw[name], dtype=np.float32)


def npz_pos(raw: Mapping[str, Any]) -> torch.Tensor:
    """Return spatial coordinates from a scenario `.npz` payload."""
    return _float32_tensor(np.column_stack([raw["x"], raw["y"]]))


def npz_gno_input(raw: Mapping[str, Any]) -> torch.Tensor:
    """Return GNO node input features in `GNO_INPUT_COLS` order."""
    n_nodes = len(raw["x"])
    return _float32_tensor(
        np.column_stack([_column(raw, col, n_nodes) for col in GNO_INPUT_COLS])
    )


def npz_gno_target(raw: Mapping[str, Any]) -> torch.Tensor:
    """Return GNO training targets in `GNO_TARGET_COLS` order."""
    return _float32_tensor(np.column_stack([raw[col] for col in GNO_TARGET_COLS]))
