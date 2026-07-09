"""Data schema helpers for the GNO surrogate (inference subset)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

POS_COLS = ["x", "y"]
GNO_INPUT_COLS = ["x", "y", "sdf", "u_init", "v_init", "angle_of_attack", "reynolds"]
GNO_TARGET_COLS = ["u", "v", "p", "k", "omega", "nut"]

INPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "u_init": ("u0", "u_inf", "u_freestream"),
    "v_init": ("v0", "v_inf", "v_freestream"),
}

TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "nut": ("nu_t", "nu_turb", "turbulent_viscosity"),
}

SCENARIO_SCALAR_COLS = {"reynolds", "angle_of_attack"}

# Target channels that are strictly positive and span many orders of magnitude.
# Training applied log10(x + TARGET_LOG_EPS) BEFORE z-score normalization, so
# inference inverts: 10**x - TARGET_LOG_EPS.
LOG_TARGET_COLS = ("k", "omega", "nut")
TARGET_LOG_EPS = 1.0e-10


def _candidate_names(name: str, aliases: Mapping[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    if aliases is None:
        return (name,)
    return (name, *aliases.get(name, ()))


def resolve_key(
    raw: Mapping[str, Any],
    name: str,
    *,
    aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> str | None:
    """Resolve a canonical key or one of its aliases in ``raw``."""
    for candidate in _candidate_names(name, aliases):
        if candidate in raw:
            return candidate
    return None


def require_keys(
    raw: Mapping[str, Any],
    names: list[str] | tuple[str, ...],
    *,
    aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """Return the canonical keys that are missing from ``raw``."""
    missing: list[str] = []
    for name in names:
        if resolve_key(raw, name, aliases=aliases) is None:
            missing.append(name)
    return missing


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
    key = resolve_key(raw, name, aliases=INPUT_ALIASES)
    if key is None:
        raise KeyError(f"Missing required input column {name!r}")
    if name in SCENARIO_SCALAR_COLS:
        return np.full(n_nodes, float(raw[key]), dtype=np.float32)
    return np.asarray(raw[key], dtype=np.float32)


def _target_column(raw: Mapping[str, Any], name: str) -> np.ndarray:
    key = resolve_key(raw, name, aliases=TARGET_ALIASES)
    if key is None:
        raise KeyError(f"Missing required target column {name!r}")
    return np.asarray(raw[key], dtype=np.float32)


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
    return _float32_tensor(
        np.column_stack([_target_column(raw, col) for col in GNO_TARGET_COLS])
    )
