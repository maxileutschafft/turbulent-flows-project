"""Definitive-screening-style design utilities for surrogate hyperparameters.

The repository does not train the surrogate yet, so this module focuses on the
screening stage: selecting a compact 3-level experiment matrix that is well
suited to identify the most influential training hyperparameters before any
deeper Bayesian optimization.

The design generator is DSD-sized and DSD-shaped: it uses the classical
``2k + 1`` / ``2k + 3`` run budget and selects rows from the full 3-level cube
by maximizing the information content of a main-effects + quadratic model.
This is a practical screening methodology for the current repository.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import csv
import json

import numpy as np


LEVELS = (-1, 0, 1)


@dataclass(frozen=True, slots=True)
class ScreeningFactor:
    """One factor in a 3-level screening design."""

    name: str
    low: float
    center: float
    high: float
    kind: str = "linear"
    integer: bool = False

    def value(self, level: int) -> float:
        if level not in LEVELS:
            raise ValueError(f"Unsupported coded level {level!r} for factor {self.name!r}")
        if level == -1:
            value = self.low
        elif level == 0:
            value = self.center
        else:
            value = self.high
        if self.integer:
            return int(round(value))
        return float(value)


@dataclass(frozen=True, slots=True)
class ScreeningObjective:
    """Metrics that the training campaign should optimize."""

    primary_metrics: tuple[str, ...]
    weights: Mapping[str, float]
    secondary_metrics: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class ScreeningDesign:
    """Generated screening plan with coded and actual levels."""

    factors: tuple[ScreeningFactor, ...]
    coded_matrix: np.ndarray
    objective: ScreeningObjective

    @property
    def runs(self) -> int:
        return int(self.coded_matrix.shape[0])

    def to_rows(self) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        for run_id, coded_row in enumerate(self.coded_matrix, start=1):
            row: dict[str, float | int] = {"run_id": run_id}
            for factor, level in zip(self.factors, coded_row, strict=True):
                row[f"{factor.name}_coded"] = int(level)
                row[factor.name] = factor.value(int(level))
            rows.append(row)
        return rows

    def to_dict(self) -> dict:
        return {
            "factors": [asdict(factor) for factor in self.factors],
            "coded_matrix": self.coded_matrix.tolist(),
            "objective": {
                "primary_metrics": list(self.objective.primary_metrics),
                "secondary_metrics": list(self.objective.secondary_metrics),
                "weights": dict(self.objective.weights),
                "description": self.objective.description,
            },
        }


def recommended_gno_factors() -> tuple[ScreeningFactor, ...]:
    """Return a default GNO hyperparameter screening set.

    These ranges are intentionally conservative for the current checkpoint-based
    surrogate family and should be widened only after the first screening pass.
    """
    return (
        ScreeningFactor("width_node", 16, 32, 64, integer=True),
        ScreeningFactor("depth", 4, 6, 8, integer=True),
        ScreeningFactor("ker_width", 128, 256, 512, integer=True),
        ScreeningFactor("k_neighbors", 8, 16, 24, integer=True),
        ScreeningFactor("learning_rate", 3e-4, 1e-3, 3e-3, kind="log"),
        ScreeningFactor("weight_decay", 1e-6, 1e-5, 1e-4, kind="log"),
    )


def recommended_objective() -> ScreeningObjective:
    """Return the preferred objective set for the screening stage."""
    primary = (
        "val_nrmse_u",
        "val_nrmse_v",
        "val_nrmse_p",
        "val_nrmse_k",
        "val_nrmse_omega",
        "val_nrmse_nut",
    )
    secondary = (
        "val_cl_rel_err",
        "val_cd_rel_err",
        "val_near_wall_rmse",
    )
    weights = {
        "val_nrmse_u": 1.0,
        "val_nrmse_v": 1.0,
        "val_nrmse_p": 1.25,
        "val_nrmse_k": 0.75,
        "val_nrmse_omega": 0.75,
        "val_nrmse_nut": 0.75,
        "val_cl_rel_err": 2.0,
        "val_cd_rel_err": 3.0,
        "val_near_wall_rmse": 1.5,
    }
    return ScreeningObjective(
        primary_metrics=primary,
        secondary_metrics=secondary,
        weights=weights,
        description=(
            "Minimize weighted validation error across flow fields, then use "
            "lift/drag and near-wall error as tie-breakers."
        ),
    )


def screening_run_count(n_factors: int) -> int:
    """Classical DSD-sized run budget for a factor count."""
    if n_factors <= 0:
        raise ValueError("n_factors must be positive")
    return 2 * n_factors + (1 if n_factors % 2 == 0 else 3)


def _candidate_matrix(n_factors: int) -> np.ndarray:
    candidates = np.array(list(product(LEVELS, repeat=n_factors)), dtype=np.int8)
    if candidates.size == 0:
        raise ValueError("Could not build candidate matrix")
    return candidates


def _model_matrix(coded_matrix: np.ndarray) -> np.ndarray:
    main = coded_matrix.astype(np.float64)
    quad = main ** 2
    intercept = np.ones((coded_matrix.shape[0], 1), dtype=np.float64)
    return np.concatenate([intercept, main, quad], axis=1)


def _d_opt_score(coded_matrix: np.ndarray, ridge: float = 1e-9) -> float:
    phi = _model_matrix(coded_matrix)
    xtx = phi.T @ phi + ridge * np.eye(phi.shape[1], dtype=np.float64)
    sign, logdet = np.linalg.slogdet(xtx)
    if sign <= 0:
        return float("-inf")
    return float(logdet)


def _balance_penalty(coded_matrix: np.ndarray) -> float:
    target = coded_matrix.shape[0] / 3.0
    penalty = 0.0
    for col in coded_matrix.T:
        for level in LEVELS:
            penalty += float((np.count_nonzero(col == level) - target) ** 2)
    return penalty


def definitive_screening_design(
    factors: Sequence[ScreeningFactor],
    *,
    seed: int | None = None,
    max_candidates: int | None = None,
    balance_weight: float = 0.25,
) -> ScreeningDesign:
    """Build a practical DSD-style screening plan.

    The design is selected from the full 3-level hypercube by greedily adding
    rows that maximize D-optimality for a main-effects + quadratic surrogate
    model while keeping the coded levels reasonably balanced.

    Parameters
    ----------
    factors:
        Ordered factor definitions with low / center / high values.
    seed:
        Optional random seed used to break ties deterministically.
    max_candidates:
        Optional cap on the number of 3-level rows considered. If provided,
        a random subset of the full cube is used after always keeping the
        center point.
    balance_weight:
        Strength of the level-balance penalty. Increase if the design becomes
        too concentrated on one level.
    """
    factor_tuple = tuple(factors)
    n_factors = len(factor_tuple)
    if n_factors == 0:
        raise ValueError("At least one factor is required")

    n_runs = screening_run_count(n_factors)
    candidates = _candidate_matrix(n_factors)

    center = np.zeros((1, n_factors), dtype=np.int8)
    if not np.any(np.all(candidates == center, axis=1)):
        raise RuntimeError("Center point missing from candidate cube")

    if max_candidates is not None and max_candidates < len(candidates):
        rng = np.random.default_rng(seed)
        keep = {tuple(center[0].tolist())}
        others = [tuple(row.tolist()) for row in candidates if not np.all(row == 0)]
        rng.shuffle(others)
        keep.update(others[: max_candidates - 1])
        candidates = np.array(list(keep), dtype=np.int8)
        if not np.any(np.all(candidates == center, axis=1)):
            candidates = np.vstack([center, candidates])

    rng = np.random.default_rng(seed)
    selected_rows: list[np.ndarray] = [center[0]]
    selected_keys = {tuple(center[0].tolist())}

    while len(selected_rows) < n_runs:
        best_row: np.ndarray | None = None
        best_score = float("-inf")
        candidate_order = np.arange(len(candidates))
        rng.shuffle(candidate_order)
        for idx in candidate_order:
            row = candidates[idx]
            key = tuple(int(value) for value in row)
            if key in selected_keys:
                continue
            trial = np.vstack([selected_rows, row])
            score = _d_opt_score(trial) - balance_weight * _balance_penalty(trial)
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is None:
            raise RuntimeError("Could not construct a screening design from the candidate cube")
        selected_rows.append(best_row)
        selected_keys.add(tuple(int(value) for value in best_row))

    coded_matrix = np.vstack(selected_rows).astype(np.int8)
    return ScreeningDesign(
        factors=factor_tuple,
        coded_matrix=coded_matrix,
        objective=recommended_objective(),
    )


def write_screening_csv(design: ScreeningDesign, path: str | Path) -> Path:
    """Write the screening plan to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = design.to_rows()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_screening_json(design: ScreeningDesign, path: str | Path) -> Path:
    """Write the screening plan to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(design.to_dict(), handle, indent=2)
        handle.write("\n")
    return path
