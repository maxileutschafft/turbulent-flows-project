"""Design-of-experiments helpers for simulation campaigns and surrogate training.

These utilities create candidate case sets that can be executed in ANSYS or
used to build balanced surrogate training and validation splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Single CFD case specification for airfoil surrogate studies."""

    naca_code: str
    reynolds: float
    angle_of_attack: float


@dataclass(frozen=True, slots=True)
class FlowBounds:
    """Continuous design bounds for the flow parameters."""

    reynolds: tuple[float, float]
    angle_of_attack: tuple[float, float]


def _scale_unit_interval(samples: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return lo + (hi - lo) * samples


def full_factorial_cases(
    naca_codes: Sequence[str],
    reynolds_values: Sequence[float],
    angle_of_attack_values: Sequence[float],
) -> list[CaseSpec]:
    """Return a Cartesian product of discrete design points."""
    return [
        CaseSpec(naca_code=code, reynolds=float(re), angle_of_attack=float(aoa))
        for code, re, aoa in product(naca_codes, reynolds_values, angle_of_attack_values)
    ]


def latin_hypercube_cases(
    n_samples: int,
    *,
    naca_codes: Sequence[str],
    bounds: FlowBounds,
    seed: int | None = None,
) -> list[CaseSpec]:
    """Sample a mixed discrete/continuous DOE using a Latin hypercube."""
    if n_samples <= 0:
        return []
    if not naca_codes:
        raise ValueError("naca_codes must not be empty")

    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit_samples = sampler.random(n_samples)
    reynolds = _scale_unit_interval(unit_samples[:, 0], bounds.reynolds)
    aoa = _scale_unit_interval(unit_samples[:, 1], bounds.angle_of_attack)

    codes = np.asarray(list(naca_codes), dtype=object)
    code_indices = np.arange(n_samples) % len(codes)
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(code_indices)

    return [
        CaseSpec(
            naca_code=str(codes[code_indices[i]]),
            reynolds=float(reynolds[i]),
            angle_of_attack=float(aoa[i]),
        )
        for i in range(n_samples)
    ]


def greedy_maximin_subset(
    candidates: Sequence[CaseSpec],
    n_select: int,
) -> list[CaseSpec]:
    """Select a diverse subset using a greedy maximin criterion."""
    if n_select <= 0 or not candidates:
        return []
    if n_select >= len(candidates):
        return list(candidates)

    reynolds = np.array([case.reynolds for case in candidates], dtype=np.float64)
    aoa = np.array([case.angle_of_attack for case in candidates], dtype=np.float64)
    re_scale = np.ptp(reynolds) or 1.0
    aoa_scale = np.ptp(aoa) or 1.0

    def distance(i: int, j: int) -> float:
        re_term = abs(reynolds[i] - reynolds[j]) / re_scale
        aoa_term = abs(aoa[i] - aoa[j]) / aoa_scale
        geom_term = 1.0 if candidates[i].naca_code != candidates[j].naca_code else 0.0
        return float(np.sqrt(re_term**2 + aoa_term**2 + geom_term**2))

    selected = [0]
    remaining = set(range(1, len(candidates)))
    while len(selected) < n_select and remaining:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            score = min(distance(idx, chosen) for chosen in selected)
            if score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[idx] for idx in selected]