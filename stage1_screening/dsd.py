"""Definitive Screening Design (DSD) over the airfoil INPUT space.

This is a screening design over the *operating conditions* the GNO surrogate is
asked to predict — NACA geometry (m, p, t), Reynolds number and angle of attack —
NOT over training hyperparameters (that is `src/utils/screening.py`).

A Definitive Screening Design (Jones & Nachtsheim, 2011) is built from a
conference matrix ``C`` of order ``n``::

    D = [  C ]      each factor at 3 levels {-1, 0, +1}
        [ -C ]      main effects orthogonal to each other and to 2FI / quadratic
        [  0 ]      one overall center run

giving ``2n + 1`` runs. With 5 real factors we use a conference matrix of order
6 (the 6th column is a "fake"/ghost factor that provides pure-error degrees of
freedom), i.e. **13 runs**.

Why a DSD here: it is a compact first screen that tells you *which* inputs drive
the surrogate error (main effects + curvature) before you spend expensive CFD
runs. It does NOT by itself localise *where* the error peaks — the largest error
usually sits at the domain edges, which a DSD deliberately under-samples. Use the
DSD to rank the factors, then follow up with targeted edge/OOD sweeps and an
adaptive (Bayesian) search. See README.md.

Reference:
    B. Jones, C. J. Nachtsheim, "A Class of Three-Level Designs for Definitive
    Screening in the Presence of Second-Order Effects," J. Quality Technology,
    43(1), 2011.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Factors and proposed bounds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Factor:
    """One 3-level factor. ``low``/``center``/``high`` map to coded -1/0/+1."""

    name: str
    low: float
    center: float
    high: float
    integer: bool = False
    unit: str = ""

    def value(self, level: int) -> float:
        v = {-1: self.low, 0: self.center, 1: self.high}[int(level)]
        return int(round(v)) if self.integer else float(v)


# The 5 factors are the digits of the NACA-4 code plus the two flow parameters.
#   m  = 1st digit  = max camber           [% chord]
#   p  = 2nd digit  = position of max camber [1/10 chord]
#   t  = digits 3-4 = max thickness         [% chord]
#   reynolds, aoa   = freestream condition
#
# NOTE on the geometry bounds: m is kept >= 2 on purpose. For m == 0 (symmetric
# airfoil) the camber-position factor p has NO effect, which would confound the
# DSD main-effect estimate for p. Keeping camber present at every run keeps all
# five factors active and estimable.

PRESETS: dict[str, list[Factor]] = {
    # In-distribution screen: quantify interpolation quality and rank which
    # inputs drive the error INSIDE the training box (Re 1e5-5e5, AoA +-5 deg).
    "in_distribution": [
        Factor("m", 2, 4, 6, integer=True, unit="%c"),
        Factor("p", 2, 4, 6, integer=True, unit="1/10 c"),
        Factor("t", 9, 15, 21, integer=True, unit="%c"),
        Factor("reynolds", 1.0e5, 3.0e5, 5.0e5, unit="-"),
        Factor("aoa", -5.0, 0.0, 5.0, unit="deg"),
    ],
    # OOD probe: edges pushed just past the training box so the corners graze
    # out-of-distribution (AoA +-10 deg > +-5 training; Re up to 9e5 > 5e5).
    # Use this to see how fast the error grows leaving the training envelope.
    "ood_probe": [
        Factor("m", 2, 4, 6, integer=True, unit="%c"),
        Factor("p", 2, 4, 6, integer=True, unit="1/10 c"),
        Factor("t", 6, 15, 24, integer=True, unit="%c"),
        Factor("reynolds", 1.0e5, 5.0e5, 9.0e5, unit="-"),
        Factor("aoa", -10.0, 0.0, 10.0, unit="deg"),
    ],
}

# Training envelope, for flagging which runs are already out-of-distribution.
TRAIN_RE = (1.0e5, 5.0e5)
TRAIN_AOA = (-5.0, 5.0)


# --------------------------------------------------------------------------- #
# Conference matrix + DSD construction
# --------------------------------------------------------------------------- #


def _legendre_symbol(a: int, p: int) -> int:
    a %= p
    if a == 0:
        return 0
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else 1


def conference_matrix(n: int) -> np.ndarray:
    """Symmetric conference matrix of order ``n`` via the Paley construction.

    Requires ``p = n - 1`` prime with ``p % 4 == 1`` (symmetric case). Order 6
    (p = 5) covers up to 6 factors, which is enough for this study.
    """
    p = n - 1
    if p < 2 or any(p % k == 0 for k in range(2, int(p**0.5) + 1)):
        raise ValueError(f"conference_matrix: n-1={p} must be prime")
    if p % 4 != 1:
        raise ValueError(f"conference_matrix: n-1={p} must be == 1 (mod 4) for the symmetric construction")

    q = np.array([[_legendre_symbol(j - i, p) for j in range(p)] for i in range(p)], dtype=float)
    c = np.zeros((n, n))
    c[0, 1:] = 1.0
    c[1:, 0] = 1.0
    c[1:, 1:] = q
    return c


def _dsd_matrix(n_factors: int) -> tuple[np.ndarray, int]:
    """Return the coded DSD matrix (2n+1, n) and the conference order n used."""
    n = n_factors
    while True:
        try:
            c = conference_matrix(n)
            break
        except ValueError:
            n += 1
            if n > n_factors + 4:
                raise RuntimeError("No conference matrix found near the requested factor count")
    design = np.vstack([c, -c, np.zeros((1, n))])
    return design, n


# --------------------------------------------------------------------------- #
# Public design object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Run:
    run_id: int
    naca_code: str
    reynolds: float
    aoa: float
    coded: dict[str, int]
    ood: bool


class DSDesign:
    def __init__(self, preset: str = "in_distribution"):
        if preset not in PRESETS:
            raise KeyError(f"Unknown preset {preset!r}; choose from {list(PRESETS)}")
        self.preset = preset
        self.factors = PRESETS[preset]
        coded, self.n_conf = _dsd_matrix(len(self.factors))
        self.coded_matrix = coded[:, : len(self.factors)].astype(int)  # drop ghost cols
        self.n_ghost = self.n_conf - len(self.factors)
        self.runs = self._build_runs()

    def _build_runs(self) -> list[Run]:
        runs: list[Run] = []
        for rid, row in enumerate(self.coded_matrix, start=1):
            vals = {f.name: f.value(int(lvl)) for f, lvl in zip(self.factors, row)}
            m, p, t = int(vals["m"]), int(vals["p"]), int(vals["t"])
            code = f"{m}{p}{t:02d}"
            re = float(vals["reynolds"])
            aoa = float(vals["aoa"])
            ood = not (TRAIN_RE[0] <= re <= TRAIN_RE[1] and TRAIN_AOA[0] <= aoa <= TRAIN_AOA[1])
            runs.append(
                Run(
                    run_id=rid,
                    naca_code=code,
                    reynolds=re,
                    aoa=aoa,
                    coded={f.name: int(lvl) for f, lvl in zip(self.factors, row)},
                    ood=ood,
                )
            )
        return runs

    # -- exports ----------------------------------------------------------- #

    def to_rows(self) -> list[dict]:
        rows = []
        for r in self.runs:
            row = {"run_id": r.run_id, "naca_code": r.naca_code, "reynolds": r.reynolds, "aoa": r.aoa, "ood": int(r.ood)}
            row.update({f"{k}_coded": v for k, v in r.coded.items()})
            rows.append(row)
        return rows

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return path

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "preset": self.preset,
            "factors": [asdict(f) for f in self.factors],
            "n_runs": len(self.runs),
            "conference_order": self.n_conf,
            "ghost_factors": self.n_ghost,
            "coded_matrix": self.coded_matrix.tolist(),
            "runs": [asdict(r) for r in self.runs],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    # -- verification ------------------------------------------------------ #

    def verify(self) -> dict[str, bool]:
        """Check the core DSD properties. Returns a dict of pass/fail flags."""
        c = conference_matrix(self.n_conf)
        checks = {}
        # 1. Conference matrix: C C^T = (n-1) I
        checks["conference_orthogonal"] = bool(
            np.allclose(c @ c.T, (self.n_conf - 1) * np.eye(self.n_conf))
        )
        x = self.coded_matrix.astype(float)
        # 2. Every factor uses exactly 3 levels {-1,0,1}
        checks["three_levels"] = all(set(np.unique(x[:, j])) <= {-1.0, 0.0, 1.0} for j in range(x.shape[1]))
        # 3. Each factor column is balanced (sum 0) -> main effect unbiased by mean
        checks["columns_zero_sum"] = bool(np.allclose(x.sum(axis=0), 0.0))
        # 4. Main effects mutually orthogonal (off-diagonal of X^T X == 0)
        g = x.T @ x
        checks["main_effects_orthogonal"] = bool(np.allclose(g - np.diag(np.diag(g)), 0.0))
        # 5. Main effects orthogonal to every quadratic (x_j^2) column
        quad = x**2
        checks["main_vs_quadratic_orthogonal"] = bool(np.allclose(x.T @ quad, 0.0))
        return checks


def build_dsd(preset: str = "in_distribution") -> DSDesign:
    return DSDesign(preset)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate a DSD over the airfoil input space.")
    ap.add_argument("--preset", default="in_distribution", choices=list(PRESETS))
    ap.add_argument("--out", default="output/stage1/dsd_design.csv")
    ap.add_argument("--json", default="output/stage1/dsd_design.json")
    args = ap.parse_args()

    d = build_dsd(args.preset)
    print(f"Preset: {args.preset}   runs: {len(d.runs)}   conference order: {d.n_conf} "
          f"({d.n_ghost} ghost factor(s))")
    print("\nFactor bounds (coded -1 / 0 / +1):")
    for f in d.factors:
        print(f"  {f.name:9s} {f.low:>8g} / {f.center:>8g} / {f.high:>8g}  [{f.unit}]")
    print("\nRuns:")
    print(f"  {'#':>2} {'NACA':>5} {'Re':>10} {'AoA':>6}  OOD")
    for r in d.runs:
        print(f"  {r.run_id:>2} {r.naca_code:>5} {r.reynolds:>10.0f} {r.aoa:>6.1f}  {'*' if r.ood else ''}")
    print("\nVerification:")
    for k, v in d.verify().items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    csv_path = d.write_csv(args.out)
    json_path = d.write_json(args.json)
    print(f"\nWrote {csv_path}\nWrote {json_path}")
