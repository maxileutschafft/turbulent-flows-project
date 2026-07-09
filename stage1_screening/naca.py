"""NACA-4 surface coordinates in airfoil (Selig) order for XFOIL / NeuralFoil.

Mirrors ``src/utils/naca_geometry.py`` (closed-TE thickness variant) but returns
a single ordered (N, 2) loop: upper surface TE->LE, then lower surface LE->TE.
Kept standalone so the Stage-1 tools have no dependency on the package layout.
"""

from __future__ import annotations

import numpy as np


def naca4_selig(code: str, n: int = 160) -> np.ndarray:
    """Return ordered (2n-1, 2) surface coordinates, TE(upper) -> LE -> TE(lower)."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Expected a 4-digit NACA code, got {code!r}")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    # cosine spacing clusters points at LE and TE for a clean panel geometry
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))

    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        theta = np.zeros_like(x)
    else:
        yc = np.where(x <= p, m / p**2 * (2 * p * x - x**2),
                      m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2))
        dyc = np.where(x <= p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
        theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = np.column_stack([xu[::-1], yu[::-1]])   # TE -> LE
    lower = np.column_stack([xl[1:], yl[1:]])       # LE -> TE (skip duplicate LE)
    return np.vstack([upper, lower])


def write_dat(code: str, path, n: int = 160) -> str:
    """Write a Selig .dat file XFOIL can LOAD. Returns the path as str."""
    coords = naca4_selig(code, n=n)
    with open(path, "w") as fh:
        fh.write(f"NACA {code}\n")
        for xx, yy in coords:
            fh.write(f"{xx:.6f} {yy:.6f}\n")
    return str(path)
