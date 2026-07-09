"""NACA 4-digit airfoil geometry using the standard analytical formula."""

from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np


def naca4_coords(code: str, n: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (xu, yu, xl, yl) surface coordinates for a NACA 4-digit airfoil.

    chord is normalised to [0, 1]; upper surface first (LE->TE), lower surface reversed (TE->LE).
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Expected a 4-digit NACA code, got {code!r}")

    m = int(code[0]) / 100.0   # max camber
    p = int(code[1]) / 10.0    # position of max camber
    t = int(code[2:]) / 100.0  # max thickness

    x = np.linspace(0.0, 1.0, n)

    # Thickness distribution (closed trailing edge variant)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        theta = np.zeros_like(x)
    else:
        yc = np.where(
            x <= p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2),
        )
        dyc_dx = np.where(
            x <= p,
            2 * m / p**2 * (p - x),
            2 * m / (1 - p) ** 2 * (p - x),
        )
        theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl


def naca4_polygon(code: str, n: int = 300) -> mpatches.Polygon:
    """Return a closed matplotlib Polygon for the NACA 4-digit airfoil silhouette."""
    xu, yu, xl, yl = naca4_coords(code, n=n)
    # Trace: upper surface LE->TE, lower surface TE->LE
    verts = np.column_stack([
        np.concatenate([xu, xl[::-1]]),
        np.concatenate([yu, yl[::-1]]),
    ])
    return mpatches.Polygon(verts, closed=True, facecolor="#a0a0a0", edgecolor="black", linewidth=0.8, zorder=3)
