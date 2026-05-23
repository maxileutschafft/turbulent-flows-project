"""C-mesh generation utilities for NACA 4-digit airfoils.

This is a trimmed copy of `physics_guard.utils.mesh` containing only the
node-building pieces required to generate a structured C-mesh in memory
(no connectivity, face data, or wall-distance helpers).
"""

import math

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# C-mesh constants (inlined from generate_c_mesh.py)
# ---------------------------------------------------------------------------

_CHORD = 1.0
_NU = 1e-5

_N_AIRFOIL  = 241
_N_WAKE     = 120
_N_LAYERS   = 120
_YPLUS      = 0.8
_RE_DESIGN  = 5.0e5
_X_OUTLET   = 25.0
_D_FAR      = 20.0

_CMESH_MATCH_TOL = 5e-3   # max distance between NPZ node and C-mesh cell center


def _naca4_params(code: str) -> tuple[float, float, float]:
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"NACA code must be 4 digits, got {code!r}")
    return int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0


def naca4_closed_te(naca_code: str, n_points: int = 121, chord: float = _CHORD) -> np.ndarray:
    """Closed-TE NACA 4-digit surface coordinates, shape [2*n_points-1, 2].

    Uses the -0.1036 trailing-edge coefficient (closed TE, no concave corner)
    — this is the variant used by generate_c_mesh.py, NOT the open-TE version
    in utils.naca_geometry.
    """
    m, p, t = _naca4_params(naca_code)
    beta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1036 * x ** 4
    )
    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            (m / p ** 2) * (2.0 * p * x - x ** 2),
            (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2),
        )
        dyc_dx = np.where(
            x < p,
            (2.0 * m / p ** 2) * (p - x),
            (2.0 * m / (1.0 - p) ** 2) * (p - x),
        )
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta);  yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta);  yl = yc - yt * np.cos(theta)
    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    return np.vstack([upper, lower]) * chord


def _first_cell_height(yplus: float, re: float, U: float, nu: float) -> float:
    cf = 0.058 * re ** (-0.2)
    u_tau = U * np.sqrt(0.5 * cf)
    return float(yplus * nu / u_tau)


def _solve_growth(L: float, y1: float, n_cells: int) -> float:
    if y1 * n_cells >= L * (1.0 - 1e-12):
        return 1.0
    return float(brentq(
        lambda r: y1 * (r ** n_cells - 1.0) / (r - 1.0) - L,
        1.0 + 1e-12, 5.0, xtol=1e-12,
    ))


def _geom_nodes(L: float, y1: float, n_cells: int) -> np.ndarray:
    r = _solve_growth(L, y1, n_cells)
    s = np.zeros(n_cells + 1)
    h = y1
    for k in range(n_cells):
        s[k + 1] = s[k] + h
        h *= r
    if s[-1] != 0.0:
        s *= L / s[-1]
    return s


def build_c_mesh_nodes(
    naca_code: str,
    aoa_deg: float = 0.0,
    *,
    n_airfoil_target: int = _N_AIRFOIL,
    n_wake: int = _N_WAKE,
    n_layers: int = _N_LAYERS,
    x_outlet: float = _X_OUTLET,
    d_far: float = _D_FAR,
    yplus_target: float = _YPLUS,
    re_design: float = _RE_DESIGN,
    chord: float = _CHORD,
    nu: float = _NU,
) -> tuple[np.ndarray, int, int]:
    """Return (nodes [ni, nj, 2], n_airfoil, n_wake) for the structured C-mesh."""
    n_half = (n_airfoil_target + 1) // 2
    airfoil = naca4_closed_te(naca_code, n_points=n_half, chord=chord)
    n_airfoil = airfoil.shape[0]

    m_inf = math.tan(math.radians(aoa_deg))

    U_design = re_design * nu / chord
    y1 = _first_cell_height(yplus_target, re_design, U_design, nu)

    te_sp = 0.5 * (
        float(np.linalg.norm(airfoil[1] - airfoil[0]))
        + float(np.linalg.norm(airfoil[-1] - airfoil[-2]))
    )
    s_te = _geom_nodes(x_outlet - 1.0, te_sp, n_wake - 1)
    x_wake = 1.0 + s_te
    y_wake = m_inf * s_te

    wake_top = np.column_stack([x_wake[::-1], y_wake[::-1]])
    wake_bot = np.column_stack([x_wake,        y_wake])
    inner = np.vstack([wake_top, airfoil[1:-1], wake_bot])
    ni = inner.shape[0]

    chord_mid = 0.5
    ff = np.empty_like(inner)
    fr_top = np.linspace(0.0, 1.0, n_wake)
    ff[:n_wake, 0] = x_outlet - (x_outlet - chord_mid) * fr_top
    ff[:n_wake, 1] = d_far
    n_int = n_airfoil - 2
    fr_arc = (np.arange(n_int) + 1.0) / (n_airfoil - 1)
    th = 0.5 * np.pi + fr_arc * np.pi
    ff[n_wake:n_wake + n_int, 0] = chord_mid + d_far * np.cos(th)
    ff[n_wake:n_wake + n_int, 1] =             d_far * np.sin(th)
    fr_bot = np.linspace(0.0, 1.0, n_wake)
    ff[n_wake + n_int:, 0] = chord_mid + (x_outlet - chord_mid) * fr_bot
    ff[n_wake + n_int:, 1] = -d_far

    nj = n_layers + 1
    nodes = np.empty((ni, nj, 2))
    for i in range(ni):
        L = float(np.linalg.norm(ff[i] - inner[i]))
        s = _geom_nodes(L, y1, n_layers)
        eta = s / max(L, 1e-30)
        nodes[i, :, 0] = (1.0 - eta) * inner[i, 0] + eta * ff[i, 0]
        nodes[i, :, 1] = (1.0 - eta) * inner[i, 1] + eta * ff[i, 1]
    return nodes, n_airfoil, n_wake
