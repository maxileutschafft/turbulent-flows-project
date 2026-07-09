"""Shared field-error and force-integration helpers for Stage 0 and Stage 1.

Kept backend-agnostic (pure numpy): it works on any dict of node fields that
follows the dataset / prediction `.npz` schema (x, y, p, ... plus a surface
locator `is_wall` or `sdf`). Both the GNO prediction and the CFD ground truth
are run through the SAME functions, so any systematic bias of the integrator
cancels in the surrogate-vs-truth comparison.
"""

from __future__ import annotations

import numpy as np

CHORD = 1.0
NU = 1.0e-5  # kinematic viscosity used in the dataset (mesh.py / scenario.py)


# --------------------------------------------------------------------------- #
# Field error
# --------------------------------------------------------------------------- #


def nrmse(pred: np.ndarray, truth: np.ndarray, *, eps: float = 1e-12) -> float:
    """Normalised RMSE: RMSE(pred, truth) / (max(truth) - min(truth))."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    rng = float(np.ptp(truth))
    return rmse / (rng + eps)


def relative_l2(pred: np.ndarray, truth: np.ndarray, *, eps: float = 1e-12) -> float:
    """Relative L2 error ||pred - truth|| / ||truth||."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + eps))


# --------------------------------------------------------------------------- #
# Surface extraction
# --------------------------------------------------------------------------- #


def surface_indices(fields: dict) -> np.ndarray:
    """Ordered node indices tracing the airfoil surface.

    Prefers ``is_wall`` (structured C-mesh wall cells are already stored in
    surface order, so we keep their ascending index order -> a clean loop).
    Falls back to the innermost ``|sdf|`` ring ordered by angle (less robust).
    """
    if "is_wall" in fields:
        idx = np.where(np.asarray(fields["is_wall"]).astype(bool))[0]
        return np.sort(idx)  # ascending flat index == surface traversal order
    if "sdf" in fields:
        sdf = np.abs(np.asarray(fields["sdf"], dtype=float))
        thr = np.quantile(sdf, 0.02)
        idx = np.where(sdf <= thr)[0]
        x = np.asarray(fields["x"])[idx]
        y = np.asarray(fields["y"])[idx]
        ang = np.arctan2(y - y.mean(), x - x.mean())
        return idx[np.argsort(ang)]
    raise KeyError("fields need 'is_wall' or 'sdf' to locate the airfoil surface")


# --------------------------------------------------------------------------- #
# Force coefficients (pressure-based)
# --------------------------------------------------------------------------- #


def cl_cd_from_fields(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    surf_idx: np.ndarray,
    reynolds: float,
    aoa_deg: float,
    *,
    rho: float = 1.0,
) -> tuple[float, float]:
    """Pressure-based Cl, Cd by integrating p over the ordered surface loop.

    Outward normals are auto-oriented (pointing away from the surface centroid),
    so the result is independent of the loop's winding direction. Friction drag
    is NOT included -> Cd is biased low; use it consistently on both prediction
    and truth. ``p`` is treated as kinematic pressure (p/rho) with rho=1 by
    default; both sides use the same convention so the comparison is unaffected.
    """
    xs = np.asarray(x)[surf_idx].astype(float)
    ys = np.asarray(y)[surf_idx].astype(float)
    ps = np.asarray(p)[surf_idx].astype(float)

    # panels: consecutive surface points, closing the loop (last -> first)
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    dx, dy = xn - xs, yn - ys
    nx, ny = dy, -dx                       # normal * panel length (unnormalised)
    mx, my = 0.5 * (xs + xn), 0.5 * (ys + yn)   # panel midpoints
    pmid = 0.5 * (ps + np.roll(ps, -1))

    # orient normals outward (away from centroid)
    cx, cy = xs.mean(), ys.mean()
    if np.sum(nx * (mx - cx) + ny * (my - cy)) < 0:
        nx, ny = -nx, -ny

    # pressure force on the body = -integral p n dA
    fx = -np.sum(pmid * nx)
    fy = -np.sum(pmid * ny)

    u_mag = reynolds * NU / CHORD
    q = 0.5 * rho * u_mag**2 * CHORD
    a = np.radians(aoa_deg)
    lift = -fx * np.sin(a) + fy * np.cos(a)
    drag = fx * np.cos(a) + fy * np.sin(a)
    return lift / q, drag / q


def cl_cd_from_scenario_fields(fields: dict, p: np.ndarray, *, rho: float = 1.0) -> tuple[float, float]:
    """Convenience wrapper: pull x/y/meta from a scenario dict, integrate ``p``."""
    idx = surface_indices(fields)
    return cl_cd_from_fields(
        fields["x"], fields["y"], p, idx,
        float(fields["reynolds"]), float(fields["angle_of_attack"]), rho=rho,
    )


def cl_cd_from_wall(
    wall_p: np.ndarray,
    wall_normal: np.ndarray,
    wall_length: np.ndarray,
    u_mag: float,
    aoa_deg: float,
    *,
    wall_shear: np.ndarray | None = None,
) -> tuple[float, float]:
    """Accurate Cl, Cd from the dataset's wall-face arrays (kinematic units).

    Uses the surface tables described in the dataset README: pressure force
    ``-p n dA`` plus, if ``wall_shear`` (kinematic tau_w vector) is given, the
    viscous force ``tau_w dA``. ``wall_normal`` points from the wall into the
    fluid; ``q = 0.5 u_mag^2`` is the kinematic dynamic pressure; chord = 1.

    Pass ``wall_shear=None`` (e.g. for a GNO prediction that has no wall-shear
    output) to get a pressure-only estimate -> Cd biased low.
    """
    wall_p = np.asarray(wall_p, dtype=float)
    n = np.asarray(wall_normal, dtype=float)
    dl = np.asarray(wall_length, dtype=float)[:, None]

    f_press = -np.sum(wall_p[:, None] * n * dl, axis=0)          # pressure force
    f = f_press.copy()
    if wall_shear is not None:
        f = f + np.sum(np.asarray(wall_shear, dtype=float) * dl, axis=0)  # + viscous

    q = 0.5 * float(u_mag) ** 2
    a = np.radians(aoa_deg)
    lift = -f[0] * np.sin(a) + f[1] * np.cos(a)
    drag = f[0] * np.cos(a) + f[1] * np.sin(a)
    return lift / q, drag / q


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


def cl_cd_from_wall_faces(
    x: np.ndarray,
    y: np.ndarray,
    field_p: np.ndarray,
    wall_xy: np.ndarray,
    wall_normal: np.ndarray,
    wall_length: np.ndarray,
    u_mag: float,
    aoa_deg: float,
) -> tuple[float, float]:
    """Cl, Cd from a VOLUME pressure field sampled onto the wall faces.

    The dataset's ``wall_cell`` indexes the *uncropped* mesh, so we map each
    wall face to its nearest stored cell geometrically (KD-tree on x,y). Use the
    SAME function on the GNO prediction and on the truth field: the absolute
    values won't match the dataset's stored cl/cd (different forceCoeffs
    convention + open trailing edge), but that convention cancels in the
    prediction-vs-truth comparison, which is what the surrogate test needs.
    """
    from scipy.spatial import cKDTree

    xy = np.column_stack([np.asarray(x), np.asarray(y)])
    _, nn = cKDTree(xy).query(np.asarray(wall_xy))
    return cl_cd_from_wall(np.asarray(field_p)[nn], wall_normal, wall_length, u_mag, aoa_deg)


def parse_naca(code: str) -> tuple[int, int, int]:
    """(m, p, t) integer digits from a 4-digit NACA code string."""
    code = str(code)
    return int(code[0]), int(code[1]), int(code[2:])
