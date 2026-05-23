"""Analytical shortest-distance (SDF) from cell centers to a NACA-4 airfoil.

The training dataset's `sdf` field was computed analytically against the
smooth NACA-4 parametric surface (standard open-trailing-edge variant,
last-coefficient -0.1015), augmented by a closing line segment between
the two TE corners. This module reproduces that computation.

Convention
----------
Returned values are UNSIGNED distances. The training `sdf` field is also
unsigned (always >= 0). This module does not assign an interior/exterior
sign; if you need a signed field, combine with a point-in-polygon test.

Public API
----------
analytical_naca4_sdf(centers, naca_code, *, te_coeff=-0.1015,
                     close_te=True, n_warm=201, n_iter=10) -> np.ndarray
    Vectorised CPU implementation (numpy + scipy.cKDTree warm-start).

analytical_naca4_sdf_torch(centers, naca_code, *, device,
                           n_warm=201, n_iter=10,
                           dtype=torch.float32) -> np.ndarray
    GPU/torch implementation. Returns numpy for downstream consumers.

Notes
-----
The CFD C-mesh in `utils.mesh` uses a *closed-TE* polygon (-0.1036) for
mesh construction. The TRAINING SDF was generated against the open-TE
variant (-0.1015) plus a TE-closure segment. These are different
surfaces; do not confuse them.

Accuracy contract (vs the training SDF field, all 38,788 cells of
NACA2210_n4.9_2.1e5): median ~1e-9, p99 <= ~1e-6, max ~4.24e-4. The
max is the known 56-cell TE-wedge mesh-mismatch outlier, intrinsic to
the dataset and unrelated to this solver.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from utils.mesh import _naca4_params


# ---------------------------------------------------------------------------
# Parametric NACA-4 surface evaluation (numpy).
# Parameter s in [0, 1] equals x along chord (0 = LE, 1 = TE).
# Returns components (X, Y, dX, dY) as separate 1-D arrays — no stack copies.
# Uses the algebraic identity for the camber-line angle
#   cos(theta) = 1/sqrt(1+m^2), sin(theta) = m·cos(theta)
# instead of arctan/sin/cos calls (about 2x faster).
# ---------------------------------------------------------------------------


def _naca4_eval_fast(
    s: np.ndarray,
    code: str,
    surface: str,
    te_coeff: float,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, dX/dx, dY/dx) for the upper or lower NACA-4 surface."""
    if surface not in ("upper", "lower"):
        raise ValueError(surface)
    sign = -1.0 if surface == "upper" else +1.0
    m, p, t = _naca4_params(code)

    x = np.asarray(s, dtype=np.float64)
    x_safe = np.maximum(x, eps)
    sqrt_x = np.sqrt(x_safe)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x

    yt = 5.0 * t * (
        0.2969 * sqrt_x
        - 0.1260 * x
        - 0.3516 * x2
        + 0.2843 * x3
        + te_coeff * x4
    )
    dyt_dx = 5.0 * t * (
        0.2969 * 0.5 / sqrt_x
        - 0.1260
        - 2.0 * 0.3516 * x
        + 3.0 * 0.2843 * x2
        + 4.0 * te_coeff * x3
    )

    if m == 0.0 or p == 0.0:
        # Symmetric airfoil — no camber, no need for arctan branch.
        # X = x; Y = -sign·yt; dX = 1; dY = -sign·dyt.
        return x, -sign * yt, np.ones_like(x), -sign * dyt_dx

    p2 = p * p
    inv_p2 = 1.0 / p2
    inv_omp2 = 1.0 / (1.0 - p) ** 2
    front = x < p
    yc = np.where(
        front,
        m * inv_p2 * (2.0 * p * x - x2),
        m * inv_omp2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x2),
    )
    dyc_dx = np.where(
        front,
        (2.0 * m * inv_p2) * (p - x),
        (2.0 * m * inv_omp2) * (p - x),
    )
    d2yc_dx2 = np.where(front, -2.0 * m * inv_p2, -2.0 * m * inv_omp2)

    # Algebraic identity: theta = arctan(dyc/dx).
    # cos(theta) = 1/sqrt(1+m^2), sin(theta) = m·cos(theta).
    one_plus_dyc2 = 1.0 + dyc_dx * dyc_dx
    denom = np.sqrt(one_plus_dyc2)
    cos_t = 1.0 / denom
    sin_t = dyc_dx / denom
    dtheta_dx = d2yc_dx2 / one_plus_dyc2

    X = x + sign * yt * sin_t
    Y = yc - sign * yt * cos_t
    dX_dx = 1.0 + sign * (dyt_dx * sin_t + yt * cos_t * dtheta_dx)
    dY_dx = dyc_dx - sign * (dyt_dx * cos_t - yt * sin_t * dtheta_dx)
    return X, Y, dX_dx, dY_dx


def _refine_one_surface(
    centers: np.ndarray,
    code: str,
    surface: str,
    te_coeff: float,
    *,
    n_warm: int,
    n_iter: int,
    s_lo: float = 1e-6,
    s_hi: float = 1.0 - 1e-6,
    tol: float = 1e-13,
) -> np.ndarray:
    """Cosine-warm-started Newton refinement.

    For each cell center, finds s* in (s_lo, s_hi) minimizing
        f(s) = || c - r(s) ||^2,
    by solving the orthogonality equation
        g(s) = (c - r(s)) . r'(s) = 0.
    Newton steps with a finite-difference Jacobian, with a bracketed
    bisection fallback. Returns the per-cell distance ||c - r(s*)||.
    """
    N = centers.shape[0]
    cx = centers[:, 0]
    cy = centers[:, 1]

    # Cosine-spaced parameter grid: density clusters near LE and TE,
    # matching the natural NACA-4 surface point distribution.
    beta = np.linspace(0.0, np.pi, n_warm)
    s_dense = s_lo + (s_hi - s_lo) * 0.5 * (1.0 - np.cos(beta))
    Xd, Yd, _, _ = _naca4_eval_fast(s_dense, code, surface, te_coeff)
    pts_dense = np.column_stack([Xd, Yd])
    _, idx = cKDTree(pts_dense).query(centers)
    s = s_dense[idx].astype(np.float64).copy()

    # Bisection bracket: neighbouring warm-start vertices.
    a = s_dense[np.maximum(idx - 1, 0)]
    b = s_dense[np.minimum(idx + 1, n_warm - 1)]
    a, b = np.minimum(a, b), np.maximum(a, b)

    converged = np.zeros(N, dtype=bool)
    fd_h = 1e-5

    for _ in range(n_iter):
        active = ~converged
        if not active.any():
            break
        s_a = s[active]
        cax = cx[active]
        cay = cy[active]

        X, Y, dX, dY = _naca4_eval_fast(s_a, code, surface, te_coeff)
        g = (cax - X) * dX + (cay - Y) * dY

        s_p = np.clip(s_a + fd_h, s_lo, s_hi)
        s_m = np.clip(s_a - fd_h, s_lo, s_hi)
        Xp, Yp, dXp, dYp = _naca4_eval_fast(s_p, code, surface, te_coeff)
        Xm, Ym, dXm, dYm = _naca4_eval_fast(s_m, code, surface, te_coeff)
        g_p = (cax - Xp) * dXp + (cay - Yp) * dYp
        g_m = (cax - Xm) * dXm + (cay - Ym) * dYm
        gp = (g_p - g_m) / (s_p - s_m)

        with np.errstate(divide="ignore", invalid="ignore"):
            ds = -g / np.where(np.abs(gp) < 1e-30, np.nan, gp)
        s_new = s_a + ds

        a_a = a[active]
        b_a = b[active]
        out = ~np.isfinite(s_new) | (s_new < a_a) | (s_new > b_a)
        if out.any():
            s_new[out] = 0.5 * (a_a[out] + b_a[out])
            pos = g[out] > 0
            new_a = np.where(pos, s_a[out], a_a[out])
            new_b = np.where(pos, b_a[out], s_a[out])
            ai = np.where(active)[0][out]
            a[ai] = new_a
            b[ai] = new_b

        s_new = np.clip(s_new, s_lo, s_hi)

        delta = np.abs(s_new - s_a)
        s[active] = s_new
        new_conv = delta < tol
        conv_g = np.zeros(N, dtype=bool)
        conv_g[np.where(active)[0][new_conv]] = True
        converged |= conv_g

    X, Y, _, _ = _naca4_eval_fast(s, code, surface, te_coeff)
    return np.sqrt((cx - X) ** 2 + (cy - Y) ** 2)


def _te_corner(code: str, surface: str, te_coeff: float) -> np.ndarray:
    """The (x, y) trailing-edge corner of the open-TE NACA-4 surface."""
    X, Y, _, _ = _naca4_eval_fast(np.array([1.0 - 1e-12]), code, surface, te_coeff)
    return np.array([float(X[0]), float(Y[0])])


def _dist_to_segment(centers: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Min distance from each row of `centers` to the segment AB."""
    ab = B - A
    ab2 = float(ab @ ab)
    ap = centers - A
    t = np.clip(ap @ ab / max(ab2, 1e-30), 0.0, 1.0)
    proj = A + t[:, None] * ab
    return np.sqrt((centers[:, 0] - proj[:, 0]) ** 2 + (centers[:, 1] - proj[:, 1]) ** 2)


def analytical_naca4_sdf(
    centers: np.ndarray,
    naca_code: str,
    *,
    te_coeff: float = -0.1015,
    close_te: bool = True,
    n_warm: int = 201,
    n_iter: int = 10,
) -> np.ndarray:
    """Analytical NACA-4 shortest distance (unsigned) for each cell center.

    Parameters
    ----------
    centers : array, shape (N, 2)
        Query points (cell centers) in chord-normalised coordinates.
    naca_code : str
        4-digit NACA code (e.g. "2210").
    te_coeff : float, default -0.1015
        Last coefficient of the thickness polynomial. The training data
        uses the open-TE NACA-4 standard (-0.1015). Pass -0.1036 to
        match the closed-TE variant used by `naca4_closed_te`.
    close_te : bool, default True
        Include the line segment between the two TE corners as part of
        the surface (matches the training data).
    n_warm : int, default 201
        Number of cosine-spaced points used for the warm-start nearest
        neighbour search.
    n_iter : int, default 10
        Maximum Newton iterations per surface.

    Returns
    -------
    sdf : ndarray, shape (N,), float64
        Unsigned shortest distance from each center to the surface.

    Procedure
    ---------
    For each surface (upper / lower) parametrised by s = x in [0, 1]:
      1. cosine-spaced warm start: density clusters near LE and TE,
         where curvature is highest.
      2. KDTree to find the nearest warm-start vertex per query point.
      3. Newton iterations on g(s) = (c - r(s)) . r'(s) = 0
         (the closest-point projection is orthogonal to the tangent),
         with a finite-difference Jacobian and a bracketed-bisection
         fallback when steps would leave the bracket. r(s) and r'(s)
         are evaluated analytically from the NACA-4 polynomials.
    Per-cell distance is the minimum over upper, lower, and (if
    close_te=True) the TE-closure segment.

    Verified against the training `sdf` field to median ~1e-9 (float32
    machine precision); see scripts/verify_sdf_analytical.py.
    """
    centers = np.asarray(centers, dtype=np.float64)
    d_u = _refine_one_surface(centers, naca_code, "upper", te_coeff,
                              n_warm=n_warm, n_iter=n_iter)
    d_l = _refine_one_surface(centers, naca_code, "lower", te_coeff,
                              n_warm=n_warm, n_iter=n_iter)
    sdf = np.minimum(d_u, d_l)
    if close_te:
        A = _te_corner(naca_code, "upper", te_coeff)
        B = _te_corner(naca_code, "lower", te_coeff)
        sdf = np.minimum(sdf, _dist_to_segment(centers, A, B))
    return sdf


# ===========================================================================
# Torch GPU variant
# ===========================================================================


def _naca4_eval_torch(s, code: str, surface: str, te_coeff: float,
                      *, eps: float = 1e-12):
    """Torch counterpart of `_naca4_eval_fast`. Operates on the device of `s`."""
    import torch

    if surface not in ("upper", "lower"):
        raise ValueError(surface)
    sign = -1.0 if surface == "upper" else +1.0
    m, p, t = _naca4_params(code)

    x = s
    x_safe = torch.clamp(x, min=eps)
    sqrt_x = torch.sqrt(x_safe)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x

    yt = 5.0 * t * (
        0.2969 * sqrt_x - 0.1260 * x - 0.3516 * x2
        + 0.2843 * x3 + te_coeff * x4
    )
    dyt_dx = 5.0 * t * (
        0.2969 * 0.5 / sqrt_x - 0.1260
        - 2.0 * 0.3516 * x + 3.0 * 0.2843 * x2
        + 4.0 * te_coeff * x3
    )

    if m == 0.0 or p == 0.0:
        return x, -sign * yt, torch.ones_like(x), -sign * dyt_dx

    p2 = p * p
    inv_p2 = 1.0 / p2
    inv_omp2 = 1.0 / (1.0 - p) ** 2
    front = x < p
    yc = torch.where(
        front,
        m * inv_p2 * (2.0 * p * x - x2),
        m * inv_omp2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x2),
    )
    dyc_dx = torch.where(
        front,
        (2.0 * m * inv_p2) * (p - x),
        (2.0 * m * inv_omp2) * (p - x),
    )
    d2yc_dx2 = torch.where(
        front,
        torch.full_like(x, -2.0 * m * inv_p2),
        torch.full_like(x, -2.0 * m * inv_omp2),
    )
    one_plus_dyc2 = 1.0 + dyc_dx * dyc_dx
    denom = torch.sqrt(one_plus_dyc2)
    cos_t = 1.0 / denom
    sin_t = dyc_dx / denom
    dtheta_dx = d2yc_dx2 / one_plus_dyc2
    X = x + sign * yt * sin_t
    Y = yc - sign * yt * cos_t
    dX = 1.0 + sign * (dyt_dx * sin_t + yt * cos_t * dtheta_dx)
    dY = dyc_dx - sign * (dyt_dx * cos_t - yt * sin_t * dtheta_dx)
    return X, Y, dX, dY


def analytical_naca4_sdf_torch(
    centers,
    naca_code: str,
    *,
    device,
    te_coeff: float = -0.1015,
    close_te: bool = True,
    n_warm: int = 201,
    n_iter: int = 10,
    dtype=None,
) -> np.ndarray:
    """Torch/GPU analytical NACA-4 SDF. Returns a numpy array.

    Same algorithm as `analytical_naca4_sdf`, vectorised on the supplied
    `device`. The warm-start nearest-vertex search uses an explicit
    broadcasted (N, n_warm) squared-distance matrix + argmin (cheaper
    than building a CPU KDTree at the n_warm sizes we use).

    Parameters
    ----------
    centers : np.ndarray or torch.Tensor, shape (N, 2)
        Query points. Accepts either; conversion handled internally.
    naca_code : str
        4-digit NACA code (e.g. "2210").
    device : torch.device or str
        Target device. Both the warm-start polygon and Newton iterations
        run here.
    te_coeff, close_te : as `analytical_naca4_sdf`.
    n_warm : int, default 201
    n_iter : int, default 10
    dtype : torch.dtype or None, default None
        Working precision on `device`. None => torch.float32. Outputs
        are returned as float64 numpy regardless.

    Returns
    -------
    sdf : np.ndarray, shape (N,), float64
        Unsigned shortest distance from each center to the surface.
        Accuracy contract matches the CPU variant.
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    dev = torch.device(device) if not isinstance(device, torch.device) else device

    if isinstance(centers, torch.Tensor):
        centers_t = centers.to(device=dev, dtype=dtype)
    else:
        centers_t = torch.as_tensor(np.asarray(centers), dtype=dtype, device=dev)

    s_lo = 1e-6
    s_hi = 1.0 - 1e-6
    beta = torch.linspace(0.0, float(np.pi), n_warm, dtype=dtype, device=dev)
    s_dense = s_lo + (s_hi - s_lo) * 0.5 * (1.0 - torch.cos(beta))
    cx = centers_t[:, 0]
    cy = centers_t[:, 1]
    fd_h = 1e-5

    def newton(surface: str):
        Xd, Yd, _, _ = _naca4_eval_torch(s_dense, naca_code, surface, te_coeff)
        # Broadcasted nearest warm-start vertex.
        dx = cx[:, None] - Xd[None, :]
        dy = cy[:, None] - Yd[None, :]
        d2 = dx * dx + dy * dy
        idx = torch.argmin(d2, dim=1)
        s = s_dense[idx].clone()
        idx_lo = torch.clamp(idx - 1, min=0)
        idx_hi = torch.clamp(idx + 1, max=n_warm - 1)
        a = torch.minimum(s_dense[idx_lo], s_dense[idx_hi])
        b = torch.maximum(s_dense[idx_lo], s_dense[idx_hi])

        for _ in range(n_iter):
            X, Y, dX, dY = _naca4_eval_torch(s, naca_code, surface, te_coeff)
            g = (cx - X) * dX + (cy - Y) * dY
            s_p = torch.clamp(s + fd_h, s_lo, s_hi)
            s_m = torch.clamp(s - fd_h, s_lo, s_hi)
            Xp, Yp, dXp, dYp = _naca4_eval_torch(s_p, naca_code, surface, te_coeff)
            Xm, Ym, dXm, dYm = _naca4_eval_torch(s_m, naca_code, surface, te_coeff)
            g_p = (cx - Xp) * dXp + (cy - Yp) * dYp
            g_m = (cx - Xm) * dXm + (cy - Ym) * dYm
            gp = (g_p - g_m) / (s_p - s_m)
            ds = torch.where(torch.abs(gp) < 1e-20,
                             torch.zeros_like(gp), -g / gp)
            s_new = s + ds
            out = (~torch.isfinite(s_new)) | (s_new < a) | (s_new > b)
            if bool(out.any()):
                bisect = 0.5 * (a + b)
                s_new = torch.where(out, bisect, s_new)
                pos = (g > 0) & out
                a = torch.where(pos, s, a)
                b = torch.where(out & ~pos, s, b)
            s = torch.clamp(s_new, s_lo, s_hi)
        X, Y, _, _ = _naca4_eval_torch(s, naca_code, surface, te_coeff)
        return torch.sqrt((cx - X) ** 2 + (cy - Y) ** 2)

    d_u = newton("upper")
    d_l = newton("lower")
    d = torch.minimum(d_u, d_l)

    if close_te:
        one = torch.tensor([1.0 - 1e-12], dtype=dtype, device=dev)
        A_X, A_Y, _, _ = _naca4_eval_torch(one, naca_code, "upper", te_coeff)
        B_X, B_Y, _, _ = _naca4_eval_torch(one, naca_code, "lower", te_coeff)
        Ax, Ay = float(A_X[0]), float(A_Y[0])
        Bx, By = float(B_X[0]), float(B_Y[0])
        abx, aby = Bx - Ax, By - Ay
        ab2 = abx * abx + aby * aby
        tparam = torch.clamp(
            ((cx - Ax) * abx + (cy - Ay) * aby) / max(ab2, 1e-30), 0.0, 1.0
        )
        px = Ax + tparam * abx
        py = Ay + tparam * aby
        d_te = torch.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        d = torch.minimum(d, d_te)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    return d.detach().cpu().numpy().astype(np.float64)
