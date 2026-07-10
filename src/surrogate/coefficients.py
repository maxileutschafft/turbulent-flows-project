"""Aerodynamic force coefficients (Cl/Cd) from a surrogate prediction.

Computed directly from the near-body airfoil-wall boundary using the exact
structured C-mesh geometry that inference already builds
(``utils.mesh.build_c_mesh_nodes``) — the same adjacent-cell, face-area
(``Sf``) and geometric wall-normal-distance (``ndc`` = 1/deltaCoeffs) data
OpenFOAM's own ``forces``/``forceCoeffs`` functionObject uses internally.

This is a self-contained port of the coefficient path from the companion
``physics-control-loop`` project (``surrogate.coefficients`` +
``physics_oracle.of_residuals.mesh_geometry.build_airfoil_wall_patch`` +
``physics_oracle.of_residuals.inlet.lift_drag_axes``), collapsed into one
module so the web app can report Cl/Cd without pulling in the OpenFOAM /
physics-oracle stack. The wall-patch geometry is derived from the same
``build_c_mesh_nodes`` node grid used by ``utils.scenario.build_scenario``, so
the cell indexing (``i*(nj-1)+j``) is bit-consistent with the cropped
prediction's ``mesh_indices``.

Validated in the source project against a fresh OpenFOAM run's own
``forceCoeffs`` output (NACA2210, Re=174200, AoA=-1.6): Cl within 0.12%, Cd
within 0.06%.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from utils.mesh import build_c_mesh_nodes

# Pseudo-3D slab depth used when the 2-D C-mesh is extruded one cell in z.
# Matches the source project's ``EXTRUSION_DZ`` so Sf/magSf carry the same
# span factor and the per-unit-span coefficients below are consistent.
EXTRUSION_DZ = 0.01

_TINY = 1.0e-10


# ---------------------------------------------------------------------------
# Projection axes (ported from physics_oracle.of_residuals.inlet)
# ---------------------------------------------------------------------------

def lift_drag_axes(aoa_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors ``(e_drag, e_lift)`` for projecting a force onto the
    drag/lift axes at this angle of attack."""
    alpha = math.radians(aoa_deg)
    e_drag = np.array([math.cos(alpha), math.sin(alpha)])
    e_lift = np.array([-math.sin(alpha), math.cos(alpha)])
    return e_drag, e_lift


# ---------------------------------------------------------------------------
# Cell centres (ported from utils.meshing.cell_centres in the source project)
# ---------------------------------------------------------------------------

def cell_centres(nodes: np.ndarray) -> np.ndarray:
    """Area-weighted (shoelace) cell centres for the C-mesh quad grid.

    Matches the cell centres OpenFOAM reports. The Gmsh ``.msh`` writer
    serialises node coordinates as ``%.10e``, so the same rounding is applied
    here first for bit-accuracy against the OpenFOAM-derived geometry.

    Returns ``((ni-1)*(nj-1), 2)`` float64 centroids in raster order
    ``i*(nj-1)+j``.
    """
    _fmt10e = np.frompyfunc(lambda v: float("%.10e" % v), 1, 1)
    nodes_r = _fmt10e(nodes).astype(np.float64)

    x0 = nodes_r[:-1, :-1, 0]
    y0 = nodes_r[:-1, :-1, 1]
    x1 = nodes_r[1:, :-1, 0]
    y1 = nodes_r[1:, :-1, 1]
    x2 = nodes_r[1:, 1:, 0]
    y2 = nodes_r[1:, 1:, 1]
    x3 = nodes_r[:-1, 1:, 0]
    y3 = nodes_r[:-1, 1:, 1]

    cross01 = x0 * y1 - x1 * y0
    cross12 = x1 * y2 - x2 * y1
    cross23 = x2 * y3 - x3 * y2
    cross30 = x3 * y0 - x0 * y3

    two_A = cross01 + cross12 + cross23 + cross30

    cx = ((x0 + x1) * cross01
          + (x1 + x2) * cross12
          + (x2 + x3) * cross23
          + (x3 + x0) * cross30) / (3.0 * two_A)
    cy = ((y0 + y1) * cross01
          + (y1 + y2) * cross12
          + (y2 + y3) * cross23
          + (y3 + y0) * cross30) / (3.0 * two_A)

    cc = np.stack([cx.ravel(order="C"), cy.ravel(order="C")], axis=1)
    return cc.astype(np.float64)


# ---------------------------------------------------------------------------
# Boundary-face geometry (ported from mesh_geometry._boundary_face_geometry)
# ---------------------------------------------------------------------------

def _boundary_face_geometry(
    A: np.ndarray, B: np.ndarray, C_own: np.ndarray, dz: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Face-area vector, magnitude, centre, and boundary deltaCoeffs (ndc) for
    one boundary edge ``(A, B)`` (2D node coords) extruded to depth ``dz``,
    oriented outward from the owner cell centre ``C_own`` (3D)."""
    mid_xy = 0.5 * (A + B)
    Cf_3d = np.array([mid_xy[0], mid_xy[1], dz / 2])

    edge_len = float(np.linalg.norm(B - A))
    mag_sf = edge_len * dz

    e = B - A
    Sf_raw = np.array([e[1], -e[0], 0.0]) * dz

    # Orient outward: Sf . (Cf - C_owner) > 0
    if np.dot(Sf_raw, Cf_3d - C_own) < 0.0:
        Sf_raw = -Sf_raw

    unit_area = Sf_raw / mag_sf
    delta_b = Cf_3d - C_own
    delta_norm = float(np.linalg.norm(delta_b))
    dot_val = float(np.dot(unit_area, delta_b))
    ndc = 1.0 / max(dot_val, 0.05 * delta_norm)

    return Sf_raw, mag_sf, Cf_3d, ndc


# ---------------------------------------------------------------------------
# Airfoil-wall patch (ported from mesh_geometry.build_airfoil_wall_patch)
# ---------------------------------------------------------------------------

def build_airfoil_wall_patch(
    naca_code: str,
    aoa_deg: float,
    *,
    device="cpu",
    dtype=torch.float64,
) -> tuple[int, dict]:
    """Build the ``airfoilWalls`` boundary patch (cells, Sf, magSf, Cf, ndc).

    The airfoil-wall patch is always exactly the ``j=0`` row for
    ``i in [n_wake-1, n_wake+n_airfoil-2)``, matching the wall-mask
    classification in ``utils.scenario.build_scenario``. The mesh nodes come
    from the same ``build_c_mesh_nodes`` grid, so the returned cell indices
    (``i*(nj-1)+j``) line up with the cropped prediction's ``mesh_indices``.

    Returns ``(n_cells, patch)`` — total full-mesh cell count and a patch dict.
    """
    nodes, n_airfoil, n_wake = build_c_mesh_nodes(naca_code, aoa_deg=aoa_deg)
    ni, nj, _ = nodes.shape
    n_cells = (ni - 1) * (nj - 1)

    _fmt10e = np.frompyfunc(lambda v: float("%.10e" % v), 1, 1)
    nodes_r = _fmt10e(nodes).astype(np.float64)
    cc_2d = cell_centres(nodes)   # (n_cells, 2), %.10e rounding applied inside

    dz = EXTRUSION_DZ
    j = 0
    i_lo, i_hi = n_wake - 1, n_wake + n_airfoil - 2

    cells, Sf_list, magSf_list, Cf_list, ndc_list = [], [], [], [], []
    for i in range(i_lo, i_hi):
        c = i * (nj - 1) + j
        A, B = nodes_r[i, 0], nodes_r[i + 1, 0]
        C_own = np.array([cc_2d[c, 0], cc_2d[c, 1], dz / 2])
        Sf_raw, mag_sf, Cf_3d, ndc_b = _boundary_face_geometry(A, B, C_own, dz)

        cells.append(c)
        Sf_list.append(Sf_raw)
        magSf_list.append(mag_sf)
        Cf_list.append(Cf_3d)
        ndc_list.append(ndc_b)

    def _t(arr, long: bool = False) -> torch.Tensor:
        t = torch.tensor(np.asarray(arr), dtype=torch.long if long else dtype)
        return t.to(device)

    patch = {
        "name": "airfoilWalls",
        "type": "wall",
        "cells": _t(np.array(cells, dtype=np.int64), long=True),
        "Sf": _t(np.array(Sf_list, dtype=np.float64)),
        "magSf": _t(np.array(magSf_list, dtype=np.float64)),
        "Cf": _t(np.array(Cf_list, dtype=np.float64)),
        "ndc": _t(np.array(ndc_list, dtype=np.float64)),
    }
    return n_cells, patch


# ---------------------------------------------------------------------------
# Coefficient integral (ported from surrogate.coefficients.compute_coefficients)
# ---------------------------------------------------------------------------

def compute_coefficients(
    n_cells: int,
    wall_patch: dict,
    mesh_indices: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    *,
    aoa_deg: float,
    reynolds: float,
    nu: float = 1e-5,
) -> dict:
    """Near-body Cl/Cd from the exact airfoil-wall boundary geometry.

    Parameters
    ----------
    n_cells       : total full-mesh cell count (from ``build_airfoil_wall_patch``).
    wall_patch    : the ``airfoilWalls`` patch dict (``cells``, ``Sf``,
                    ``magSf``, ``ndc``).
    mesh_indices  : (N_cropped,) int64 — full-mesh cell indices of the cropped
                    prediction (row i of ``u``/``v``/``p`` is cell
                    ``mesh_indices[i]``). Must cover every wall-patch cell.
    u, v, p       : (N_cropped,) predicted velocity / kinematic-pressure at the
                    cropped cells, same row order as ``mesh_indices``.
    aoa_deg, reynolds : scenario parameters (drive freestream + projection axes).
    nu            : kinematic viscosity (default 1e-5 m²/s).

    Returns
    -------
    dict with keys ``cl``, ``cd``, ``cd_pressure``, ``cd_friction``, ``ld``.
    """
    mesh_indices = np.asarray(mesh_indices, dtype=np.int64)

    wall_cells = wall_patch["cells"].detach().cpu().numpy()

    # Map each wall-adjacent full-mesh cell to its row in the cropped arrays.
    full_to_row = np.full(int(n_cells), -1, dtype=np.int64)
    full_to_row[mesh_indices] = np.arange(len(mesh_indices))
    wall_rows = full_to_row[wall_cells]
    if np.any(wall_rows < 0):
        n_missing = int(np.sum(wall_rows < 0))
        raise ValueError(
            f"{n_missing}/{len(wall_cells)} airfoil-wall-adjacent cells are not "
            "covered by mesh_indices; cannot compute wall-boundary coefficients."
        )

    p_wall = np.asarray(p, dtype=np.float64)[wall_rows]        # (Nw,)
    u_wall = np.asarray(u, dtype=np.float64)[wall_rows]        # (Nw,)
    v_wall = np.asarray(v, dtype=np.float64)[wall_rows]        # (Nw,)

    Sf = wall_patch["Sf"].detach().cpu().numpy().astype(np.float64)[:, :2]     # (Nw,2)
    magSf = wall_patch["magSf"].detach().cpu().numpy().astype(np.float64)      # (Nw,)
    ndc = wall_patch["ndc"].detach().cpu().numpy().astype(np.float64)          # (Nw,)

    nhat = Sf / magSf[:, None]
    that = np.column_stack([-nhat[:, 1], nhat[:, 0]])
    u_tang = u_wall * that[:, 0] + v_wall * that[:, 1]

    # nutLowReWallFunction -> nut=0 at the wall; molecular viscosity only.
    # snGrad-consistent wall shear: tau = nu * ndc * (U_wall_tang(=0) - U_adjacent_tang)
    tau_w = nu * ndc * u_tang                                            # (Nw,)

    inv_dz = 1.0 / EXTRUSION_DZ

    f_pressure = (p_wall[:, None] * Sf).sum(axis=0) * inv_dz             # (2,) force ON the airfoil
    f_friction = (tau_w[:, None] * that * magSf[:, None]).sum(axis=0) * inv_dz

    F_total = f_pressure + f_friction

    e_drag, e_lift = lift_drag_axes(aoa_deg)

    D_pressure = float(np.dot(f_pressure, e_drag))
    D_friction = float(np.dot(f_friction, e_drag))
    L = float(np.dot(F_total, e_lift))
    D = D_pressure + D_friction

    U_inf = reynolds * nu
    q = 0.5 * U_inf ** 2

    cl = L / q
    cd = D / q
    cd_pressure = D_pressure / q
    cd_friction = D_friction / q
    ld = cl / cd if abs(cd) > _TINY else 0.0

    return {
        "cl": float(cl),
        "cd": float(cd),
        "cd_pressure": float(cd_pressure),
        "cd_friction": float(cd_friction),
        "ld": float(ld),
    }
