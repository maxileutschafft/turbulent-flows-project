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
prediction's ``mesh_indices``. The source project builds the patch as torch
tensors (it shares an autograd geometry pipeline); here the integral is pure
numpy, so the patch is numpy throughout.

Validated in the source project against a fresh OpenFOAM run's own
``forceCoeffs`` output (NACA2210, Re=174200, AoA=-1.6): Cl within 0.12%, Cd
within 0.06%.
"""
from __future__ import annotations

import math

import numpy as np

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
# Wall-row cell centres
# ---------------------------------------------------------------------------

def _wall_cell_centroids(
    nodes_r: np.ndarray, i_lo: int, i_hi: int,
) -> np.ndarray:
    """Area-weighted (shoelace) centroids for the ``j=0`` cells ``i in [i_lo,
    i_hi)`` — the airfoil-wall row only.

    The full-mesh companion in the source project computes centroids for every
    one of the ~57k cells; the Cl/Cd surface integral only ever reads the ~240
    wall-adjacent cells, so this evaluates just that row. Matches the cell
    centres OpenFOAM reports (nodes are expected pre-rounded to ``%.10e``,
    as the Gmsh ``.msh`` writer serialises them).

    Cell ``(i, 0)`` spans nodes ``(i,0), (i+1,0), (i+1,1), (i,1)``; the shoelace
    winding and formula match the source ``cell_centres`` exactly. Degenerate
    (near-zero-area) cells fall back to the 4-corner mean instead of dividing
    by ~0, so a collapsed trailing-edge quad can never inject inf/nan into the
    downstream force integral.
    """
    x0, y0 = nodes_r[i_lo:i_hi, 0, 0], nodes_r[i_lo:i_hi, 0, 1]          # (i,   0)
    x1, y1 = nodes_r[i_lo + 1:i_hi + 1, 0, 0], nodes_r[i_lo + 1:i_hi + 1, 0, 1]  # (i+1, 0)
    x2, y2 = nodes_r[i_lo + 1:i_hi + 1, 1, 0], nodes_r[i_lo + 1:i_hi + 1, 1, 1]  # (i+1, 1)
    x3, y3 = nodes_r[i_lo:i_hi, 1, 0], nodes_r[i_lo:i_hi, 1, 1]          # (i,   1)

    c01 = x0 * y1 - x1 * y0
    c12 = x1 * y2 - x2 * y1
    c23 = x2 * y3 - x3 * y2
    c30 = x3 * y0 - x0 * y3
    two_A = c01 + c12 + c23 + c30

    safe = np.abs(two_A) > 1e-30
    den = np.where(safe, 3.0 * two_A, 1.0)   # avoid 0/0 in the discarded branch
    cx = ((x0 + x1) * c01 + (x1 + x2) * c12 + (x2 + x3) * c23 + (x3 + x0) * c30) / den
    cy = ((y0 + y1) * c01 + (y1 + y2) * c12 + (y2 + y3) * c23 + (y3 + y0) * c30) / den

    cx = np.where(safe, cx, 0.25 * (x0 + x1 + x2 + x3))
    cy = np.where(safe, cy, 0.25 * (y0 + y1 + y2 + y3))
    return np.column_stack([cx, cy]).astype(np.float64)


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
    mesh: tuple[np.ndarray, int, int] | None = None,
) -> tuple[int, dict]:
    """Build the ``airfoilWalls`` boundary patch (cells, Sf, magSf, Cf, ndc).

    The airfoil-wall patch is always exactly the ``j=0`` row for
    ``i in [n_wake-1, n_wake+n_airfoil-2)``, matching the wall-mask
    classification in ``utils.scenario.build_scenario``. The mesh nodes come
    from the same ``build_c_mesh_nodes`` grid, so the returned cell indices
    (``i*(nj-1)+j``) line up with the cropped prediction's ``mesh_indices``.

    Parameters
    ----------
    mesh : ``(nodes, n_airfoil, n_wake)`` or None
        Pre-built C-mesh from ``build_c_mesh_nodes``. Pass the same tuple used
        for ``build_scenario`` to avoid rebuilding the mesh a second time per
        prediction; when None the mesh is built here.

    Returns ``(n_cells, patch)`` — total full-mesh cell count and a patch dict
    of numpy arrays.
    """
    if mesh is None:
        mesh = build_c_mesh_nodes(naca_code, aoa_deg=aoa_deg)
    nodes, n_airfoil, n_wake = mesh
    ni, nj, _ = nodes.shape
    n_cells = (ni - 1) * (nj - 1)

    # Round nodes to the Gmsh %.10e serialisation precision, matching the
    # polyMesh OpenFOAM builds its geometry from.
    _fmt10e = np.frompyfunc(lambda v: float("%.10e" % v), 1, 1)
    nodes_r = _fmt10e(nodes).astype(np.float64)

    dz = EXTRUSION_DZ
    j = 0
    i_lo, i_hi = n_wake - 1, n_wake + n_airfoil - 2

    # Centroids for the wall row only (owner-cell centres for outward
    # orientation + ndc); the other ~99.5% of cells are never read.
    wall_cc = _wall_cell_centroids(nodes_r, i_lo, i_hi)

    cells, Sf_list, magSf_list, Cf_list, ndc_list = [], [], [], [], []
    for idx, i in enumerate(range(i_lo, i_hi)):
        c = i * (nj - 1) + j
        A, B = nodes_r[i, 0], nodes_r[i + 1, 0]
        C_own = np.array([wall_cc[idx, 0], wall_cc[idx, 1], dz / 2])
        Sf_raw, mag_sf, Cf_3d, ndc_b = _boundary_face_geometry(A, B, C_own, dz)

        cells.append(c)
        Sf_list.append(Sf_raw)
        magSf_list.append(mag_sf)
        Cf_list.append(Cf_3d)
        ndc_list.append(ndc_b)

    patch = {
        "name": "airfoilWalls",
        "type": "wall",
        "cells": np.array(cells, dtype=np.int64),
        "Sf": np.array(Sf_list, dtype=np.float64),
        "magSf": np.array(magSf_list, dtype=np.float64),
        "Cf": np.array(Cf_list, dtype=np.float64),
        "ndc": np.array(ndc_list, dtype=np.float64),
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
                    ``magSf``, ``ndc``) as numpy arrays.
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

    ``ld`` is 0.0 whenever drag is non-positive (``cd <= _TINY``): a small
    negative ``cd`` from surrogate error would otherwise yield a large or
    negative "lift-to-drag ratio" that reads as a real aerodynamic quantity.
    """
    mesh_indices = np.asarray(mesh_indices, dtype=np.int64)

    wall_cells = np.asarray(wall_patch["cells"], dtype=np.int64)

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

    Sf = np.asarray(wall_patch["Sf"], dtype=np.float64)[:, :2]     # (Nw,2)
    magSf = np.asarray(wall_patch["magSf"], dtype=np.float64)      # (Nw,)
    ndc = np.asarray(wall_patch["ndc"], dtype=np.float64)          # (Nw,)

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
    ld = cl / cd if cd > _TINY else 0.0

    return {
        "cl": float(cl),
        "cd": float(cd),
        "cd_pressure": float(cd_pressure),
        "cd_friction": float(cd_friction),
        "ld": float(ld),
    }
