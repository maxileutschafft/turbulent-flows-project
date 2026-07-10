"""Build an in-memory inference scenario from (naca_code, reynolds, angle_of_attack).

Generates a structured C-mesh, computes per-cell features (x, y, sdf, u_init,
v_init, is_wall, ...), and returns a dict matching the on-disk `.npz` schema
consumed by `surrogate.gno.infer.infer`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from utils.mesh import build_c_mesh_nodes
from utils.sdf import analytical_naca4_sdf, analytical_naca4_sdf_torch

if TYPE_CHECKING:
    import torch

DeviceLike = Union[str, "torch.device", None]


def build_scenario(
    naca_code: str,
    reynolds: float,
    angle_of_attack: float,
    *,
    bbox: tuple[tuple[float, float], tuple[float, float]] = ((-1.5, 3.5), (-1.5, 1.5)),
    save_mesh_to: Path | None = None,
    device: DeviceLike = None,
    mesh: tuple[np.ndarray, int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Generate a scenario dict for GNO inference.

    Parameters
    ----------
    naca_code : str
        4-digit NACA code (e.g. "2210").
    reynolds : float
        Chord-based Reynolds number.
    angle_of_attack : float
        Angle of attack in degrees. Positive = leading edge up.
    bbox : ((xmin, xmax), (ymin, ymax))
        Spatial crop applied to the full C-mesh's cell centers.
    save_mesh_to : Path | None
        If given, write the (pre-crop) node coordinates, (post-crop) cell
        centers, and (post-crop) quad connectivity to an HDF5 file.
    device : torch.device | str | None
        If a CUDA device, the SDF is computed on-GPU via
        `analytical_naca4_sdf_torch(device=device)`; otherwise the
        vectorised CPU `analytical_naca4_sdf` is used.
    mesh : (nodes, n_airfoil, n_wake) or None
        Pre-built C-mesh from `build_c_mesh_nodes`. When given, it is used as
        is instead of rebuilding — lets a caller that also needs the raw mesh
        (e.g. the Cl/Cd wall patch) build it once and share it. When None the
        mesh is built here.

    Returns
    -------
    dict with keys matching the on-disk `.npz` schema:
        x, y, sdf, u_init, v_init     float32 [N]
        reynolds, angle_of_attack     float32 scalar (0-d)
        naca_code                     0-d str array
        is_wall                       bool    [N]
    """
    # --- 1. Build C-mesh nodes ------------------------------------------------
    if mesh is None:
        mesh = build_c_mesh_nodes(naca_code, aoa_deg=angle_of_attack)
    nodes, n_airfoil, n_wake = mesh
    ni, nj, _ = nodes.shape
    ni_c, nj_c = ni - 1, nj - 1

    # --- 2. Cell centers (shape [ni_c, nj_c, 2]) ------------------------------
    centers = 0.25 * (
        nodes[:-1, :-1] + nodes[1:, :-1] +
        nodes[:-1, 1:]  + nodes[1:, 1:]
    )

    # --- 3. Flatten in C-order; build (ci, cj) for each flat index ------------
    centers_flat = centers.reshape(-1, 2)              # [N_full, 2]
    ci_grid, cj_grid = np.meshgrid(
        np.arange(ni_c), np.arange(nj_c), indexing="ij"
    )
    ci_flat = ci_grid.reshape(-1)
    cj_flat = cj_grid.reshape(-1)

    # --- 4. Wall mask BEFORE cropping ----------------------------------------
    # Wall cells lie on the airfoil's inner face: j=0 row, with i in the
    # airfoil arc (skipping the wake-cut indices on each side).
    is_wall_full = (
        (cj_flat == 0)
        & (ci_flat >= (n_wake - 1))
        & (ci_flat <= (n_wake + n_airfoil - 3))
    )

    # --- 5. Apply bbox crop ---------------------------------------------------
    (xmin, xmax), (ymin, ymax) = bbox
    x_full = centers_flat[:, 0]
    y_full = centers_flat[:, 1]
    keep = (
        (x_full >= xmin) & (x_full <= xmax)
        & (y_full >= ymin) & (y_full <= ymax)
    )
    centers_cropped = centers_flat[keep]               # [N, 2]
    is_wall = is_wall_full[keep]
    # Full-mesh cell index of each kept (cropped) cell, in the same raster
    # order (i*(nj-1)+j) as the C-mesh. Lets downstream code map the cropped
    # prediction rows back onto the full mesh — e.g. the airfoil-wall patch
    # used for the Cl/Cd surface integral (`surrogate.coefficients`).
    mesh_indices = np.flatnonzero(keep).astype(np.int64)

    # --- 6. SDF from airfoil surface (analytical, matches training) ---------
    # The training dataset's `sdf` was computed analytically against the
    # smooth open-TE NACA-4 surface plus a TE-closure segment, NOT the
    # closed-TE polygon used for mesh construction. Reproduce that here so
    # downstream inference sees consistent inputs.
    _use_cuda = False
    if device is not None:
        try:
            import torch
            _dev = device if isinstance(device, torch.device) else torch.device(device)
            _use_cuda = _dev.type == "cuda"
        except Exception:
            _use_cuda = False
    if _use_cuda:
        sdf = analytical_naca4_sdf_torch(centers_cropped, naca_code, device=_dev)
    else:
        sdf = analytical_naca4_sdf(centers_cropped, naca_code)
    sdf = sdf.astype(np.float32)

    # --- 7. Freestream velocity ----------------------------------------------
    chord = 1.0
    nu = 1e-5
    U_mag = reynolds * nu / chord
    aoa_rad = math.radians(angle_of_attack)
    u_init_scalar = U_mag * math.cos(aoa_rad)
    v_init_scalar = U_mag * math.sin(aoa_rad)

    N = centers_cropped.shape[0]
    x = centers_cropped[:, 0].astype(np.float32)
    y = centers_cropped[:, 1].astype(np.float32)
    u_init = np.full(N, u_init_scalar, dtype=np.float32)
    v_init = np.full(N, v_init_scalar, dtype=np.float32)

    scenario = {
        "x":               x,
        "y":               y,
        "sdf":             sdf,
        "u_init":          u_init,
        "v_init":          v_init,
        "reynolds":        np.float32(reynolds),
        "angle_of_attack": np.float32(angle_of_attack),
        "naca_code":       np.array(naca_code),
        "is_wall":         is_wall.astype(bool),
        "mesh_indices":    mesh_indices,
    }

    # --- 8. Optional mesh dump (for downstream visualisation/debugging) -------
    if save_mesh_to is not None:
        import h5py

        save_mesh_to = Path(save_mesh_to)
        save_mesh_to.parent.mkdir(parents=True, exist_ok=True)

        # quad connectivity pre-crop, then index by `keep`
        # CCW order: bottom-left (i,j), bottom-right (i+1,j),
        #            top-right (i+1,j+1), top-left (i,j+1)
        # Flat node index = i * nj + j.
        ci_arr = ci_flat
        cj_arr = cj_flat
        bl = ci_arr       * nj + cj_arr
        br = (ci_arr + 1) * nj + cj_arr
        tr = (ci_arr + 1) * nj + (cj_arr + 1)
        tl = ci_arr       * nj + (cj_arr + 1)
        connectivity_full = np.column_stack([bl, br, tr, tl]).astype(np.int64)
        connectivity = connectivity_full[keep]

        with h5py.File(save_mesh_to, "w") as fh:
            fh.create_dataset("points",       data=nodes.reshape(-1, 2))
            fh.create_dataset("cell_centers", data=centers_cropped)
            fh.create_dataset("connectivity", data=connectivity)
            fh.attrs["naca_code"]       = naca_code
            fh.attrs["reynolds"]        = float(reynolds)
            fh.attrs["angle_of_attack"] = float(angle_of_attack)
            fh.attrs["bbox_x_min"]      = float(xmin)
            fh.attrs["bbox_x_max"]      = float(xmax)
            fh.attrs["bbox_y_min"]      = float(ymin)
            fh.attrs["bbox_y_max"]      = float(ymax)

        print(f"Saved mesh → {save_mesh_to}")

    return scenario
