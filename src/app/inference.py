"""Multi-model surrogate inference and field rendering for the web app.

Public API
----------
available_devices() -> list[str]
    Return subset of ["cuda","mps","cpu"] that are actually usable on this
    machine, ordered cuda > mps > cpu.

default_device() -> str
    Return the auto-selected device string (same logic as select_device()).

load_model(name="gno", device=None)
    Idempotent per name. Call once at server start to warm-load the
    checkpoint into CPU RAM.  Subsequent calls for the same name are no-ops.

predict(naca, reynolds, aoa_deg, model="gno", device=None) -> dict
    Run inference for the given flow conditions and model.  Results are cached
    in memory so repeated calls with identical parameters are instant.
    Cache key is device-independent (output is numerically identical across
    devices); device only governs where a cache-miss computation runs.

streamline_field_svg(pos, u, v, naca, aoa_deg, ...) -> str
    Render streamlines as a transparent SVG in screen-coordinate frame.

scalar_field_svg(pos, values, naca, aoa_deg) -> (svg_str, vmin, vmax)
    Render a scalar field as a masked viridis filled-contour SVG.

colorbar_ticks(vmin, vmax) -> list[str]
    Return 4 tick strings ordered HIGH->LOW for the colorbar.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-model registry — populated lazily by load_model(name)
# Models are always stored in CPU RAM; the device arg is accepted for API
# compatibility but placement is always CPU.
# ---------------------------------------------------------------------------
_MODELS: dict[str, dict] = {}  # name -> {model, y_norm, ...}  (CPU-resident)
_LOCK = threading.Lock()

# Valid model names
VALID_MODELS = {"gno", "transolver", "domino"}

# Valid device strings
_VALID_DEVICES = {"cuda", "mps", "cpu"}

# In-memory prediction cache: (model, naca, round(reynolds), round(aoa, 2)) -> dict
# Bounded LRU — keeps a long-running server from accumulating per-node result
# arrays without bound as users sweep the sliders.
_CACHE_MAXSIZE = 20
_CACHE: OrderedDict[tuple, dict] = OrderedDict()

# Checkpoint locations relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CKPT_PATHS = {
    "gno": (
        _PROJECT_ROOT
        / "checkpoints"
        / "gno_w32_d6_k16_lr0.001_NACA_4_Digit_for_ML"
        / "best.pt"
    ),
    "transolver": (
        _PROJECT_ROOT
        / "checkpoints"
        / "transolver_h208_l4_s32_lr0.001_NACA_4_Digit_for_ML"
        / "best.pt"
    ),
    "domino": (
        _PROJECT_ROOT
        / "checkpoints"
        / "domino_bl496_bf18_m4_k16_g64_surf256_lr0.001_NACA_4_Digit_for_ML"
        / "best.pt"
    ),
}

# Screen-coordinate transform constants (must match app.html exactly)
_PIV, _OX, _CY, _S = 0.25, 130.0, 130.0, 200.0


# ---------------------------------------------------------------------------
# Device availability
# ---------------------------------------------------------------------------

def available_devices() -> list[str]:
    """Return the subset of ["cuda","mps","cpu"] usable on this machine.

    Order: cuda first, then mps, then cpu (cpu always present).
    """
    from surrogate.gno.infer import available_devices as _available_devices
    return _available_devices()


def default_device() -> str:
    """Return the auto-selected device string (cuda > mps > cpu)."""
    from surrogate.gno.infer import select_device
    return select_device().type


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(name: str = "gno", device: str | None = None) -> dict:
    """Load the named checkpoint into CPU RAM (idempotent per name).

    Parameters
    ----------
    name : str
        One of ``"gno"``, ``"transolver"``, ``"domino"``.
    device : str or None
        Accepted for API compatibility but ignored — models always reside in
        CPU RAM and are moved to the GPU only for a cache-miss forward pass.

    Returns
    -------
    dict
        The model bundle stored in ``_MODELS[name]`` (CPU-resident).
    """
    if name not in VALID_MODELS:
        raise ValueError(f"name must be one of {VALID_MODELS}, got {name!r}")

    if device is not None and device not in _VALID_DEVICES:
        raise ValueError(f"device must be one of {_VALID_DEVICES}, got {device!r}")

    if name in _MODELS:
        return _MODELS[name]

    import torch

    from surrogate.gno.utils import UnitGaussianNormalizer

    cpu = torch.device("cpu")
    ckpt_path = _CKPT_PATHS[name]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    logger.info("Loading %s checkpoint into CPU RAM", name)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    ta = ckpt["args"]

    if name == "gno":
        from surrogate.gno.model import KernelNN

        model = KernelNN(
            width_node=ta["width_node"],
            ker_width=ta["ker_width"],
            depth=ta["depth"],
        ).to(cpu)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        y_norm = UnitGaussianNormalizer.from_stats(ckpt["y_norm"], device=cpu)

        bundle: dict = {
            "model": model,
            "y_norm": y_norm,
            "k": ta["k"],
        }
        logger.info(
            "GNO loaded (CPU RAM): width_node=%d, ker_width=%d, depth=%d, k=%d",
            ta["width_node"], ta["ker_width"], ta["depth"], ta["k"],
        )

    elif name == "transolver":
        from surrogate.transolver.model import Transolver

        model = Transolver(
            functional_dim=7,
            out_dim=6,
            n_hidden=ta["n_hidden"],
            n_layers=ta["n_layers"],
            n_head=ta["n_head"],
            mlp_ratio=ta["mlp_ratio"],
            slice_num=ta["slice_num"],
            dropout=ta.get("dropout", 0.0),
            act=ta.get("act", "gelu"),
        ).to(cpu)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        y_norm = UnitGaussianNormalizer.from_stats(ckpt["y_norm"], device=cpu)

        bundle = {
            "model": model,
            "y_norm": y_norm,
        }
        logger.info(
            "Transolver loaded (CPU RAM): n_hidden=%d, n_layers=%d, n_head=%d, slice_num=%d",
            ta["n_hidden"], ta["n_layers"], ta["n_head"], ta["slice_num"],
        )

    elif name == "domino":
        from surrogate.domino.model import DoMINO

        k_surf = int(ta.get("k_surf", 8))
        grid_res = ta["grid_res"]
        k_neighbors = ta["k_neighbors"]

        model = DoMINO(
            input_features=3,
            out_dim=6,
            base_layer=ta["base_layer"],
            base_layer_surf=int(ta.get("base_layer_surf", 416)),
            base_filters=ta["base_filters"],
            num_modes=ta["num_modes"],
            k_neighbors=k_neighbors,
            k_surf=k_surf,
            grid_res=grid_res,
        ).to(cpu)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        y_norm = UnitGaussianNormalizer.from_stats(ckpt["y_norm"], device=cpu)

        bundle = {
            "model": model,
            "y_norm": y_norm,
            "grid_res": grid_res,
            "k_neighbors": k_neighbors,
        }
        logger.info(
            "DoMINO loaded (CPU RAM): base_layer=%d, base_filters=%d, num_modes=%d, "
            "k_neighbors=%d, k_surf=%d, grid_res=%d",
            ta["base_layer"], ta["base_filters"], ta["num_modes"],
            k_neighbors, k_surf, grid_res,
        )
    else:
        raise ValueError(f"Unhandled model name {name!r}")

    _MODELS[name] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Device memory helpers
# ---------------------------------------------------------------------------


def _free_device_memory(device_str: str) -> None:
    """Release cached GPU memory after a forward pass."""
    import torch
    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif (
        device_str == "mps"
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()


def _forward_to_cpu(
    model_name: str,
    m,
    x,
    pos,
    bundle: dict,
    torch_device,
) -> "torch.Tensor":
    """Run model-specific forward on *torch_device* and return pred_enc on CPU.

    All GPU intermediates (edge_index, edge_attr, aux tensors, raw forward
    output) are local to this function and become unreferenced on return, so
    a subsequent ``empty_cache()`` can fully reclaim them.
    """
    import torch

    with torch.no_grad():
        if model_name == "gno":
            from surrogate.gno.graph import build_graph
            k = bundle["k"]
            edge_index, edge_attr = build_graph(pos, k)
            pred_enc = m(
                x.to(torch_device),
                edge_index.to(torch_device),
                edge_attr.to(torch_device),
            )

        elif model_name == "transolver":
            pred_enc = m(x.to(torch_device))

        elif model_name == "domino":
            from surrogate.domino.datapipe import build_aux
            grid_res = bundle["grid_res"]
            k_neighbors = bundle["k_neighbors"]
            aux_cpu = build_aux(x, grid_res=grid_res, k_neighbors=k_neighbors)
            aux = {
                k_: v.to(torch_device) if isinstance(v, torch.Tensor) else v
                for k_, v in aux_cpu.items()
            }
            raw_result = m(
                x.to(torch_device),
                aux=aux,
                predict_volume=True,
                predict_surface=False,
            )
            # DoMINO returns a dict when both heads are active; volume-only
            # may return a tensor or dict — handle both
            if isinstance(raw_result, dict):
                pred_enc = raw_result["volume"]
            else:
                pred_enc = raw_result

        else:
            raise ValueError(f"Unhandled model name {model_name!r}")

    return pred_enc.detach().to("cpu")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    naca: str,
    reynolds: float,
    aoa_deg: float,
    model: str = "gno",
    device: str | None = None,
) -> dict:
    """Run surrogate inference and return per-node velocity and pressure fields.

    Parameters
    ----------
    naca : str
        4-digit NACA code, e.g. ``"2412"``.
    reynolds : float
        Chord-based Reynolds number.
    aoa_deg : float
        Angle of attack in degrees.
    model : str
        One of ``"gno"``, ``"transolver"``, ``"domino"``.
    device : str or None
        Device to run computation on (``"cuda"``, ``"mps"``, ``"cpu"``).
        Defaults to ``default_device()`` when None.
        The prediction cache is device-independent: a cache hit returns the
        stored result regardless of which device is requested.  A cache miss
        runs computation on the requested device.

    Returns
    -------
    dict with keys
        ``pos``   : (N, 2) float32 ndarray — node (x, y) in CFD frame
        ``u``     : (N,) float32 ndarray  — x-velocity (m/s)
        ``v``     : (N,) float32 ndarray  — y-velocity (m/s)
        ``p``     : (N,) float32 ndarray  — pressure
        ``k``     : (N,) float32 ndarray  — turbulent kinetic energy
        ``omega`` : (N,) float32 ndarray  — specific dissipation rate
        ``nut``   : (N,) float32 ndarray  — turbulent viscosity
    """
    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}, got {model!r}")

    # Cache key is device-independent: numerics are identical across devices.
    cache_key = (model, naca, round(reynolds), round(aoa_deg, 2))
    if cache_key in _CACHE:
        _CACHE.move_to_end(cache_key)
        return _CACHE[cache_key]

    # Resolve device and load model (always CPU-resident; device only governs
    # where the forward pass runs on a cache miss).
    device_str = device if device is not None else default_device()
    bundle = load_model(model, device_str)

    import torch
    from surrogate.gno.schema import (
        apply_target_log_inverse,
        npz_gno_input,
        npz_pos,
    )
    from utils.scenario import build_scenario

    torch_device = torch.device(device_str)

    # 1. Build the in-memory scenario (C-mesh + bbox crop + SDF + freestream
    #    features), same pipeline the notebooks use.
    scenario = build_scenario(naca, reynolds, aoa_deg, device=None)

    # 2. Extract tensors for the model (both on CPU)
    pos = npz_pos(scenario)       # (N, 2) — CPU tensor
    x = npz_gno_input(scenario)   # (N, 7) — CPU tensor

    m = bundle["model"]
    y_norm = bundle["y_norm"]

    # 3. Forward pass — serialised with a lock so concurrent requests don't
    #    fight over the GPU.  The model is moved to the GPU only for the
    #    duration of the forward, then immediately returned to CPU RAM so
    #    idle VRAM usage stays at ~0.
    on_device = device_str != "cpu"
    with _LOCK:
        try:
            if on_device:
                m.to(torch_device)
            pred_enc_cpu = _forward_to_cpu(model, m, x, pos, bundle, torch_device)
        finally:
            if on_device:
                m.to("cpu")
            _free_device_memory(device_str)

    # 4. Decode on CPU — y_norm is CPU-resident so no device transfer needed.
    pred = apply_target_log_inverse(y_norm.decode(pred_enc_cpu)).numpy()

    # pred shape: (N, 6) — u, v, p, k, omega, nut  (numpy, CPU)
    result = {
        "pos":   pos.numpy(),   # (N, 2) float32 — already on CPU
        "u":     pred[:, 0],
        "v":     pred[:, 1],
        "p":     pred[:, 2],
        "k":     pred[:, 3],
        "omega": pred[:, 4],
        "nut":   pred[:, 5],
    }

    _CACHE[cache_key] = result
    _CACHE.move_to_end(cache_key)
    while len(_CACHE) > _CACHE_MAXSIZE:
        _CACHE.popitem(last=False)
    return result


# ---------------------------------------------------------------------------
# Shared screen-coordinate helpers
# ---------------------------------------------------------------------------

def _to_screen(pos: np.ndarray, aoa_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Transform CFD node coordinates to screen (pixel) coordinates.

    Uses the frozen transform constants matching app.html:
      PIV=0.25, OX=130, CY=130, S=200

    Parameters
    ----------
    pos     : (N, 2) ndarray — (x, y) in CFD frame
    aoa_deg : angle of attack in degrees

    Returns
    -------
    X, Y : (N,) ndarrays in screen-pixel frame (origin top-left)
    """
    A = math.radians(aoa_deg)
    ca, sa = math.cos(A), math.sin(A)
    x_cfd = pos[:, 0]
    y_cfd = pos[:, 1]
    dx = x_cfd - _PIV
    dy = y_cfd
    xr = _PIV + dx * ca + dy * sa
    yr = -dx * sa + dy * ca
    X = _OX + xr * _S
    Y = _CY - yr * _S
    return X, Y


def _airfoil_screen_path(naca: str, aoa_deg: float):
    """Return a matplotlib.path.Path for the airfoil polygon in screen coords.

    The polygon is closed: upper surface LE->TE then lower surface TE->LE.
    Used for masking (contains_points).

    Parameters
    ----------
    naca    : 4-digit NACA code
    aoa_deg : angle of attack in degrees

    Returns
    -------
    matplotlib.path.Path
    """
    from matplotlib.path import Path as MplPath
    from utils.naca_geometry import naca4_coords

    xu, yu, xl, yl = naca4_coords(naca, n=300)
    # Upper LE->TE then lower TE->LE (closed polygon)
    verts_cfd = np.column_stack([
        np.concatenate([xu, xl[::-1]]),
        np.concatenate([yu, yl[::-1]]),
    ])  # (2n, 2)

    A = math.radians(aoa_deg)
    ca, sa = math.cos(A), math.sin(A)
    af_dx = verts_cfd[:, 0] - _PIV
    af_dy = verts_cfd[:, 1]
    af_xr = _PIV + af_dx * ca + af_dy * sa
    af_yr = -af_dx * sa + af_dy * ca
    af_X = _OX + af_xr * _S
    af_Y = _CY - af_yr * _S
    af_screen = np.column_stack([af_X, af_Y])

    return MplPath(af_screen)


# ---------------------------------------------------------------------------
# Streamline computation
# ---------------------------------------------------------------------------

def streamline_field_svg(
    pos: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    naca: str,
    aoa_deg: float,
    *,
    density: float = 0.6,
) -> str:
    """Render streamlines as a transparent SVG in screen-coordinate frame.

    Parameters
    ----------
    pos     : (N, 2) ndarray — node (x, y) in CFD frame
    u, v    : (N,) ndarrays — velocity components in CFD frame
    naca    : 4-digit NACA code
    aoa_deg : angle of attack in degrees
    density : matplotlib streamplot density (higher = more lines)

    Returns
    -------
    SVG string with viewBox="0 0 480 260", width/height="100%".
    Transparent background, no axes, no arrows, colour #43707d.
    """
    import matplotlib
    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    # ------------------------------------------------------------------
    # 1. Screen-coordinate transform (via shared helper)
    # ------------------------------------------------------------------
    X, Y = _to_screen(pos, aoa_deg)

    A = math.radians(aoa_deg)
    ca, sa = math.cos(A), math.sin(A)

    # Screen velocities: rotate same way, then flip v for y-down convention
    u_r = u * ca + v * sa
    v_r = -u * sa + v * ca
    U_screen = u_r
    V_screen = -v_r  # y-down: positive V means moving DOWN in screen space

    # ------------------------------------------------------------------
    # 2. Triangulate and interpolate onto regular grid
    # ------------------------------------------------------------------
    triang = mtri.Triangulation(X, Y)
    u_interp = mtri.LinearTriInterpolator(triang, U_screen)
    v_interp = mtri.LinearTriInterpolator(triang, V_screen)

    xi = np.linspace(0, 480, 241)
    yi = np.linspace(0, 260, 131)
    Xi, Yi = np.meshgrid(xi, yi)

    Ug = np.ma.filled(u_interp(Xi, Yi), np.nan)
    Vg = np.ma.filled(v_interp(Xi, Yi), np.nan)

    # ------------------------------------------------------------------
    # 3. Mask airfoil interior (critical: prevents spurious triangles)
    # ------------------------------------------------------------------
    af_path = _airfoil_screen_path(naca, aoa_deg)
    grid_pts = np.column_stack([Xi.ravel(), Yi.ravel()])
    inside = af_path.contains_points(grid_pts).reshape(Xi.shape)

    Ug[inside] = np.nan
    Vg[inside] = np.nan

    # ------------------------------------------------------------------
    # 4. Render with matplotlib (72 dpi → 480×260 pt figure)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(480 / 72, 260 / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 480)
    ax.set_ylim(260, 0)  # INVERTED: y=0 at top, matching SVG/screen convention

    strm = ax.streamplot(
        xi, yi, Ug, Vg,
        color="#43707d",
        linewidth=1.0,
        density=density,
        arrowstyle="-",            # no arrowheads
        broken_streamlines=False,  # continuous long lines (matplotlib >= 3.6)
    )
    # Belt-and-suspenders: hide arrows if any were drawn
    try:
        strm.arrows.set_visible(False)
    except Exception:
        pass

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # ------------------------------------------------------------------
    # 5. Export to SVG string and post-process
    # ------------------------------------------------------------------
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)

    return _postprocess_svg(buf.getvalue())


# ---------------------------------------------------------------------------
# Scalar field rendering
# ---------------------------------------------------------------------------

def scalar_field_svg(
    pos: np.ndarray,
    values: np.ndarray,
    naca: str,
    aoa_deg: float,
    *,
    levels: int = 50,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> tuple[str, float, float]:
    """Render a scalar field as a masked viridis filled-contour SVG.

    Parameters
    ----------
    pos     : (N, 2) ndarray — node (x, y) in CFD frame
    values  : (N,) ndarray — scalar values at each node
    naca    : 4-digit NACA code
    aoa_deg : angle of attack in degrees
    levels  : number of filled-contour bands (default 50)
    vmin    : lower bound of the colour/contour range. Uses ``values.min()``
              when None.
    vmax    : upper bound of the colour/contour range. Uses ``values.max()``
              when None.

    Returns
    -------
    (svg_str, vmin, vmax)
        svg_str : SVG string with viewBox="0 0 480 260", width/height="100%"
        vmin    : minimum value (float)
        vmax    : maximum value (float)
    """
    import matplotlib
    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    # ------------------------------------------------------------------
    # 1. Transform to screen coordinates
    # ------------------------------------------------------------------
    X, Y = _to_screen(pos, aoa_deg)

    # ------------------------------------------------------------------
    # 2. Triangulate and mask airfoil interior
    # ------------------------------------------------------------------
    triang = mtri.Triangulation(X, Y)

    # Compute per-triangle centroids
    tri_indices = triang.triangles  # (M, 3)
    cx = X[tri_indices].mean(axis=1)
    cy = Y[tri_indices].mean(axis=1)
    centroids = np.column_stack([cx, cy])

    # Mask triangles whose centroid falls inside the airfoil polygon
    af_path = _airfoil_screen_path(naca, aoa_deg)
    tri_mask = af_path.contains_points(centroids)
    triang.set_mask(tri_mask)

    # ------------------------------------------------------------------
    # 3. Data range
    # ------------------------------------------------------------------
    vmin = float(values.min()) if vmin is None else float(vmin)
    vmax = float(values.max()) if vmax is None else float(vmax)

    # ------------------------------------------------------------------
    # 4. Render (same figure setup as streamline_field_svg)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(480 / 72, 260 / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 480)
    ax.set_ylim(260, 0)  # INVERTED: y=0 at top

    ax.tricontourf(triang, values, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # ------------------------------------------------------------------
    # 5. Export to SVG string and post-process
    # ------------------------------------------------------------------
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)

    return _postprocess_svg(buf.getvalue()), vmin, vmax


# ---------------------------------------------------------------------------
# SVG post-processing (shared)
# ---------------------------------------------------------------------------

def _postprocess_svg(svg_str: str, *, width: int = 480, height: int = 260) -> str:
    """Strip XML declaration, DOCTYPE, and fix viewBox/width/height."""
    import re

    svg_str = re.sub(r'<\?xml[^?]*\?>', '', svg_str)
    svg_str = re.sub(r'<!DOCTYPE[^>]*>', '', svg_str)
    svg_str = re.sub(
        r'<svg\s[^>]*>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="100%" height="100%" '
        f'preserveAspectRatio="xMidYMid meet">',
        svg_str,
        count=1,
    )
    return svg_str.strip()


# ---------------------------------------------------------------------------
# Field registry
# ---------------------------------------------------------------------------

def _get_field_values(result: dict, view: str) -> np.ndarray:
    """Extract scalar field values from a predict() result dict."""
    if view == "vel_mag":
        return np.sqrt(result["u"] ** 2 + result["v"] ** 2)
    return result[view]


# Registry: view_name -> (label, value_extractor_fn)
# Labels match FIELD_LABELS in src/utils/inference/fields.py exactly.
FIELD_REGISTRY: dict[str, tuple[str, Any]] = {
    "vel_mag": ("|v| [m/s]",   lambda r: np.sqrt(r["u"] ** 2 + r["v"] ** 2)),
    "u":       ("u [m/s]",     lambda r: r["u"]),
    "v":       ("v [m/s]",     lambda r: r["v"]),
    "p":       ("p [Pa]",      lambda r: r["p"]),
    "k":       ("k [m²/s²]",   lambda r: r["k"]),
    "omega":   ("ω [1/s]",     lambda r: r["omega"]),
    "nut":     ("νᵗ [m²/s]",   lambda r: r["nut"]),
}

VALID_VIEWS = {"streamlines"} | set(FIELD_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Colorbar ticks
# ---------------------------------------------------------------------------

def colorbar_ticks(vmin: float, vmax: float) -> list[str]:
    """Return 4 tick strings ordered HIGH->LOW (top->bottom of colorbar).

    Scientific notation (e.g. "3.2e7") when abs >= 1e4 or 0 < abs < 1e-2,
    otherwise ~2-3 significant figures (e.g. "1.8", "-2.7").
    """
    ticks = np.linspace(vmax, vmin, 4)
    result = []
    for t in ticks:
        at = abs(t)
        if at == 0.0:
            result.append("0")
        elif at >= 1e4 or (at < 1e-2 and at > 0):
            # Scientific notation, compact
            s = f"{t:.2e}"
            import re
            m = re.match(r'(-?)(\d+\.\d+)e([+-]\d+)', s)
            if m:
                sign, mantissa, exp = m.groups()
                mantissa = mantissa.rstrip('0')
                if mantissa.endswith('.'):
                    mantissa += '0'
                exp_int = int(exp)
                result.append(f"{sign}{mantissa}e{exp_int}")
            else:
                result.append(s)
        else:
            # 2-3 significant figures
            if at >= 100:
                result.append(f"{t:.0f}")
            elif at >= 10:
                result.append(f"{t:.1f}")
            else:
                result.append(f"{t:.2g}")
    return result
