"""Inference visualization — scalar field contour plots and velocity streamlines."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from utils.naca_geometry import naca4_polygon

# Display labels for each scalar field.
FIELD_LABELS: dict[str, str] = {
    "u": "u [m/s]",
    "v": "v [m/s]",
    "vel_mag": "|v| [m/s]",
    "p": "p [Pa]",
    "k": "k [m²/s²]",
    "omega": "ω [1/s]",
    "nut": "νᵗ [m²/s]",
}

# Column index within the 6-channel GNO output / target tensor (GNO_TARGET_COLS order).
_FIELD_COL: dict[str, int] = {
    col: i for i, col in enumerate(["u", "v", "p", "k", "omega", "nut"])
}


def get_naca_code(result: dict) -> str | None:
    """Return the NACA 4-digit code string from an infer() result dict, or None."""
    raw = result.get("raw", {})
    code = raw.get("naca_code")
    return str(code) if code is not None else None


def field_values(result: dict, field: str, source: str = "prediction") -> np.ndarray:
    """Extract a named scalar field as a flat numpy array from an infer() result.

    Args:
        result: dict returned by ``infer()``.
        field:  one of ``u | v | p | k | omega | nut | vel_mag``.
        source: ``"prediction"`` or ``"truth"``.

    Returns:
        1-D float32 array of length N (number of mesh nodes).
    """
    tensor = result["predictions"] if source == "prediction" else result["target"]
    data = tensor.numpy()
    if field == "vel_mag":
        return np.sqrt(data[:, _FIELD_COL["u"]] ** 2 + data[:, _FIELD_COL["v"]] ** 2)
    return data[:, _FIELD_COL[field]]


def plot_scalar_field(
    result: dict,
    field: str,
    source: str = "prediction",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Tricontourf contour plot of a single scalar field from an infer() result.

    Args:
        result: dict returned by ``infer()``.
        field:  ``u | v | p | k | omega | nut | vel_mag``.
        source: ``"prediction"`` (default) or ``"truth"``.
        title:  custom figure title; auto-generated if None.
        ax:     existing Axes to draw on; new Figure/Axes created if None.

    Returns:
        The matplotlib Figure.
    """
    pos = result["pos"].numpy()
    x, y = pos[:, 0], pos[:, 1]
    values = field_values(result, field, source)
    naca_code = get_naca_code(result)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    triang = tri.Triangulation(x, y)
    tcf = ax.tricontourf(triang, values, levels=50, cmap="viridis")
    fig.colorbar(tcf, ax=ax, label=FIELD_LABELS.get(field, field))

    if naca_code is not None:
        ax.add_patch(naca4_polygon(naca_code))

    if title is None:
        src_lbl = "GNO prediction" if source == "prediction" else "Ground truth"
        title = f"{src_lbl} — {FIELD_LABELS.get(field, field)}"
        if naca_code is not None:
            title = f"NACA {naca_code}  {title}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    if own_fig:
        fig.tight_layout()
    return fig


def plot_velocity_streamlines(
    result: dict,
    source: str = "prediction",
    grid_res: int = 200,
    density: float = 1.5,
) -> plt.Figure:
    """Velocity streamline plot colour-mapped by flow speed.

    u and v are interpolated from the unstructured mesh to a regular grid via
    ``matplotlib.tri.LinearTriInterpolator`` before calling ``streamplot``.

    Args:
        result:   dict returned by ``infer()``.
        source:   ``"prediction"`` (default) or ``"truth"``.
        grid_res: grid points along each axis for the regular interpolation grid.
        density:  matplotlib streamplot density parameter.

    Returns:
        The matplotlib Figure.
    """
    pos = result["pos"].numpy()
    x_raw, y_raw = pos[:, 0], pos[:, 1]
    tensor = result["predictions"] if source == "prediction" else result["target"]
    data = tensor.numpy()
    u_raw = data[:, _FIELD_COL["u"]]
    v_raw = data[:, _FIELD_COL["v"]]
    speed_raw = np.sqrt(u_raw ** 2 + v_raw ** 2)
    naca_code = get_naca_code(result)

    triang = tri.Triangulation(x_raw, y_raw)
    # Explicitly cast to Python float (float64) so np.linspace produces a
    # float64 array — matplotlib's streamplot requires equally-spaced float64.
    xi = np.linspace(float(x_raw.min()), float(x_raw.max()), grid_res)
    yi = np.linspace(float(y_raw.min()), float(y_raw.max()), grid_res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    u_grid = np.ma.filled(tri.LinearTriInterpolator(triang, u_raw)(xi_grid, yi_grid), np.nan)
    v_grid = np.ma.filled(tri.LinearTriInterpolator(triang, v_raw)(xi_grid, yi_grid), np.nan)
    speed_grid = np.ma.filled(
        tri.LinearTriInterpolator(triang, speed_raw)(xi_grid, yi_grid), np.nan
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    strm = ax.streamplot(
        xi, yi, u_grid, v_grid,
        color=speed_grid,
        cmap="viridis",
        density=density,
        linewidth=1.0,
    )
    fig.colorbar(strm.lines, ax=ax, label=FIELD_LABELS["vel_mag"])

    if naca_code is not None:
        ax.add_patch(naca4_polygon(naca_code))

    src_lbl = "GNO prediction" if source == "prediction" else "Ground truth"
    title = f"Velocity streamlines — {src_lbl}"
    if naca_code is not None:
        title = f"NACA {naca_code}  {title}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(x_raw.min(), x_raw.max())
    ax.set_ylim(y_raw.min(), y_raw.max())
    fig.tight_layout()
    return fig
