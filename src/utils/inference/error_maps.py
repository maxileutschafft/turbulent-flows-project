"""Inference error-map utilities: prediction | ground truth | |error| plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from utils.naca_geometry import naca4_polygon
from utils.inference.fields import FIELD_LABELS, field_values, get_naca_code


def plot_error_map(
    result: dict,
    field: str,
    title: str | None = None,
) -> plt.Figure:
    """Three-panel comparison figure: prediction | ground truth | |error|.

    The prediction and ground-truth panels share a common colour scale so they
    are directly comparable. The error panel uses its own scale with the
    ``'hot_r'`` colourmap. RMSE across all mesh nodes is shown in the figure
    super-title.

    Args:
        result: dict returned by ``infer()``.
        field:  ``u | v | p | k | omega | nut | vel_mag``.
        title:  prefix for the super-title; auto-generated if None.

    Returns:
        The matplotlib Figure.
    """
    pos = result["pos"].numpy()
    x, y = pos[:, 0], pos[:, 1]
    pred = field_values(result, field, source="prediction")
    gt = field_values(result, field, source="truth")
    error = np.abs(pred - gt)
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    naca_code = get_naca_code(result)
    label = FIELD_LABELS.get(field, field)

    triang = tri.Triangulation(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin = float(min(pred.min(), gt.min()))
    vmax = float(max(pred.max(), gt.max()))

    panels = [
        (axes[0], pred, "Prediction", "viridis", vmin, vmax),
        (axes[1], gt, "Ground truth", "viridis", vmin, vmax),
        (axes[2], error, "|Error|", "hot_r", None, None),
    ]
    for ax, values, sub_title, cmap, lo, hi in panels:
        kwargs: dict = {"levels": 50, "cmap": cmap}
        if lo is not None:
            kwargs["vmin"] = lo
            kwargs["vmax"] = hi
        tcf = ax.tricontourf(triang, values, **kwargs)
        fig.colorbar(tcf, ax=ax, label=label, fraction=0.046, pad=0.04)
        if naca_code is not None:
            ax.add_patch(naca4_polygon(naca_code))
        ax.set_title(sub_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    prefix = title or (f"NACA {naca_code}  {label}" if naca_code else label)
    fig.suptitle(f"{prefix}   RMSE = {rmse:.4e}", fontsize=12)
    fig.tight_layout()
    return fig
