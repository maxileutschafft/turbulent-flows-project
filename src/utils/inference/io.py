"""Persist GNO inference predictions to disk in the dataset .npz format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from surrogate.gno.schema import GNO_TARGET_COLS


def _is_scalar_like(value: Any) -> bool:
    if np.isscalar(value):
        return True
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return True
    return False


def save_predictions_npz(result: dict, path: str | Path) -> None:
    """Write GNO predictions as a .npz file mirroring the dataset format.

    Saves x/y mesh coordinates, predicted target channels (u, v, p, k, omega,
    nut), scalar metadata from the input file (naca_code, reynolds,
    angle_of_attack) and checkpoint provenance. The file can be reloaded with
    ``numpy.load`` and used by the field-plotting utilities.

    Args:
        result: dict returned by ``infer()``.
        path:   destination .npz path (parent directories are created as needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pos = result["pos"].numpy()            # [N, 2]
    preds = result["predictions"].numpy()  # [N, 6]
    raw: dict[str, Any] = result.get("raw", {})

    payload: dict[str, Any] = {
        "x": pos[:, 0],
        "y": pos[:, 1],
    }
    for i, col in enumerate(GNO_TARGET_COLS):
        payload[col] = preds[:, i]

    for key, value in raw.items():
        if key in payload:
            continue
        if _is_scalar_like(value):
            payload[key] = value

    meta = result.get("meta", {})
    payload["checkpoint"] = str(meta.get("checkpoint", ""))
    payload["ckpt_epoch"] = int(meta.get("ckpt_epoch", -1))
    if "target_cols" in meta:
        payload["target_cols"] = np.array(meta["target_cols"], dtype=object)
    if "input_cols" in meta:
        payload["input_cols"] = np.array(meta["input_cols"], dtype=object)

    np.savez_compressed(path, **payload)
    print(f"Saved predictions -> {path}")
