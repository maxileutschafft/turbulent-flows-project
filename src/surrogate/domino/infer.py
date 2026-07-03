"""Inference implementation for trained DoMINO checkpoints on `.npz` scenarios.

Volume prediction only (wall-pressure surface head is not exposed here — the
web app and notebooks only consume the volume fields: u, v, p, k, omega, nut).
"""

import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from surrogate.domino.datapipe import build_aux
from surrogate.domino.model import DoMINO
from surrogate.gno.infer import select_device
from surrogate.gno.schema import (
    GNO_TARGET_COLS,
    apply_target_log_inverse,
    npz_gno_input,
    npz_gno_target,
    npz_pos,
)
from surrogate.gno.utils import UnitGaussianNormalizer


def infer(
    input_path: Path | str | Mapping[str, np.ndarray],
    ckpt_path: Path,
    device: torch.device | None = None,
) -> dict:
    """Run DoMINO volume inference and return a result dict.

    Args:
        input_path: Either a path (str / `Path`) to a `.npz` scenario file, OR
            a mapping (e.g. dict) from key → numpy array already loaded in
            memory (matching the `.npz` schema). Mappings are used directly
            without disk I/O — useful for in-memory generated scenarios.
        ckpt_path:  Path to a DoMINO checkpoint produced by train.py.
        device:     Torch device. Auto-selected if None.

    Returns:
        {
            "pos":         [N, 2]  float32  spatial coordinates
            "predictions": [N, 6]  float32  volume model output (physical units)
            "target":      [N, 6]  float32  ground truth [u, v, p, k, omega, nut]
            "raw":         dict    raw arrays from the scenario
            "meta":        dict    checkpoint epoch/loss, input path, timing
        }
    """
    if isinstance(input_path, Mapping):
        raw = dict(input_path)
        input_label = "<in-memory>"
    else:
        input_path = Path(input_path)
        input_label = str(input_path)
        with np.load(input_path) as raw_npz:
            raw = {key: raw_npz[key] for key in raw_npz.files}
    ckpt_path = Path(ckpt_path)

    if device is None:
        device = select_device()

    # --- load checkpoint ---
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    train_args = ckpt["args"]

    k_surf = int(train_args.get("k_surf", 8))

    model = DoMINO(
        input_features=3,
        out_dim=6,
        base_layer=train_args["base_layer"],
        base_layer_surf=int(train_args.get("base_layer_surf", 416)),
        base_filters=train_args["base_filters"],
        num_modes=train_args["num_modes"],
        k_neighbors=train_args["k_neighbors"],
        k_surf=k_surf,
        grid_res=train_args["grid_res"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- load input data ---
    pos = npz_pos(raw)
    x = npz_gno_input(raw)
    has_target = all(col in raw for col in ("u", "v", "p", "k", "omega", "nut"))
    target = npz_gno_target(raw) if has_target else None

    # --- volume normalizer ---
    if "y_norm" in ckpt:
        y_norm = UnitGaussianNormalizer.from_stats(ckpt["y_norm"])
    else:
        raise RuntimeError(
            "Checkpoint has no y_norm. Re-run training with the current code to save it."
        )

    # --- build aux ---
    grid_res = train_args["grid_res"]
    k_neighbors = train_args["k_neighbors"]
    print(f"Building aux (grid_res={grid_res}, k_neighbors={k_neighbors})...")
    t0 = time.time()
    aux_cpu = build_aux(x, grid_res=grid_res, k_neighbors=k_neighbors)
    aux = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in aux_cpu.items()}
    print(f"  done ({time.time()-t0:.1f}s)")

    x_dev = x.to(device)
    y_norm.to(device)

    # --- forward pass (volume head only) ---
    print("Running inference...")
    t0 = time.time()
    with torch.no_grad():
        raw_result = model(x_dev, aux=aux, predict_volume=True, predict_surface=False)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    pred_enc = raw_result["volume"] if isinstance(raw_result, dict) else raw_result

    # denormalise: un-z-score, then invert log for {k, omega, nut}
    pred_log = y_norm.decode(pred_enc)
    predictions = apply_target_log_inverse(pred_log).cpu()

    # Free GPU memory
    del model, x_dev, aux, pred_enc
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "pos": pos,
        "predictions": predictions,
        "target": target,
        "raw": raw,
        "meta": {
            "input": input_label,
            "checkpoint": str(ckpt_path),
            "ckpt_epoch": ckpt["epoch"],
            "ckpt_loss": ckpt["loss"],
            "target_cols": GNO_TARGET_COLS,
            "elapsed_s": elapsed,
        },
    }

    return result
