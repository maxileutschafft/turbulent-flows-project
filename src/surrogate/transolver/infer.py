"""Inference implementation for trained Transolver checkpoints on `.npz` scenarios."""

import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from surrogate.gno.infer import select_device
from surrogate.gno.schema import (
    GNO_TARGET_COLS,
    apply_target_log_inverse,
    npz_gno_input,
    npz_gno_target,
    npz_pos,
)
from surrogate.gno.utils import UnitGaussianNormalizer
from surrogate.transolver.model import Transolver


def _load_normalizer(stats: dict) -> UnitGaussianNormalizer:
    """Reconstruct a UnitGaussianNormalizer from saved mean/std tensors."""
    norm = UnitGaussianNormalizer.__new__(UnitGaussianNormalizer)
    norm.mean = stats["mean"]
    norm.std = stats["std"]
    norm.eps = 1e-5
    return norm


def infer(
    input_path: Path | str | Mapping[str, np.ndarray],
    ckpt_path: Path,
    device: torch.device | None = None,
) -> dict:
    """Run Transolver inference and return a result dict.

    Args:
        input_path: Either a path (str / `Path`) to a `.npz` scenario file, OR
            a mapping (e.g. dict) from key → numpy array already loaded in
            memory (matching the `.npz` schema). Mappings are used directly
            without disk I/O — useful for in-memory generated scenarios.
        ckpt_path:  Path to a model checkpoint produced by train.py.
        device:     Torch device. Auto-selected if None.

    Returns:
        {
            "pos":         [N, 2]  float32  spatial coordinates
            "predictions": [N, 6]  float32  model output (denormalised, physical units)
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

    model = Transolver(
        functional_dim=7,
        out_dim=6,
        n_hidden=train_args["n_hidden"],
        n_layers=train_args["n_layers"],
        n_head=train_args["n_head"],
        mlp_ratio=train_args["mlp_ratio"],
        slice_num=train_args["slice_num"],
        dropout=train_args.get("dropout", 0.0),
        act=train_args.get("act", "gelu"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- load input data ---
    pos = npz_pos(raw)
    x = npz_gno_input(raw)
    has_target = all(col in raw for col in ("u", "v", "p", "k", "omega", "nut"))
    target = npz_gno_target(raw) if has_target else None

    # --- target normalizer ---
    if "y_norm" in ckpt:
        y_norm = _load_normalizer(ckpt["y_norm"])
    else:
        raise RuntimeError(
            "Checkpoint has no y_norm. Re-run training with the current code to save it."
        )

    x_dev = x.to(device)

    # --- forward pass (no graph needed) ---
    print("Running Transolver inference...")
    t0 = time.time()
    with torch.no_grad():
        pred_enc = model(x_dev)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    # denormalise: un-z-score, then invert log for {k, omega, nut}
    y_norm.to(device)
    pred_log = y_norm.decode(pred_enc)
    predictions = apply_target_log_inverse(pred_log).cpu()

    # Free GPU memory
    del model, x_dev, pred_enc
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
