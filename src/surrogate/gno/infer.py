"""Inference implementation for trained KernelNN checkpoints on `.npz` scenarios."""

import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from surrogate.gno.graph import build_graph
from surrogate.gno.model import KernelNN
from surrogate.gno.schema import (
    GNO_INPUT_COLS,
    GNO_TARGET_COLS,
    INPUT_ALIASES,
    TARGET_ALIASES,
    apply_target_log_inverse,
    require_keys,
    npz_gno_input,
    npz_gno_target,
    npz_pos,
)
from surrogate.gno.utils import UnitGaussianNormalizer


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    """Run model inference and return a result dict.

    Args:
        input_path: Either a path (str / `Path`) to a `.npz` scenario file, OR
            a mapping (e.g. dict) from key -> numpy array already loaded in
            memory (matching the `.npz` schema). Mappings are used directly
            without disk I/O — useful for in-memory generated scenarios.
        ckpt_path:  Path to a model checkpoint produced by train.py.
        device:     Torch device. Auto-selected if None.

    Returns:
        {
            "pos":         [N, 2]  float32  spatial coordinates
            "predictions": [N, 6]  float32  model output (denormalised)
            "target":      [N, 6]  float32  ground truth [u, v, p, k, omega, nut]
            "meta":        dict    checkpoint epoch/loss, input path, timing
        }
    """
    # Resolve `input_path` to (a) an `input_label` for logging/meta and
    # (b) `raw`, the dict-of-arrays scenario payload.
    if isinstance(input_path, Mapping):
        raw = dict(input_path)
        input_label = "<in-memory>"
        input_path_for_load: Path | None = None
    else:
        input_path_for_load = Path(input_path)
        input_label = str(input_path_for_load)
        raw = None  # loaded below
    ckpt_path = Path(ckpt_path)

    if device is None:
        device = select_device()

    # --- load checkpoint ---
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    train_args = ckpt["args"]

    input_mean = train_args.get("input_mean")
    input_std = train_args.get("input_std")
    sdf_col = train_args.get("sdf_col", 2)

    model = KernelNN(
        width_node=train_args["width_node"],
        ker_width=train_args["ker_width"],
        depth=train_args["depth"],
        input_mean=input_mean,
        input_std=input_std,
        sdf_col=sdf_col,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- load input data ---
    if raw is None:
        assert input_path_for_load is not None
        with np.load(input_path_for_load) as raw_npz:
            raw = {key: raw_npz[key] for key in raw_npz.files}

    missing_inputs = require_keys(raw, GNO_INPUT_COLS, aliases=INPUT_ALIASES)
    if missing_inputs:
        raise KeyError(f"Input scenario is missing required columns: {missing_inputs}")
    pos = npz_pos(raw)
    x = npz_gno_input(raw)

    # `npz_gno_target` requires all 6 target channels; in-memory scenarios
    # generated for prediction won't have ground truth, so skip in that case.
    has_target = not require_keys(raw, GNO_TARGET_COLS, aliases=TARGET_ALIASES)
    target = npz_gno_target(raw) if has_target else None

    # --- target normalizer (inputs are normalized inside the model) ---
    if "y_norm" in ckpt:
        y_norm = _load_normalizer(ckpt["y_norm"])
    else:
        print("Warning: checkpoint has no y_norm — recomputing from input data.")
        y_norm = UnitGaussianNormalizer(target) if target is not None else None

    # --- build graph ---
    k = train_args["k"]
    print(f"Building graph (k={k})...")
    t0 = time.time()
    edge_index, edge_attr = build_graph(pos, k)
    print(f"  {edge_index.shape[1]:,} edges  ({time.time()-t0:.1f}s)")

    x_dev = x.to(device)  # raw inputs — KernelNN normalizes internally
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    # --- forward pass ---
    print("Running inference...")
    t0 = time.time()
    with torch.no_grad():
        pred_enc = model(x_dev, edge_index, edge_attr)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    # denormalise predictions: un-z-score, then invert log for {k, omega, nut}
    if y_norm is not None:
        pred_log = y_norm.decode(pred_enc)
        predictions = apply_target_log_inverse(pred_log).cpu()
    else:
        predictions = pred_enc.cpu()

    # Free GPU memory — all results have been moved to CPU above.
    del model, x_dev, edge_index, edge_attr, pred_enc
    if y_norm is not None:
        del pred_log
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
            "target_cols": tuple(train_args.get("target_cols", GNO_TARGET_COLS)),
            "input_cols": tuple(train_args.get("input_cols", GNO_INPUT_COLS)),
            "elapsed_s": elapsed,
        },
    }

    return result
