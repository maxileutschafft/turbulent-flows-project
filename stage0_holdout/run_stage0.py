"""Stage 0 - evaluate the GNO on held-out dataset splits (no new simulation).

The `Kokoslocke/NACA_4_Digit_for_ML` dataset is already split into folders
`train/ val/ test/ ood/` (profile-level, no geometry leakage). Each `.npz`
stores the truth fields (u, v, p, k, omega, nut), the scalars `angle_of_attack`,
`naca_code`, `reynolds`, the converged `cl`/`cd`, and wall-face arrays
(`wall_xy`, `wall_normal`, `wall_length`, `wall_p`, `wall_shear`, `wall_cell`).

Stage 0 runs the surrogate on a chosen split (default `test`) and reports:
  * per-channel field NRMSE (u, v, p, k, omega, nut)  -- the primary metric,
  * integrated Cl / Cd, GNO vs truth, computed with the SAME wall integrator on
    both (convention-independent -- see below),
  * correlation of the error with the inputs (AoA, Re, thickness, camber),
  * the worst cases.

Cl/Cd note: the dataset's stored `cl`/`cd` come from OpenFOAM `forceCoeffs` with
a convention/reference this repo can't fully reproduce (and the airfoil wall
loop is open at the trailing edge). So we do NOT compare the GNO against the
stored `cl`. Instead we integrate the GNO's *and* the truth's pressure field
with one identical wall integrator; the convention cancels and the difference is
purely the surrogate's field error. The stored `cl` is kept as a context column.

Usage
-----
    uv run python stage0_holdout/run_stage0.py --split test     # clean held-out
    uv run python stage0_holdout/run_stage0.py --split ood      # out-of-distribution
    uv run python stage0_holdout/run_stage0.py --self-test      # no torch/dataset
"""

from __future__ import annotations

import argparse
import csv
import re as _re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage_common.aero_post import (  # noqa: E402
    cl_cd_from_fields, cl_cd_from_wall_faces, nrmse, parse_naca, surface_indices,
)

CHANNELS = ["u", "v", "p", "k", "omega", "nut"]
P_INDEX = CHANNELS.index("p")
LOG_CHANNELS = {"k", "omega", "nut"}   # log10-scaled targets -> score in log space (avoids 10**x overflow)
_NAME_RE = _re.compile(r"NACA(\d{4})_([pn])([\d.]+)_([\d.eE+]+)")


def channel_nrmse(pred_col: np.ndarray, truth_col: np.ndarray, log_space: bool) -> float:
    """NRMSE for one channel; log-channels are scored in log10 space so that
    out-of-distribution 10**x overflow does not blow the metric up to inf."""
    if log_space:
        pred_col = np.log10(np.clip(pred_col, 1e-30, 1e30))
        truth_col = np.log10(np.clip(truth_col, 1e-30, 1e30))
    return nrmse(pred_col, truth_col)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def parse_case_name(stem: str) -> tuple[str, float, float]:
    """`NACA2210_n1.6_1.7e5` -> (code='2210', aoa=-1.6, reynolds=170000). Fallback only."""
    m = _NAME_RE.search(stem)
    if not m:
        raise ValueError(f"cannot parse case name {stem!r}")
    code, sign, aoa, re_str = m.groups()
    return code, float(aoa) * (-1.0 if sign == "n" else 1.0), float(re_str)


def read_case_meta(d: dict, stem: str) -> tuple[str, float, float, float | None]:
    """Prefer values stored in the .npz; fall back to the filename."""
    try:
        c_name, a_name, r_name = parse_case_name(stem)
    except ValueError:
        c_name, a_name, r_name = "0000", float("nan"), float("nan")
    code = str(np.ravel(d["naca_code"])[0]) if "naca_code" in d else c_name
    aoa = float(np.ravel(d["angle_of_attack"])[0]) if "angle_of_attack" in d else a_name
    reynolds = float(np.ravel(d["reynolds"])[0]) if "reynolds" in d else r_name
    cl_stored = float(np.ravel(d["cl"])[0]) if "cl" in d else None
    return code, aoa, reynolds, cl_stored


def u_mag_of(d: dict, reynolds: float) -> float:
    if "u_init" in d and "v_init" in d:
        return float(np.hypot(np.ravel(d["u_init"])[0], np.ravel(d["v_init"])[0]))
    return reynolds * 1.0e-5


def field_cl_cd(d: dict, field_p: np.ndarray, aoa: float, u_mag: float) -> tuple[float, float]:
    """Integrate a pressure field to Cl/Cd. Uses the dataset wall faces if
    present (mapped by geometry), else the is_wall/sdf fallback."""
    if all(k in d for k in ("wall_xy", "wall_normal", "wall_length")):
        return cl_cd_from_wall_faces(d["x"], d["y"], field_p, d["wall_xy"],
                                     d["wall_normal"], d["wall_length"], u_mag, aoa)
    idx = surface_indices(d)
    return cl_cd_from_fields(d["x"], d["y"], field_p, idx, u_mag / 1.0e-5, aoa)


# --------------------------------------------------------------------------- #
# per-sample evaluation
# --------------------------------------------------------------------------- #


def evaluate_sample(d: dict, pred: np.ndarray, truth: np.ndarray,
                    code: str, aoa: float, reynolds: float, cl_stored: float | None) -> dict:
    row: dict = {"naca_code": code, "reynolds": reynolds, "aoa": aoa}
    m, p_pos, t = parse_naca(code)
    row["camber"], row["camber_pos"], row["thickness"] = m, p_pos, t

    for i, ch in enumerate(CHANNELS):
        row[f"nrmse_{ch}"] = channel_nrmse(pred[:, i], truth[:, i], ch in LOG_CHANNELS)
    row["nrmse_mean"] = float(np.mean([row[f"nrmse_{c}"] for c in CHANNELS]))

    u_mag = u_mag_of(d, reynolds)
    cl_g, cd_g = field_cl_cd(d, pred[:, P_INDEX], aoa, u_mag)
    cl_t, cd_t = field_cl_cd(d, truth[:, P_INDEX], aoa, u_mag)   # SAME integrator -> convention cancels
    row["cl_gno"], row["cd_gno"] = cl_g, cd_g
    row["cl_truth"], row["cd_truth"] = cl_t, cd_t
    row["cl_stored"] = cl_stored if cl_stored is not None else np.nan
    row["cl_abs_err"] = abs(cl_g - cl_t)
    row["cl_rel_err"] = abs(cl_g - cl_t) / max(abs(cl_t), 0.05)   # floor avoids near-zero-lift blow-up
    return row


# --------------------------------------------------------------------------- #
# real dataset run
# --------------------------------------------------------------------------- #


def _posix_shim():
    import pathlib
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath  # the checkpoint was pickled on Linux


def process_file(fpath: str | Path, ckpt: str | Path, device: str | None) -> dict:
    """Run inference on one .npz and return its evaluation row."""
    from surrogate.gno.infer import infer
    from surrogate.gno.schema import npz_gno_target

    f = Path(fpath)
    d = dict(np.load(f, allow_pickle=True))
    code, aoa, reynolds, cl_stored = read_case_meta(d, f.stem)
    d.setdefault("angle_of_attack", np.float32(aoa))
    d.setdefault("naca_code", np.array(code))
    dev = None if device in (None, "auto") else device
    result = infer(d, Path(ckpt), device=dev)
    pred = result["predictions"].cpu().numpy()
    truth = (result["target"].cpu().numpy() if result.get("target") is not None
             else npz_gno_target(d).numpy())
    return evaluate_sample(d, pred, truth, code, aoa, reynolds, cl_stored)


def _worker(payload):
    """Top-level worker for the process pool (spawn-safe on Windows)."""
    fpath, ckpt, device, torch_threads = payload
    _posix_shim()
    import torch
    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        return ("ok", process_file(fpath, ckpt, device))
    except Exception as exc:  # noqa: BLE001
        return ("err", f"{Path(fpath).name}: {exc}")


def run_real(args) -> list[dict]:
    import os
    _posix_shim()
    from surrogate.gno.infer import select_device

    split_dir = Path(args.data_dir) / args.split
    files = sorted(split_dir.glob("*.npz"))
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(files)   # representative spread across profiles
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .npz files under {split_dir} (is the download finished?)")

    device = None if args.device == "auto" else args.device
    dev = select_device() if device is None else device
    print(f"Split '{args.split}': {len(files)} samples  jobs={args.jobs}  device={dev}")

    rows: list[dict] = []
    if args.jobs and args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        tt = max(1, (os.cpu_count() or args.jobs) // args.jobs)   # torch threads per worker
        payload = [(str(f), str(args.ckpt), args.device, tt) for f in files]
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for status, res in ex.map(_worker, payload):
                done += 1
                if status == "ok":
                    rows.append(res)
                else:
                    print(f"  [skip] {res}")
                if done % 10 == 0 or done == len(files):
                    print(f"  {done}/{len(files)}")
    else:
        for n, f in enumerate(files, 1):
            try:
                rows.append(process_file(f, args.ckpt, args.device))
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {f.name}: {exc}")
            if n % 10 == 0 or n == len(files):
                print(f"  {n}/{len(files)}")
    return rows


# --------------------------------------------------------------------------- #
# self-test (real mesh, synthetic fields, no torch/dataset)
# --------------------------------------------------------------------------- #


def run_self_test(args) -> list[dict]:
    from utils.scenario import build_scenario
    print("SELF-TEST: synthetic fields on real meshes (no model, no dataset).")
    rng = np.random.default_rng(0)
    cases = [("2412", 2.0e5, 2.0), ("4415", 3.0e5, 5.0), ("6409", 4.0e5, -3.0),
             ("0012", 1.5e5, 0.0), ("2415", 5.0e5, 8.0)]
    rows = []
    for code, re, aoa in cases:
        s = build_scenario(code, re, aoa)
        x = s["x"].astype(float)
        y = s["y"].astype(float)
        N = len(x)
        U = re * 1e-5
        truth = np.zeros((N, 6))
        truth[:, 0] = U * (1.0 + 0.4 * np.tanh(2.0 * y))
        truth[:, 1] = U * (0.3 * np.sin(3.0 * x))
        truth[:, 2] = -y * (1.0 + 0.1 * aoa) + 0.1 * np.cos(2.0 * x)
        truth[:, 3] = 0.5 + 0.5 * np.sin(4.0 * x) ** 2
        truth[:, 4] = 10.0 * (1.0 + 0.5 * np.cos(3.0 * y))
        truth[:, 5] = 1e-3 * (1.0 + 0.8 * np.sin(2.0 * x + y))
        deg = 0.02 + 0.02 * abs(aoa)
        pred = truth + deg * rng.standard_normal(truth.shape) * np.std(truth, axis=0, keepdims=True)
        d = {"x": s["x"], "y": s["y"], "is_wall": s["is_wall"], "u_init": s["u_init"], "v_init": s["v_init"]}
        rows.append(evaluate_sample(d, pred, truth, code, aoa, re, None))
    return rows


# --------------------------------------------------------------------------- #
# aggregation, correlation, plots
# --------------------------------------------------------------------------- #


def summarize(rows: list[dict], out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (out_dir / f"stage0_{tag}_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    print("\nPer-channel NRMSE (mean over samples):")
    for ch in CHANNELS:
        vals = np.array([r[f"nrmse_{ch}"] for r in rows])
        print(f"  {ch:6s}  mean {vals.mean():.4f}   median {np.median(vals):.4f}   max {vals.max():.4f}")
    cl_abs = np.array([r["cl_abs_err"] for r in rows])
    print(f"\nCl error (GNO vs integrated truth):  mean|dCl| {cl_abs.mean():.4f}  median {np.median(cl_abs):.4f}"
          f"  max {cl_abs.max():.4f}")

    print("\nCorrelation of mean field-NRMSE with inputs (Pearson r):")
    y = np.array([r["nrmse_mean"] for r in rows])
    for feat in ("aoa", "reynolds", "thickness", "camber"):
        x = np.array([r[feat] for r in rows], dtype=float)
        rr = float(np.corrcoef(x, y)[0, 1]) if np.ptp(x) > 0 else float("nan")
        print(f"  {feat:10s} r = {rr:+.3f}")

    worst = sorted(rows, key=lambda r: r["nrmse_mean"], reverse=True)[:5]
    print("\nWorst 5 by mean field-NRMSE:")
    for r in worst:
        print(f"  NACA {r['naca_code']}  Re{r['reynolds']/1e3:.0f}k  a{r['aoa']:+.1f}  "
              f"nrmse={r['nrmse_mean']:.4f}  dCl={r['cl_abs_err']:.3f}")

    _plots(rows, out_dir, tag)
    print(f"\nWrote stage0_{tag}_results.csv and plots to {out_dir}/")


def _plots(rows: list[dict], out_dir: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [[r[f"nrmse_{ch}"] for r in rows] for ch in CHANNELS]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(CHANNELS) + 1))
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("NRMSE"); ax.set_title(f"Stage 0 [{tag}]: per-channel field error")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"nrmse_per_channel_{tag}.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    y = [r["nrmse_mean"] for r in rows]
    for ax, feat, lab in ((axes[0], "aoa", "angle of attack [deg]"),
                          (axes[1], "reynolds", "Reynolds"),
                          (axes[2], "thickness", "thickness [%c]")):
        ax.scatter([r[feat] for r in rows], y, s=30, c="#2e6e8e", edgecolor="k", linewidth=0.4)
        ax.set_xlabel(lab); ax.set_ylabel("mean field-NRMSE"); ax.grid(alpha=0.3)
    axes[1].set_title(f"Stage 0 [{tag}]: where does the surrogate error grow?")
    fig.tight_layout(); fig.savefig(out_dir / f"error_vs_inputs_{tag}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    xt = [r["cl_truth"] for r in rows]; yg = [r["cl_gno"] for r in rows]
    ax.scatter(xt, yg, s=40, c="#b5461f", edgecolor="k", linewidth=0.4, zorder=3)
    lo, hi = min(xt + yg), max(xt + yg)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Cl (truth, integrated)"); ax.set_ylabel("Cl (GNO, integrated)")
    ax.set_title(f"Stage 0 [{tag}]: Cl, GNO vs truth (same integrator)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"cl_parity_{tag}.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/NACA_4_Digit_for_ML")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "ood"],
                    help="dataset split folder to evaluate (test = clean held-out)")
    ap.add_argument("--ckpt", default="checkpoints/gno_w32_d6_k16_lr0.001_NACA_4_Digit_for_ML/best.pt")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N samples")
    ap.add_argument("--shuffle", action="store_true", help="shuffle before --limit (spread across profiles)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (e.g. 20 on a 24-core VM)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--out-dir", default="output/stage0")
    ap.add_argument("--self-test", action="store_true", help="synthetic fields on real meshes; no torch/dataset")
    args = ap.parse_args()

    rows = run_self_test(args) if args.self_test else run_real(args)
    if not rows:
        raise SystemExit("No samples evaluated.")
    summarize(rows, Path(args.out_dir), "selftest" if args.self_test else args.split)


if __name__ == "__main__":
    main()
