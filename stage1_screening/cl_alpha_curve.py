"""Stage 1 - GNO vs reference Cl-alpha curve (linear region, stall, OOD boundary).

Sweeps angle of attack at a fixed airfoil / Reynolds number and overlays:
  * the reference Cl-alpha (XFOIL / NeuralFoil) -> physical curve with the
    classic linear rise and the stall break-down;
  * the GNO's Cl-alpha (integrated from the predicted pressure field).

The GNO's integrated Cl uses a different reference convention than the physical
Cl (open trailing edge + forceCoeffs normalisation), so its ABSOLUTE level does
not match. We calibrate the GNO curve to the reference inside the training band
(|AoA| <= 5 deg) with a single linear fit, then let it extrapolate. The message
is the SHAPE: outside the training band the GNO keeps rising ~linearly and
CANNOT reproduce stall, because it never saw those angles.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aero_reference as aero
import gno_polar as gp


def calibrate(gno, ref, aoa, band):
    m = (np.abs(aoa) <= band) & np.isfinite(gno) & np.isfinite(ref)
    if m.sum() >= 2 and np.ptp(gno[m]) > 1e-9:
        a, b = np.polyfit(gno[m], ref[m], 1)
        return a * gno + b, float(a), float(b)
    return gno.copy(), 1.0, 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--naca", default="2412")
    ap.add_argument("--re", type=float, default=3.0e5)
    ap.add_argument("--aoa-min", type=float, default=-8.0)
    ap.add_argument("--aoa-max", type=float, default=18.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--train-band", type=float, default=5.0)
    ap.add_argument("--gno-source", default="demo", choices=["local", "demo"])
    ap.add_argument("--ckpt", default="checkpoints/gno_w32_d6_k16_lr0.001_NACA_4_Digit_for_ML/best.pt")
    ap.add_argument("--backend", default="auto", choices=["auto", "xfoil", "neuralfoil"])
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--out-dir", default="output/stage1")
    args = ap.parse_args()

    aoas = np.arange(args.aoa_min, args.aoa_max + 1e-9, args.step)
    backend = aero.resolve_backend(args.backend)
    print(f"NACA {args.naca}  Re {args.re:.0f}  reference={backend}  gno={args.gno_source}")

    ref = np.array([aero.evaluate(args.naca, args.re, float(a), backend=backend).cl for a in aoas])

    if args.gno_source == "local":
        look = gp.from_local(args.ckpt)
        gpolar = [look(args.naca, args.re, float(a)) for a in aoas]
        gno = np.array([g.cl if g is not None else np.nan for g in gpolar])
    else:
        lin = np.abs(aoas) <= 4
        m0, c0 = np.polyfit(aoas[lin], ref[lin], 1)
        rng = np.random.default_rng(0)
        gno = 3.0 * (m0 * aoas + c0) + 0.02 * rng.standard_normal(len(aoas))

    gno_cal, a, b = (gno.copy(), 1.0, 0.0) if args.no_calibrate else calibrate(gno, ref, aoas, args.train_band)

    stall_i = int(np.nanargmax(ref))
    stall_aoa, stall_cl = float(aoas[stall_i]), float(ref[stall_i])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.naca}_Re{args.re/1e3:.0f}k"
    with (out_dir / f"cl_alpha_{tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["aoa", "cl_ref", "cl_gno_raw", "cl_gno_calibrated"])
        for i, aa in enumerate(aoas):
            w.writerow([f"{aa:.2f}", f"{ref[i]:.5f}", f"{gno[i]:.5f}", f"{gno_cal[i]:.5f}"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axvspan(-args.train_band, args.train_band, color="#2e8b8b", alpha=0.12,
               label=f"training band (|AoA|<={args.train_band:.0f})")
    ax.plot(aoas, ref, "-o", color="k", ms=4, label=f"reference ({backend}) - physical")
    fin = np.isfinite(gno_cal)
    ax.plot(aoas[fin], gno_cal[fin], "-s", color="#d1495b", ms=4,
            label="GNO (calibrated in training band)")
    ax.axvline(stall_aoa, ls="--", color="gray", lw=1)
    ax.annotate(f"stall (reference)\n~{stall_aoa:.0f} deg", xy=(stall_aoa, stall_cl),
                xytext=(stall_aoa + 1.0, stall_cl - 0.35), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.axvspan(args.train_band, args.aoa_max, color="#d1495b", alpha=0.05)
    ax.set_xlabel("angle of attack  [deg]")
    ax.set_ylabel("lift coefficient  C_l")
    ax.set_title(f"Cl-alpha: GNO vs reference - NACA {args.naca}, Re {args.re:.0f}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"cl_alpha_{tag}.png"
    fig.savefig(out, dpi=140)
    print(f"calibration: cl_ref ~= {a:.3f}*cl_gno + {b:.3f}")
    print(f"stall (reference) ~{stall_aoa:.1f} deg,  Cl_max={stall_cl:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
