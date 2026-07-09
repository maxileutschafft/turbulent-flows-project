"""Stage-1 screening: GNO vs. viscous-panel reference (XFOIL / NeuralFoil).

Pipeline:
    1. Build the DSD over the input space (dsd.py) and save the design.
    2. For every DSD run, get the reference Cl/Cd (XFOIL or NeuralFoil).
    3. For every DSD run, get the GNO Cl/Cd (server / csv / npz / demo).
    4. Merge, compute errors, and estimate DSD main effects: which input drives
       the surrogate error the most.
    5. Plots: Cl/Cd parity, per-run error bars, main-effect ranking, and a
       Cl-alpha sweep for the center airfoil (GNO vs reference, OOD shaded).

Run everything self-contained (no server, no XFOIL) with:
    python run_screening.py --gno-source demo --backend neuralfoil
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import dsd as dsd_mod
import aero_reference as aero
import gno_polar as gp


@dataclass
class Record:
    run_id: int
    naca_code: str
    reynolds: float
    aoa: float
    ood: bool
    cl_ref: float
    cd_ref: float
    cl_gno: float
    cd_gno: float
    ref_conf: float | None

    @property
    def dcl(self) -> float:
        return self.cl_gno - self.cl_ref

    @property
    def dcd(self) -> float:
        return self.cd_gno - self.cd_ref

    @property
    def rel_cl(self) -> float:
        return abs(self.dcl) / max(abs(self.cl_ref), 1e-6)

    @property
    def rel_cd(self) -> float:
        return abs(self.dcd) / max(abs(self.cd_ref), 1e-6)


# --------------------------------------------------------------------------- #
# demo GNO stand-in (no server needed): reference + OOD-growing degradation
# --------------------------------------------------------------------------- #


def demo_gno(ref: aero.Polar, rng: np.random.Generator) -> tuple[float, float]:
    aoa_ood = max(0.0, abs(ref.aoa) - 5.0) / 5.0          # >0 beyond +-5 deg
    re_ood = max(0.0, (ref.reynolds - 5.0e5) / 5.0e5) + max(0.0, (1.0e5 - ref.reynolds) / 1.0e5)
    t = int(str(ref.naca_code)[2:]) / 100.0
    thick = max(0.0, (t - 0.15) / 0.15)                    # thick airfoils harder
    deg = 0.03 + 0.30 * aoa_ood + 0.20 * re_ood + 0.10 * thick
    deg += 0.01 * rng.standard_normal()
    cl = ref.cl * (1.0 - deg) + 0.02 * rng.standard_normal()
    cd = ref.cd * (1.0 + 1.5 * deg)
    return cl, cd


# --------------------------------------------------------------------------- #
# main-effect estimation (orthogonal DSD -> simple projection)
# --------------------------------------------------------------------------- #


def main_effects(design: dsd_mod.DSDesign, response: dict[int, float]) -> list[tuple[str, float]]:
    names = [f.name for f in design.factors]
    coded = {n: [] for n in names}
    resp = []
    for run in design.runs:
        if run.run_id not in response or not np.isfinite(response[run.run_id]):
            continue
        for n in names:
            coded[n].append(run.coded[n])
        resp.append(response[run.run_id])
    resp = np.asarray(resp)
    effects = []
    for n in names:
        c = np.asarray(coded[n], dtype=float)
        denom = np.sum(c * c)
        eff = float(np.sum(c * resp) / denom) if denom > 0 else 0.0
        effects.append((n, eff))
    effects.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return effects


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #


def make_plots(records: list[Record], effects: list[tuple[str, float]], sweep_data, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fin = [r for r in records if np.isfinite(r.cl_ref) and np.isfinite(r.cl_gno)]

    # 1. parity Cl / Cd
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, title in ((axes[0], "cl", "C_l"), (axes[1], "cd", "C_d")):
        xr = [getattr(r, f"{key}_ref") for r in fin]
        yg = [getattr(r, f"{key}_gno") for r in fin]
        cols = ["#d1495b" if r.ood else "#2e8b8b" for r in fin]
        ax.scatter(xr, yg, c=cols, s=60, edgecolor="k", linewidth=0.5, zorder=3)
        lo, hi = min(xr + yg), max(xr + yg)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=1)
        ax.set_xlabel(f"{title} reference"); ax.set_ylabel(f"{title} GNO"); ax.set_title(f"{title} parity")
        ax.grid(alpha=0.3)
    axes[0].scatter([], [], c="#2e8b8b", label="in-distribution")
    axes[0].scatter([], [], c="#d1495b", label="OOD")
    axes[0].legend()
    fig.tight_layout(); fig.savefig(out_dir / "parity.png", dpi=130); plt.close(fig)

    # 2. per-run |dCl| bar
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fin_sorted = sorted(fin, key=lambda r: r.rel_cl, reverse=True)
    labels = [f"{r.naca_code}\nRe{r.reynolds/1e3:.0f}k a{r.aoa:+.0f}" for r in fin_sorted]
    vals = [100 * r.rel_cl for r in fin_sorted]
    cols = ["#d1495b" if r.ood else "#2e8b8b" for r in fin_sorted]
    ax.bar(range(len(vals)), vals, color=cols, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("|rel. C_l error|  [%]"); ax.set_title("GNO vs reference: per-run Cl error (worst first)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "per_run_error.png", dpi=130); plt.close(fig)

    # 3. main-effect ranking
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [e[0] for e in effects][::-1]
    vals = [e[1] for e in effects][::-1]
    ax.barh(names, vals, color="#41729f", edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("main effect on |rel. C_l error|  (coded units)")
    ax.set_title("DSD main effects: which input drives the surrogate error")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "main_effects.png", dpi=130); plt.close(fig)

    # 4. Cl-alpha sweep (center airfoil)
    if sweep_data is not None:
        code, alphas, cl_ref, cl_gno = sweep_data
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.axvspan(-5, 5, color="#2e8b8b", alpha=0.10, label="training AoA range")
        ax.plot(alphas, cl_ref, "-o", color="k", ms=3, label="reference (XFOIL/NeuralFoil)")
        m = np.isfinite(cl_gno)
        ax.plot(np.asarray(alphas)[m], np.asarray(cl_gno)[m], "-s", color="#d1495b", ms=3, label="GNO")
        ax.set_xlabel("angle of attack  [deg]"); ax.set_ylabel("C_l")
        ax.set_title(f"Cl-alpha, NACA {code}  (linear -> stall, OOD outside shaded band)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "cl_alpha_sweep.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="in_distribution", choices=list(dsd_mod.PRESETS))
    ap.add_argument("--backend", default="auto", choices=["auto", "xfoil", "neuralfoil"],
                    help="reference solver (auto: xfoil if on PATH, else neuralfoil)")
    ap.add_argument("--gno-source", default="demo", choices=["demo", "local", "csv", "server", "npz"])
    ap.add_argument("--gno-csv", default=None, help="csv with naca_code,reynolds,aoa,cl,cd")
    ap.add_argument("--server-url", default=None, help="base url of the inference webapp")
    ap.add_argument("--npz-dir", default=None, help="directory of prediction .npz files")
    ap.add_argument("--ckpt", default="checkpoints/gno_w32_d6_k16_lr0.001_NACA_4_Digit_for_ML/best.pt",
                    help="checkpoint for --gno-source local")
    ap.add_argument("--out-dir", default="output/stage1")
    ap.add_argument("--xtr", type=float, default=None,
                    help="force transition x/c on both surfaces (e.g. 0.05) to emulate fully-turbulent RANS")
    ap.add_argument("--model-size", default="large", help="NeuralFoil model size")
    ap.add_argument("--no-sweep", action="store_true", help="skip the Cl-alpha sweep plot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. design
    design = dsd_mod.build_dsd(args.preset)
    design.write_csv(out_dir / "dsd_design.csv")
    design.write_json(out_dir / "dsd_design.json")
    backend = aero.resolve_backend(args.backend)
    print(f"DSD preset={args.preset}  runs={len(design.runs)}  reference backend={backend}  "
          f"gno-source={args.gno_source}")
    checks = design.verify()
    print("  DSD verify: " + ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items()))

    # 2/3. reference + gno per run
    if args.gno_source == "csv":
        gno_lookup = gp.from_csv(args.gno_csv)
    elif args.gno_source == "server":
        gno_lookup = gp.from_server(args.server_url)
    elif args.gno_source == "npz":
        gno_lookup = gp.from_prediction_dir(args.npz_dir)
    elif args.gno_source == "local":
        gno_lookup = gp.from_local(args.ckpt)
    else:
        gno_lookup = None  # demo

    records: list[Record] = []
    ref_kw = dict(xtr=args.xtr, model_size=args.model_size)
    for run in design.runs:
        ref = aero.evaluate(run.naca_code, run.reynolds, run.aoa, backend=backend,
                            **{k: v for k, v in ref_kw.items() if v is not None or k == "xtr"})
        if args.gno_source == "demo":
            cl_g, cd_g = demo_gno(ref, rng)
        else:
            g = gno_lookup(run.naca_code, run.reynolds, run.aoa)
            cl_g, cd_g = (g.cl, g.cd) if g is not None else (np.nan, np.nan)
        records.append(Record(run.run_id, run.naca_code, run.reynolds, run.aoa, run.ood,
                              ref.cl, ref.cd, cl_g, cd_g, ref.confidence))
        flag = " OOD" if run.ood else ""
        print(f"  run {run.run_id:>2} NACA {run.naca_code} Re{run.reynolds/1e3:>4.0f}k a{run.aoa:>+5.1f}{flag:>4}"
              f" | Cl ref {ref.cl:+.3f} gno {cl_g:+.3f} | Cd ref {ref.cd:.4f} gno {cd_g:.4f}")

    # 4. results csv + main effects
    with (out_dir / "results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "naca_code", "reynolds", "aoa", "ood",
                    "cl_ref", "cd_ref", "cl_gno", "cd_gno", "dcl", "dcd", "rel_cl", "rel_cd"])
        for r in records:
            w.writerow([r.run_id, r.naca_code, r.reynolds, r.aoa, int(r.ood),
                        f"{r.cl_ref:.5f}", f"{r.cd_ref:.5f}", f"{r.cl_gno:.5f}", f"{r.cd_gno:.5f}",
                        f"{r.dcl:.5f}", f"{r.dcd:.5f}", f"{r.rel_cl:.5f}", f"{r.rel_cd:.5f}"])

    response = {r.run_id: r.rel_cl for r in records}
    effects = main_effects(design, response)
    print("\nDSD main effects on |rel. Cl error| (largest first):")
    for name, eff in effects:
        print(f"  {name:9s} {eff:+.4f}")

    # sweep for center airfoil
    sweep_data = None
    if not args.no_sweep:
        code = "4415"
        re = 3.0e5
        alphas = np.arange(-12, 16.001, 1.0)
        cl_ref = []
        cl_gno = []
        for a in alphas:
            ref = aero.evaluate(code, re, float(a), backend=backend,
                                **{k: v for k, v in ref_kw.items() if v is not None or k == "xtr"})
            cl_ref.append(ref.cl)
            if args.gno_source == "demo":
                cl_gno.append(demo_gno(ref, rng)[0])
            else:
                g = gno_lookup(code, re, float(a))
                cl_gno.append(g.cl if g is not None else np.nan)
        sweep_data = (code, alphas, np.asarray(cl_ref), np.asarray(cl_gno))

    make_plots(records, effects, sweep_data, out_dir)
    print(f"\nWrote design, results.csv and plots to {out_dir}/")


if __name__ == "__main__":
    main()
