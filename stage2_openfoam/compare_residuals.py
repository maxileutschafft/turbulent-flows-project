"""Compare simpleFoam residual histories: freestream-init vs GNO warm-start.

Parses two solver logs (foamLog-style 'Solving for X, Initial residual = R')
and plots per-field residual vs iteration + reports iterations-to-converge.
"""
import argparse, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS = ["Ux", "Uy", "p", "k", "omega"]
_PAT = re.compile(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)")


def parse(log):
    hist = {f: [] for f in FIELDS}
    for line in Path(log).read_text(errors="ignore").splitlines():
        m = _PAT.search(line)
        if m and m.group(1) in hist:
            hist[m.group(1)].append(float(m.group(2)))
    return hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freestream", required=True)
    ap.add_argument("--warmstart", required=True)
    ap.add_argument("--out", default="residual_compare.png")
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()
    fs, ws = parse(args.freestream), parse(args.warmstart)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for f, c in zip(FIELDS, ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]):
        if fs[f]:
            ax.semilogy(fs[f], "--", color=c, alpha=0.6, lw=1)
        if ws[f]:
            ax.semilogy(ws[f], "-", color=c, lw=1.6, label=f)
    ax.axhline(args.tol, color="k", ls=":", lw=1)
    ax.set_xlabel("SIMPLE iteration"); ax.set_ylabel("initial residual")
    ax.set_title("Residuals: dashed = freestream init, solid = GNO warm-start")
    ax.legend(ncol=5, fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(args.out, dpi=140)

    def iters_to(hist, tol):
        p = hist.get("p", [])
        for i, r in enumerate(p):
            if r < tol:
                return i
        return len(p)
    print(f"iterations to p<{args.tol}:  freestream={iters_to(fs,args.tol)}  warmstart={iters_to(ws,args.tol)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
