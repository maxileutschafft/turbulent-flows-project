"""Build the in-distribution (test) vs out-of-distribution (ood) comparison figure."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "naca_code"}

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/stage0/vm")
test, ood = load(d/"stage0_test_results.csv"), load(d/"stage0_ood_results.csv")
chans = ["u", "v", "p", "k", "omega", "nut"]
tm = [test[f"nrmse_{c}"].mean() for c in chans]
om = [ood[f"nrmse_{c}"].mean() for c in chans]
TE, OO = "#2e8b8b", "#d1495b"

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

# 1) per-channel bars (log)
x = np.arange(len(chans)); w = 0.38
ax[0].bar(x-w/2, tm, w, label="test (in-distribution)", color=TE)
ax[0].bar(x+w/2, om, w, label="ood (out-of-distribution)", color=OO)
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(chans)
ax[0].set_ylabel("mean NRMSE (log scale)")
ax[0].set_title("Per-channel field error: in-dist vs OOD")
ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)
for i,(t,o) in enumerate(zip(tm,om)):
    ax[0].text(i+w/2, o*1.15, f"{o/t:.0f}x", ha="center", fontsize=8, color=OO)

# 2) mean NRMSE vs Reynolds (correlation sign flip)
ax[1].scatter(test["reynolds"], test["nrmse_mean"], s=16, c=TE, alpha=0.7, label="test")
ax[1].scatter(ood["reynolds"], ood["nrmse_mean"], s=16, c=OO, alpha=0.7, label="ood")
ax[1].set_xscale("log"); ax[1].set_xlabel("Reynolds number"); ax[1].set_ylabel("mean field-NRMSE")
ax[1].set_title("Error vs Reynolds — correlation flips sign")
ax[1].legend(); ax[1].grid(alpha=0.3)
rt = np.corrcoef(test["reynolds"], test["nrmse_mean"])[0,1]
ro = np.corrcoef(ood["reynolds"], ood["nrmse_mean"])[0,1]
ax[1].text(0.04, 0.96, f"test: r = {rt:+.2f}\nood:  r = {ro:+.2f}", transform=ax[1].transAxes,
           va="top", fontsize=9, bbox=dict(fc="white", alpha=0.8))

# 3) Cl error
tcl, ocl = test["cl_abs_err"].mean(), ood["cl_abs_err"].mean()
ax[2].bar(["test", "ood"], [tcl, ocl], color=[TE, OO])
ax[2].set_ylabel("mean |ΔCl|  (GNO vs truth)")
ax[2].set_title(f"Lift-coefficient error: {ocl/tcl:.1f}x worse OOD")
ax[2].grid(axis="y", alpha=0.3)
for i, v in enumerate([tcl, ocl]):
    ax[2].text(i, v, f"{v:.3f}", ha="center", va="bottom")

fig.suptitle("GNO surrogate — in-distribution (test, 133) vs out-of-distribution (ood, 129)",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
out = Path("output/stage0/stage0_test_vs_ood.png")
fig.savefig(out, dpi=140); print("wrote", out)
# also print the table
print(f"\n{'chan':6} {'test':>8} {'ood':>8} {'factor':>7}")
for c,t,o in zip(chans,tm,om): print(f"{c:6} {t:8.4f} {o:8.4f} {o/t:6.1f}x")
print(f"Cl |dCl|: test {tcl:.4f}  ood {ocl:.4f}  ({ocl/tcl:.1f}x)")
