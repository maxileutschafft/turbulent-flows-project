"""Run the GNO on a scenario and save the prediction as .npz (for the warm-start)."""
import sys as _sys
try:
    _sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pathlib
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath
from utils.scenario import build_scenario
from surrogate.gno.infer import infer
from utils.inference.io import save_predictions_npz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--naca", required=True)
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--aoa", type=float, required=True)
    ap.add_argument("--ckpt", default="checkpoints/gno_w32_d6_k16_lr0.001_NACA_4_Digit_for_ML/best.pt")
    ap.add_argument("--out", default="pred.npz")
    args = ap.parse_args()
    scen = build_scenario(args.naca, args.re, args.aoa)
    res = infer(scen, Path(args.ckpt))
    save_predictions_npz(res, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
