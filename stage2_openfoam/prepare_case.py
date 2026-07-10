"""Prepare an OpenFOAM case: build_scenario mesh -> .msh -> case dicts.
Run in the uv env (needs numpy/scipy/h5py + the repo). Then run ./Allrun in the
OpenFOAM environment.
"""
import sys as _sys
try:
    _sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
import argparse, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from utils.scenario import build_scenario
import mesh_to_foam
import make_case


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True)
    ap.add_argument("--naca", required=True)
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--aoa", type=float, required=True)
    ap.add_argument("--end-time", type=int, default=3000)
    ap.add_argument("--dz", type=float, default=0.1)
    args = ap.parse_args()
    case = Path(args.case)
    case.mkdir(parents=True, exist_ok=True)

    mesh_to_foam.dump_full_mesh(args.naca, args.aoa, case / "mesh.h5")
    s = mesh_to_foam.convert(str(case / "mesh.h5"), str(case / "mesh.msh"), dz=args.dz)
    print(f"mesh: cells={s['cells']} airfoil={s['airfoil']} farfield={s['farfield']}")
    make_case.write_case(case, args.naca, args.re, args.aoa, args.end_time, dz=args.dz)
    for helper in ("set_patch_types.py", "warmstart.py", "Allrun", "Allrun.warmstart"):
        shutil.copy(HERE / helper, case / helper)
    (case / "Allrun").chmod(0o755)
    (case / "Allrun.warmstart").chmod(0o755)
    print(f"\nprepared {case}\n  freestream run : cd {case} && ./Allrun")
    print(f"  warm-start run : predict npz, then  ./Allrun.warmstart pred.npz")


if __name__ == "__main__":
    main()
