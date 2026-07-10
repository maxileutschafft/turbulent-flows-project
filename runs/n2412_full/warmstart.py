"""Write a GNO prediction (.npz, same mesh) into the case 0/ as the initial field.

The GNO npz cell order != OpenFOAM cell order (gmshToFoam reorders), so we map by
cell-centre geometry. Run `postProcess -func writeCellCentres` first so that 0/C
exists, then this script maps each OpenFOAM cell to the nearest npz cell and
overwrites internalField in 0/{U,p,k,omega,nut}. Follow with `potentialFoam`
(makes U divergence-free) then `simpleFoam` for the warm-started run.
"""
import argparse, re
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def read_internal_vectors(path):
    txt = Path(path).read_text()
    j = txt.index("List<vector>", txt.index("internalField")) + len("List<vector>")
    k = txt.index("(", j)
    depth = 0
    for idx in range(k, len(txt)):
        if txt[idx] == "(":
            depth += 1
        elif txt[idx] == ")":
            depth -= 1
            if depth == 0:
                end = idx
                break
    vecs = re.findall(r"\(([^)]+)\)", txt[k + 1:end])
    return np.array([[float(t) for t in v.split()] for v in vecs])


def _set_internal(path, body):
    txt = Path(path).read_text()
    txt = re.sub(r"internalField\s+uniform\s+[^;]+;", f"internalField   {body};", txt, count=1)
    Path(path).write_text(txt)


def _scalar_body(vals):
    return "nonuniform List<scalar>\n%d\n(\n%s\n)" % (len(vals), "\n".join(f"{x:.8g}" for x in vals))


def _vector_body(vecs):
    return "nonuniform List<vector>\n%d\n(\n%s\n)" % (
        len(vecs), "\n".join(f"({a:.8g} {b:.8g} {c:.8g})" for a, b, c in vecs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True)
    ap.add_argument("--npz", required=True, help="GNO prediction .npz (save_predictions_npz)")
    args = ap.parse_args()
    case = Path(args.case)

    C = read_internal_vectors(case / "0" / "C")          # OpenFOAM cell centres [N,3]
    d = dict(np.load(args.npz, allow_pickle=True))
    _, nn = cKDTree(np.column_stack([d["x"], d["y"]])).query(C[:, :2])

    u, v = d["u"][nn], d["v"][nn]
    _set_internal(case / "0" / "U", _vector_body(np.column_stack([u, v, np.zeros_like(u)])))
    for fld in ("p", "k", "omega", "nut"):
        _set_internal(case / "0" / fld, _scalar_body(d[fld][nn]))
    print(f"warm-started 0/ from {args.npz} onto {len(C)} cells")


if __name__ == "__main__":
    main()
