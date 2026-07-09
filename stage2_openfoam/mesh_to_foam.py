"""Convert a build_scenario mesh dump (HDF5) -> Gmsh .msh for `gmshToFoam`.

The scenario C-mesh is 2D structured quads. We extrude it to ONE cell layer in
z (OpenFOAM's 2D convention) and emit a Gmsh 2.2 ASCII mesh with:
  * hexahedra (the extruded quads),
  * tagged boundary quads for the patches: airfoil, farfield, frontAndBack.
`gmshToFoam` then builds the polyMesh (faces, owner/neighbour, ordering) - the
error-prone part is left to that tested tool. A post-step sets patch types
(airfoil -> wall, frontAndBack -> empty); see set_patch_types() / the Allrun.

Produce the input with:
    build_scenario(naca, Re, AoA, save_mesh_to="mesh.h5")
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.naca_geometry import naca4_coords


def naca_surface_points(code: str, n: int = 400) -> np.ndarray:
    xu, yu, xl, yl = naca4_coords(code, n=n)
    return np.column_stack([np.concatenate([xu, xl]), np.concatenate([yu, yl])])


def convert(mesh_h5: str, out: str, dz: float = 0.1, wall_tol: float = 0.02) -> dict:
    from scipy.spatial import cKDTree

    with h5py.File(mesh_h5, "r") as f:
        pts = np.asarray(f["points"], dtype=float)          # [Nn, 2]
        conn = np.asarray(f["connectivity"], dtype=np.int64)  # [Nc, 4] bl,br,tr,tl
        naca = f.attrs["naca_code"]
    naca = naca.decode() if isinstance(naca, bytes) else str(naca)
    Nn, Nc = pts.shape[0], conn.shape[0]

    nodes3d = np.vstack([np.column_stack([pts, np.zeros(Nn)]),
                         np.column_stack([pts, np.full(Nn, dz)])])   # [2Nn, 3]

    # boundary edges = 2D edges used by exactly one cell
    edge_count: dict = defaultdict(int)
    edge_dir: dict = {}
    for bl, br, tr, tl in conn:
        for a, b in ((bl, br), (br, tr), (tr, tl), (tl, bl)):
            k = (a, b) if a < b else (b, a)
            edge_count[k] += 1
            edge_dir.setdefault(k, (a, b))
    boundary = [edge_dir[k] for k, c in edge_count.items() if c == 1]

    # classify airfoil vs farfield by midpoint distance to the NACA surface
    tree = cKDTree(naca_surface_points(naca))
    airfoil, farfield = [], []
    for a, b in boundary:
        d, _ = tree.query(0.5 * (pts[a] + pts[b]))
        (airfoil if d < wall_tol else farfield).append((a, b))

    # write gmsh 2.2
    L = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
         "$PhysicalNames", "4",
         '3 1 "internal"', '2 2 "airfoil"', '2 3 "farfield"', '2 4 "frontAndBack"',
         "$EndPhysicalNames", "$Nodes", str(2 * Nn)]
    for i, (x, y, z) in enumerate(nodes3d, 1):
        L.append(f"{i} {x:.9g} {y:.9g} {z:.9g}")
    L.append("$EndNodes")

    elems, eid = [], 0
    def add(typ, phys, ids):
        nonlocal eid
        eid += 1
        elems.append(f"{eid} {typ} 2 {phys} {phys} " + " ".join(str(n + 1) for n in ids))

    for bl, br, tr, tl in conn:                                  # hexes
        add(5, 1, [bl, br, tr, tl, bl + Nn, br + Nn, tr + Nn, tl + Nn])
    for a, b in airfoil:                                          # airfoil side faces
        add(3, 2, [a, b, b + Nn, a + Nn])
    for a, b in farfield:                                         # farfield side faces
        add(3, 3, [a, b, b + Nn, a + Nn])
    for bl, br, tr, tl in conn:                                   # frontAndBack (both z-faces)
        add(3, 4, [bl, br, tr, tl])
        add(3, 4, [bl + Nn, br + Nn, tr + Nn, tl + Nn])

    L += ["$Elements", str(len(elems))] + elems + ["$EndElements"]
    Path(out).write_text("\n".join(L) + "\n")
    stats = dict(naca=naca, nodes2d=Nn, cells=Nc, boundary=len(boundary),
                 airfoil=len(airfoil), farfield=len(farfield), elements=len(elems))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh_h5")
    ap.add_argument("--out", default="mesh.msh")
    ap.add_argument("--dz", type=float, default=0.1)
    ap.add_argument("--wall-tol", type=float, default=0.02)
    args = ap.parse_args()
    s = convert(args.mesh_h5, args.out, dz=args.dz, wall_tol=args.wall_tol)
    print(f"NACA {s['naca']}: nodes2D={s['nodes2d']} cells={s['cells']}  "
          f"boundary edges={s['boundary']} (airfoil={s['airfoil']}, farfield={s['farfield']})")
    print(f"wrote {args.out}: {2*s['nodes2d']} nodes, {s['elements']} elements")


if __name__ == "__main__":
    main()
