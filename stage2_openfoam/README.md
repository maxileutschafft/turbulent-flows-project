# Stage 2 — same-fidelity RANS ground truth + solver warm-start (OpenFOAM)

Runs `simpleFoam` (k-ω SST) on the **exact same C-mesh** the GNO predicts on, so
you get node-for-node field comparison (no interpolation) and a clean warm-start.
The mesh comes from `build_scenario`, so a converged OpenFOAM run should closely
reproduce the dataset's stored truth (which was made the same way: OpenFOAM,
simpleFoam, k-ω SST) — that doubles as a check of the setup before you run OOD
cases.

> OpenFOAM can't be run/tested in this repo's dev sandbox — these scripts are
> written to OpenFOAM conventions but **validate them on your cluster/VM**.

## Pipeline

```
build_scenario ──> mesh.h5 ──mesh_to_foam.py──> mesh.msh ──gmshToFoam──> polyMesh
                                                                     │
make_case.py ──> 0/ constant/ system/  ◄────────────────────────────┘
                     │
   ./Allrun (freestream)         ./Allrun.warmstart pred.npz (GNO init)
```

`mesh_to_foam.py` extrudes the 2D C-mesh to one cell layer and tags patches
(`airfoil`, `farfield`, `frontAndBack`); `gmshToFoam` builds the polyMesh and
`set_patch_types.py` sets `airfoil→wall`, `frontAndBack→empty`.

## Usage

**1. Prepare a case** (uv env — needs numpy/scipy/h5py + the repo):
```bash
uv run python stage2_openfoam/prepare_case.py --case runs/naca2412_re3e5_a5 \
    --naca 2412 --re 3e5 --aoa 5
```

**2. Freestream baseline** (OpenFOAM env):
```bash
cd runs/naca2412_re3e5_a5 && ./Allrun
# Cl/Cd -> postProcessing/forceCoeffs/ ; residuals -> log.simpleFoam
```

**3. Warm-started run** (the GNO prediction as initial field):
```bash
# generate the GNO prediction on the same scenario (uv env):
uv run python stage2_openfoam/predict_npz.py --naca 2412 --re 3e5 --aoa 5 \
    --out runs/naca2412_re3e5_a5/pred.npz
# then warm-start (OpenFOAM env):
cd runs/naca2412_re3e5_a5 && ./Allrun.warmstart pred.npz
```

**4. Compare convergence** (freestream vs warm-start):
```bash
uv run python stage2_openfoam/compare_residuals.py \
    --freestream runs/naca2412_re3e5_a5/log.simpleFoam \
    --warmstart  runs/naca2412_re3e5_a5/log.simpleFoam.warmstart
```

## Case-farm (many cases in parallel on the VM)

The cases are independent — run several at once (like Stage 0's `--jobs`):
```bash
# prepare a batch, then:
ls -d runs/*/ | xargs -P 8 -I{} sh -c 'cd {} && ./Allrun'
```

## Physics / fidelity

- `nu = 1e-5`, `chord = 1`, `U_mag = Re·ν/chord`, freestream angled by AoA
  (airfoil fixed) — identical to `scenario.py`.
- k-ω SST, wall functions on the airfoil, `freestream` BCs on the farfield.
- `forceCoeffs` writes Cl/Cd with `liftDir`/`dragDir` rotated by AoA.

## OpenFOAM version note

Dicts use the **stable ESI/.com syntax** (`transportProperties`,
`turbulenceProperties`/`RASModel`). The dataset was made with **OpenFOAM.org v13**;
if you use .org v11+, rename: `transportProperties→physicalProperties`,
`turbulenceProperties→momentumTransport`, `RASModel→model`. The physics is
identical either way.

## OOD caveat

Deep OOD (high AoA / stall, high Re) — steady `simpleFoam` may not converge
(separation, unsteadiness), the same reason your own mesh diverged. There,
switch to pseudo-transient / `pimpleFoam`, or accept that no clean steady truth
exists and fall back to Stage 1 (XFOIL/experiment).
