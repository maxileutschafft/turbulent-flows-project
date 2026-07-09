# Stage 1 — Screening the GNO surrogate against a viscous-panel reference

This package screens where the trained GNO surrogate departs from an independent
aerodynamic reference (XFOIL / NeuralFoil), driven by a **Definitive Screening
Design (DSD)** over the airfoil input space. It is the cheap, laptop-only first
layer of the testing plan; it needs no CFD and no license.

- **GNO prediction** = your deployed server/webapp output (or local `infer()`).
- **Reference** = XFOIL (real binary) or NeuralFoil (pure-numpy XFOIL surrogate).
- **Design** = a 13-run DSD over NACA geometry + Re + AoA.

```
stage1_screening/
├── dsd.py             # DSD generator + factor bounds + self-verification
├── naca.py            # NACA-4 surface coords (Selig order) for the reference
├── aero_reference.py  # XFOIL + NeuralFoil backends -> Cl, Cd, Cm
├── gno_polar.py       # GNO Cl/Cd from server / csv / prediction .npz
├── run_screening.py   # orchestrator: design -> polars -> errors -> plots
└── example_output/    # example run (demo GNO), so you can see the outputs
```

## Quick start

Fully self-contained (no server, no XFOIL — uses NeuralFoil + a synthetic GNO):

```bash
uv run python stage1_screening/run_screening.py --gno-source demo --backend neuralfoil
```

Just print/inspect the design:

```bash
uv run python stage1_screening/dsd.py --preset in_distribution
```

Real run against your deployed model:

```bash
uv run python stage1_screening/run_screening.py \
    --preset in_distribution \
    --backend xfoil \
    --gno-source server --server-url http://<your-uchile-host>:<port>
```

Dependencies (optional extras): `neuralfoil` (reference), `matplotlib` (plots).
`xfoil` must be a binary on `PATH` if you use `--backend xfoil`.

```bash
uv add neuralfoil matplotlib
```

## The Definitive Screening Design

5 factors — the three NACA-4 digits plus the two flow parameters. Coded levels
map −1 / 0 / +1 to low / center / high. A conference matrix of order 6 gives
**13 runs** (the 6th column is a ghost factor for pure-error d.o.f.).

### Proposed parameter bounds

**Preset `in_distribution`** — quantify interpolation quality *inside* the
training envelope (Re 1e5–5e5, AoA ±5°):

| Factor | low (−1) | center (0) | high (+1) | unit | note |
|---|---:|---:|---:|---|---|
| `m` max camber        | 2 | 4 | 6 | % chord | 1st NACA digit |
| `p` camber position   | 2 | 4 | 6 | 1/10 chord | 2nd NACA digit |
| `t` thickness         | 9 | 15 | 21 | % chord | digits 3–4 |
| `reynolds`            | 1.0e5 | 3.0e5 | 5.0e5 | – | training box |
| `aoa`                 | −5 | 0 | +5 | deg | training box |

**Preset `ood_probe`** — push the edges just past training to see how fast the
error grows leaving the envelope:

| Factor | low (−1) | center (0) | high (+1) | unit |
|---|---:|---:|---:|---|
| `m` | 2 | 4 | 6 | % chord |
| `p` | 2 | 4 | 6 | 1/10 chord |
| `t` | 6 | 15 | 24 | % chord |
| `reynolds` | 1.0e5 | 5.0e5 | 9.0e5 | – |
| `aoa` | −10 | 0 | +10 | deg |

Why these bounds:

- **Camber `m` kept ≥ 2.** For a symmetric airfoil (`m = 0`) the camber-position
  factor `p` has no effect, which would confound its DSD main-effect estimate.
  Keeping camber present at every run keeps all five factors estimable.
- **`p` and `t`** span realistic 4-digit geometry (max-camber at 20–60 % chord;
  thickness NACA xx06–xx24).
- **`reynolds` / `aoa`** sit on the training box for the in-distribution screen;
  `ood_probe` steps the edges past ±5° / 5e5 so the corners graze OOD.
- Runs whose Re or AoA fall outside the training envelope are flagged `ood=1`.

> **Adjust `t`, `m`, `p` bounds to match the actual geometry range in your
> training set** (`Kokoslocke/NACA_4_Digit_for_ML`). The flow bounds already
> match the Re 100k–500k / AoA ±5° range you trained on.

### What a DSD can and cannot tell you

A DSD is a **screening** design: with few runs it estimates each factor's main
effect and curvature and tells you **which inputs drive the surrogate error**.
It does **not** by itself localise *where* the error peaks — the largest error
usually sits at the domain edges, which a DSD deliberately under-samples. So:

1. Run the DSD → rank the factors (main-effect plot).
2. Follow the dominant factors with targeted edge / OOD sweeps (`ood_probe`,
   the Cl-α sweep) and, in Stage 2, an adaptive (Bayesian) search + CFD.

`dsd.py` self-verifies the design on every run (conference orthogonality, three
levels per factor, balanced columns, main effects orthogonal to each other and
to all quadratic terms).

## Reference: XFOIL vs NeuralFoil, and the fidelity caveat

Both backends are viscous panel / integral-boundary-layer methods with a
transition model. **Your GNO learned RANS k-ω data, usually run fully
turbulent.** They will *not* match absolute Cd, especially at low Re, because
XFOIL/NeuralFoil model laminar–turbulent transition. Two consequences:

- Use the reference for the **shape** of the Cl-α curve (linear slope ≈ 2π,
  stall onset) and for **trends / OOD**, not for matching Cd to the RANS truth.
- For a fairer drag comparison, trip transition near the leading edge to emulate
  fully-turbulent RANS: `--xtr 0.05`.

For a same-fidelity, field-level ground truth (u, v, p, k, ω, νt), that is
**Stage 2** (OpenFOAM / SU2 on the cluster) — not this package.

## GNO Cl/Cd source

The server prediction is the intended source. `gno_polar.py` offers three:

- `--gno-source server --server-url ...` — POST each case to your webapp. Edit
  `from_server`'s `endpoint` / `build_request` / `parse_response` to match your
  API. CPU inference on the cluster ⇒ a few seconds per call.
- `--gno-source csv --gno-csv table.csv` — a precomputed
  `naca_code,reynolds,aoa,cl,cd` table.
- `--gno-source npz --npz-dir dir/` — *experimental* pressure-only integration
  from `save_predictions_npz` files (no friction drag → Cd biased low). Prefer
  your validated server-side Cl/Cd postprocessing.

## Outputs

Written to `--out-dir` (default `output/stage1/`):

| File | Content |
|---|---|
| `dsd_design.csv` / `.json` | the 13-run design, coded + physical + OOD flag |
| `results.csv` | per run: Cl/Cd reference vs GNO, Δ and relative error |
| `parity.png` | Cl and Cd parity (GNO vs reference), OOD points highlighted |
| `per_run_error.png` | per-run |rel. Cl error|, worst first |
| `main_effects.png` | **which input drives the Cl error** (DSD main effects) |
| `cl_alpha_sweep.png` | Cl-α for the center airfoil, training band shaded |

The `main_effects.png` and the Cl-α sweep are the two presentation money-plots:
the first says *which* parameter hurts, the second shows *where* the model
breaks as you leave the training AoA range.

## Next: Stage 2

Feed the worst runs identified here into OpenFOAM/SU2 on the UChile cluster for
same-fidelity field ground truth and the solver warm-start study.
