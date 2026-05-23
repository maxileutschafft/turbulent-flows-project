# Turbulent Flows Term Project

Inference-only repository for a Graph Neural Operator (GNO) surrogate that predicts steady-state RANS solutions (`u, v, p, k, ω, νᵗ`) over NACA 4-digit airfoil meshes. The model is a `KernelNN` graph network trained externally; this repo packages the forward pass, the analytical mesh / signed-distance generator needed to feed it, and two Jupyter notebooks that exercise both pre-computed and user-defined scenarios.


## Getting Started

Install `uv` if it is not already available (see [docs.astral.sh/uv](https://docs.astral.sh/uv/) for alternative installers):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environment

Create and populate the project environment from the lockfile:

```bash
uv sync
```

For development tools (pytest, ruff), include the optional extras:

```bash
uv sync --extra dev
```

The preferred invocation pattern is `uv run <cmd>`, which executes inside the project environment without requiring manual activation:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If you prefer a sourced shell, activate the venv that `uv` created:

```bash
source .venv/bin/activate
```

### Troubleshooting on PyTorch wheel index
`pyproject.toml` pins PyTorch to the CUDA 12.8 wheel index:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

CUDA minor-version compatibility lets `cu128` wheels run on any driver `>= 525` (Linux), including the common `r550` / CUDA 12.4 driver stack. CPU-only users should remove both the `[[tool.uv.index]]` and the `[tool.uv.sources]` block for `torch` before running `uv sync`; `uv` will then resolve the default CPU wheel.


## Required Data Assets

### Model Checkpoint
The model checkpoint is included in git and should be in:

```text
checkpoints/gno_w32_d6_k16_lr0.001_NACA_4_Digit_for_ML/best.pt
```

### Scenario dataset

The simulated NACA dataset used for training is hosted on Hugging Face:

```text
Kokoslocke/NACA_4_Digit_for_ML
```

Download it into `data/raw/NACA_4_Digit_for_ML/` to run inference on samples of it:

```bash
uv run python scripts/download_dataset.py
```


## Usage — Jupyter Notebooks

Launch Jupyter inside the project environment:

```bash
uv run jupyter notebook
# or
uv run jupyter lab
```

For a headless end-to-end execution:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace inference_ground_truth.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace inference_custom_scenario.ipynb
```

### `inference_ground_truth.ipynb`

Runs the surrogate against a scenario `.npz` shipped with the dataset. The file already contains CFD-computed ground-truth fields, so the notebook can render per-channel error maps in addition to the predictions.

What the notebook does:

1. Loads the checkpoint and the scenario from configured paths.
2. Builds the required k-NN graph from the scenario node coordinates.
3. Runs `infer()` on the auto-selected device.
4. Plots scalar contours, streamlines, and 3-panel `pred / truth / |error|` maps for all six output channels.

Outputs land under `output/` (gitignored).

### `inference_custom_scenario.ipynb`

Takes a user-defined airfoil and freestream condition, generates the mesh and signed-distance field on the fly, runs inference, and plots the predicted fields. No ground truth is available, so no error maps.

The three user knobs are set in the configuration cell:

| Variable          | Type   | Description                                                       |
|-------------------|--------|-------------------------------------------------------------------|
| `naca_code`       | `str`  | NACA 4-digit identifier, e.g. `"2210"`.                           |
| `reynolds`        | `float`| Chord-based Reynolds number (sets freestream magnitude).          |
| `angle_of_attack` | `float`| Angle of attack in degrees, signed (positive = nose up).          |

The device (`cuda` / `mps` / `cpu`) is auto-selected. Set the optional `save_mesh_to` argument to write the generated structured mesh to `output/custom_mesh.h5` for downstream inspection.

## Mesh and SDF Generation

`src/utils/scenario.py::build_scenario(naca_code, reynolds, angle_of_attack)` produces an in-memory scenario dict with the same column layout as the dataset `.npz` files. Under the hood:

1. **C-mesh construction** — `src/utils/mesh.py` generates a structured C-grid of node coordinates around the airfoil. The wake region is rotated to follow the angle of attack, so the mesh is regenerated per scenario rather than reused.
2. **Signed distance field** — `src/utils/sdf.py` computes the analytical signed distance from every node to the NACA-4 surface. A warm-start from the nearest densified-polygon vertex is refined with Newton iteration on the orthogonality condition `(c − r(s)) · r'(s) = 0`. The dispatcher auto-selects a GPU implementation when CUDA is available.
3. **Freestream broadcast** — the freestream magnitude is computed from `U_mag = Re · ν / chord` with `ν = 1e-5` and `chord = 1.0`, and the (`u`, `v`) components are set from the angle of attack.


## Repository Structure

```text
turbulent-flows-project/
├── pyproject.toml                       project metadata and dependency declarations
├── uv.lock                              locked dependency resolution
├── README.md
├── inference_ground_truth.ipynb         notebook: inference on a dataset .npz
├── inference_custom_scenario.ipynb      notebook: inference on a user-defined NACA / Re / AoA
├── git-conventions.md                   guidelines how to work with git
├── checkpoints/                         trained models
│   └── gno_w32_d6_k16_lr0.001_NACA_4_Digit_for_ML/best.pt
├── data/                                scenarios (gitignored)
│   └── ...
├── output/                              notebook outputs (gitignored)
├── scripts/
│   └── download_dataset.py              Hugging Face dataset snapshot
└── src/
    ├── surrogate/gno/                   GNO model + inference path
    │   ├── model.py                     KernelNN graph network
    │   ├── layers.py                    NNConvLayer
    │   ├── utils.py                     DenseNet, UnitGaussianNormalizer
    │   ├── schema.py                    column orders and .npz helpers
    │   ├── graph.py                     k-NN edge_index, device selection
    │   └── infer.py                     public infer() entry point
    └── utils/
        ├── naca_geometry.py             analytical NACA-4 surface (open trailing edge)
        ├── mesh.py                      structured C-mesh node generator (numpy + scipy)
        ├── sdf.py                       Newton-refined analytical signed distance
        ├── scenario.py                  build_scenario(naca, Re, AoA) → in-memory scenario dict
        └── inference/                   plotting + .npz prediction export
            ├── fields.py
            ├── error_maps.py
            └── io.py
```

## GNO Architecture

A **Graph Neural Operator (GNO)** [1] is a neural-network architecture that learns a
mapping between function spaces (e.g. PDE solutions on arbitrary meshes) by
approximating an integral operator as a graph kernel. Each layer aggregates
neighbour features through edge-conditioned weight matrices produced by a small
MLP from edge attributes, making the network resolution- and discretization-agnostic.

This repository ships a `KernelNN` (`src/surrogate/gno/model.py`) with 7 input
channels (`x, y, sdf, u_init, v_init, angle_of_attack, reynolds`) and 6 output
channels (`u, v, p, k, omega, nu_t`). The network consists of an input linear
layer, 6 `NNConvLayer` blocks (hidden width 32, edge-kernel MLP width 256) with
gradient checkpointing, and an output linear layer. The connectivity graph is a
bidirectional k-nearest-neighbour graph (k = 16) built per-scenario from the
mesh cell-centre coordinates. The resulting GNO has approx. 2M parameters.

### Reference

[1] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart,
and A. Anandkumar, "Neural Operator: Graph Kernel Network for Partial
Differential Equations," arXiv:2003.03485, 2020. [Online]. Available:
https://arxiv.org/abs/2003.03485
