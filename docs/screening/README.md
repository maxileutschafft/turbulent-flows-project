# Screening Parameter Ranges

This document explains the default definitive-screening-style parameter ranges used by the repository for first-pass surrogate hyperparameter tuning.

The goal is not to exhaustively optimize the final model in one step. The goal is to identify which training choices matter most, using a compact and structured experiment plan, before moving to a finer optimizer such as Bayesian optimization or a smaller local search.

## Why these ranges are reasonable

The ranges are intentionally centered around the currently shipped surrogate family:

- The model in [src/surrogate/gno/model.py](../../src/surrogate/gno/model.py) was trained with a hidden width of 32, a depth of 6, and a kernel MLP width of 256.
- That checkpoint is already known to work on the NACA 4-digit dataset, so the screening ranges should explore values close enough to preserve stability while still being wide enough to expose trends.
- The ranges are symmetric around the current working point whenever that makes sense, with roughly a 0.5x to 2x span for the most important capacity parameters.

In a screening stage, the point is not to search the whole space. The point is to cover the local neighborhood broadly enough that the dominant effects show up clearly.

## Parameter Ranges

| Factor | Low | Center | High | Reasoning |
|---|---:|---:|---:|---|
| `width_node` | 16 | 32 | 64 | Tests whether the network is under- or over-parameterized without leaving the regime where the current architecture is known to train. A factor of 2 in either direction is a standard first screening interval for model width. |
| `depth` | 4 | 6 | 8 | Probes whether the current 6-layer message-passing stack is too shallow or too deep for the available data. Going much deeper can increase optimization difficulty and over-smoothing, so this range stays conservative. |
| `ker_width` | 128 | 256 | 512 | Controls the edge-network capacity. This is often important in graph operators because the kernel MLP must express geometry-dependent interactions. A 4x span is broad enough to detect saturation, but still practical. |
| `k_neighbors` | 8 | 16 | 24 | Balances locality against graph connectivity. Too small a `k` can fragment flow coupling; too large a `k` can blur local structure and increase compute. The current value of 16 is a sensible midpoint. |
| `learning_rate` | 3e-4 | 1e-3 | 3e-3 | Learning rate usually has the strongest effect on convergence behavior. A log-spaced 10x band around the current value is standard because optimization dynamics change multiplicatively, not linearly. |
| `weight_decay` | 1e-6 | 1e-5 | 1e-4 | Regularization needs to be screened on a log scale because the meaningful differences are order-of-magnitude changes. This range keeps the model in the low-to-moderate regularization regime. |

## Why these are good screening factors

These variables are the most defensible first-stage knobs because they directly affect surrogate generalization and training stability:

- `width_node` changes representational capacity.
- `depth` changes receptive field and message-passing propagation.
- `ker_width` changes the expressive power of the edge-conditioned kernel.
- `k_neighbors` changes the graph topology and the balance between local and global coupling.
- `learning_rate` changes how well the optimizer can find a useful basin.
- `weight_decay` changes whether the network generalizes or memorizes.

These are better first-stage screening variables than more exotic changes such as activation replacements or architecture rewrites, because they are easy to compare, easy to interpret, and directly tied to the current implementation.

## Why the objective set is reasonable

The screening objective should reward better field predictions first and engineering usefulness second.

### Primary objectives

- `val_nrmse_u`
- `val_nrmse_v`
- `val_nrmse_p`
- `val_nrmse_k`
- `val_nrmse_omega`
- `val_nrmse_nut`

These are the most direct measures of surrogate fidelity on the quantities the model is expected to predict. Normalized RMSE is appropriate because the outputs live on different scales and some of them span multiple orders of magnitude.

### Secondary objectives

- `val_cl_rel_err`
- `val_cd_rel_err`
- `val_near_wall_rmse`

These are secondary because they are more engineering-oriented or localized, but they matter for model usefulness:

- Lift and drag are the quantities most likely to matter in downstream design and validation.
- Near-wall error is important because small local mistakes near the airfoil can strongly affect pressure gradients, drag, and separation predictions.

In practice, a model that slightly improves raw field RMSE but breaks drag prediction is not a better model for this application.

## Why this is a DSD-style design and not a full optimization plan

A definitive screening design is intended to answer "which factors matter?" with a limited number of runs. It is not intended to fully optimize all hyperparameters.

That is why the repository uses the screening design as the first stage of optimization:

1. Screen the main knobs using the DSD-style plan.
2. Identify the dominant factors.
3. Narrow the range for the important factors.
4. Apply Bayesian optimization or a smaller local search in the reduced space.

This is better than jumping directly to a large grid search because it avoids spending compute on unimportant factors.

## Suggested interpretation of results

After running the screening plan:

- If `width_node` dominates, the model is capacity-limited and you should explore wider networks.
- If `depth` dominates, the operator is likely benefiting from more message-passing steps or suffering from oversmoothing.
- If `k_neighbors` dominates, the graph topology is too sparse or too dense for the current mesh and flow regime.
- If `learning_rate` dominates, optimization is the main bottleneck and you should tune the optimizer schedule before touching architecture.
- If `weight_decay` dominates, generalization is sensitive to regularization and you should refine the regularization regime before increasing model size.

## Practical recommendation for this project

For the current repository, the best first screening run is the default 6-factor design defined in [src/utils/screening.py](../../src/utils/screening.py).

That gives a compact experiment set that is large enough to expose the main effects but small enough to run on the existing dataset without ANSYS.

The screening output can be generated with:

```bash
uv run python scripts/generate_screening_design.py --output output/screening_design.csv --json output/screening_design.json
```

The resulting design should be used to train one model per row, then compare the validation metrics listed above.