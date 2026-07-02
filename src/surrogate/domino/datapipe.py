"""
datapipe.py - Auxiliary data preparation for DoMINO 2D point-cloud model.

Builds all precomputed tensors needed by the model:
  - k-NN graph on node coordinates
  - rasterized SDF grid for the CNN branch
  - normalized coordinates for grid_sample
"""

import torch


def build_aux(x: torch.Tensor, grid_res: int = 64, k_neighbors: int = 16) -> dict:
    """
    Given x [N, 7] (cols: x, y, sdf, u_init, v_init, aoa, reynolds),
    returns a dict of precomputed tensors needed by DoMINO forward.

    Args:
        x:          [N, 7] input tensor
        grid_res:   spatial resolution of the background SDF grid (H = W = grid_res)
        k_neighbors: number of nearest neighbors per node

    Returns dict with:
        "coords":           [N, 2]       node xy coordinates
        "sdf":              [N, 1]       SDF per node
        "global_params":    [4]          [u_init, v_init, aoa, re] (first node's scalars)
        "knn_idx":          [N, k]       long, k-NN indices
        "knn_rel_pos":      [N, k, 2]   relative positions to neighbors
        "grid_sdf":         [1, 1, H, W] rasterized SDF grid
        "grid_coords_norm": [N, 2]       node coords normalized to [-1, 1]
        "bbox":             [4]          [x_min, y_min, x_max, y_max]
    """
    N = x.shape[0]
    if k_neighbors >= N:
        raise ValueError(
            f"k_neighbors={k_neighbors} must be less than the number of nodes N={N}."
        )
    device = x.device

    coords = x[:, :2]           # [N, 2]
    sdf = x[:, 2:3]             # [N, 1]
    global_params = x[0, 3:7]  # [4]  u_init, v_init, aoa, reynolds

    # ------------------------------------------------------------------ #
    # Bounding box
    # ------------------------------------------------------------------ #
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()

    eps = 1e-6
    x_range = (x_max - x_min).clamp(min=eps)
    y_range = (y_max - y_min).clamp(min=eps)

    bbox = torch.stack([x_min, y_min, x_max, y_max])  # [4]

    # ------------------------------------------------------------------ #
    # Grid-normalised coords: [-1, 1] in both dimensions
    # grid_sample convention: last dim is (x, y) = (col, row) in [-1,1]
    # ------------------------------------------------------------------ #
    norm_x = (coords[:, 0] - x_min) / x_range * 2.0 - 1.0  # [N]
    norm_y = (coords[:, 1] - y_min) / y_range * 2.0 - 1.0  # [N]
    grid_coords_norm = torch.stack([norm_x, norm_y], dim=-1)  # [N, 2]

    # ------------------------------------------------------------------ #
    # Rasterize SDF onto H x W grid
    # ------------------------------------------------------------------ #
    H = W = grid_res
    # Map node coords to integer grid indices
    col_idx = ((coords[:, 0] - x_min) / x_range * (W - 1)).long().clamp(0, W - 1)
    row_idx = ((coords[:, 1] - y_min) / y_range * (H - 1)).long().clamp(0, H - 1)

    grid_sdf = torch.zeros(H, W, device=device)
    flat_idx = row_idx * W + col_idx  # [N]
    sdf_flat = sdf.squeeze(-1)        # [N]

    # scatter_reduce: average SDF values that map to the same cell
    # Fallback to manual scatter_add + count if needed
    count = torch.zeros(H * W, device=device)
    accum = torch.zeros(H * W, device=device)
    accum.scatter_add_(0, flat_idx, sdf_flat)
    count.scatter_add_(0, flat_idx, torch.ones(N, device=device))
    filled = count > 0
    accum[filled] = accum[filled] / count[filled]
    grid_sdf = accum.view(H, W).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # ------------------------------------------------------------------ #
    # k-NN graph
    # ------------------------------------------------------------------ #
    CHUNK = 2048
    if N <= CHUNK:
        dist_mat = torch.cdist(coords, coords)  # [N, N]
        # Exclude self by setting diagonal to inf
        dist_mat.fill_diagonal_(float("inf"))
        knn_idx = dist_mat.topk(k_neighbors, largest=False, dim=-1).indices  # [N, k]
    else:
        # Chunked cdist to avoid OOM for large N
        knn_idx = torch.empty(N, k_neighbors, dtype=torch.long, device=device)
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            chunk_dist = torch.cdist(coords[start:end], coords)  # [chunk, N]
            # Mask self-distances (vectorised diagonal: row local_i ↔ col start+local_i)
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            chunk_dist[rows, cols] = float("inf")
            knn_idx[start:end] = chunk_dist.topk(k_neighbors, largest=False, dim=-1).indices

    # Relative positions to neighbors
    neighbor_coords = coords[knn_idx]                     # [N, k, 2]
    knn_rel_pos = neighbor_coords - coords.unsqueeze(1)   # [N, k, 2]

    return {
        "coords": coords,
        "sdf": sdf,
        "global_params": global_params,
        "knn_idx": knn_idx,
        "knn_rel_pos": knn_rel_pos,
        "grid_sdf": grid_sdf,
        "grid_coords_norm": grid_coords_norm,
        "bbox": bbox,
    }
