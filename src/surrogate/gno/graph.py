"""k-NN graph construction + device selection.

Only the helpers actually called from `infer.py` are kept; this module exists
so the inference pipeline does NOT need to import the training script.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_edge_index(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Build bidirectional k-NN edge_index from 2D positions using CPU cKDTree."""
    pos_np = pos.detach().cpu().numpy()
    N = pos_np.shape[0]
    tree = cKDTree(pos_np)
    _, idx = tree.query(pos_np, k=k + 1)  # k+1: first column is self
    src_np = np.repeat(np.arange(N), k)
    dst_np = idx[:, 1:].flatten()
    src = torch.from_numpy(np.concatenate([src_np, dst_np]).astype(np.int64, copy=False))
    dst = torch.from_numpy(np.concatenate([dst_np, src_np]).astype(np.int64, copy=False))
    return torch.stack([src, dst], dim=0)


def build_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Build 6D edge attributes [src_x, src_y, 0, dst_x, dst_y, 0]."""
    src, dst = edge_index[0], edge_index[1]
    n_edges = edge_index.shape[1]
    zeros = torch.zeros(n_edges, 1, dtype=pos.dtype, device=pos.device)
    src_xyz = torch.cat([pos[src], zeros], dim=1)  # [E, 3]
    dst_xyz = torch.cat([pos[dst], zeros], dim=1)  # [E, 3]
    return torch.cat([src_xyz, dst_xyz], dim=1)  # [E, 6]


def build_graph(pos: torch.Tensor, k: int):
    """Build bidirectional k-NN edge_index and 6D edge_attr from 2D positions."""
    edge_index = build_edge_index(pos, k)
    edge_attr = build_edge_attr(pos, edge_index)
    return edge_index, edge_attr
