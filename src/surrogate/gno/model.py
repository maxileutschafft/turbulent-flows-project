import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import NNConvLayer
from .utils import DenseNet

# ---------------------------------------------------------------------------
# Hardcoded input normalization constants (part of the network — do NOT edit
# without retraining). Computed from the NACA_4_Digit_for_ML training split
# (581 samples, ~22.5M nodes).
#
# Column order matches schema.GNO_INPUT_COLS:
#   [x, y, sdf, u_init, v_init, angle_of_attack, reynolds]
#
# Transforms:
#   - x, y                  : z-score (per-node stats, pooled over all nodes)
#   - sdf                   : log10(sdf + SDF_LOG_EPS)  — no z-score
#   - u_init, v_init,
#     angle_of_attack,
#     reynolds              : z-score (per-sample stats, one obs per scenario)
# ---------------------------------------------------------------------------
INPUT_MEAN = (0.803942, 0.006600, 0.0,       2.466650, -0.005332, -0.141480, 247001.0)
INPUT_STD  = (0.739590, 0.335935, 1.0,       1.057862,  0.138850,  3.032353, 105913.0)
SDF_COL = 2          # index of `sdf` in GNO_INPUT_COLS — replaced by log10
SDF_LOG_EPS = 1.0e-6


class KernelNN(nn.Module):
    """Graph Neural Operator: learns a parametric PDE solution operator.

    Architecture:
        normalize_input  →  fc1  →  depth × NNConvLayer  →  fc2

    Input normalization is baked into the network using fixed constants
    (`INPUT_MEAN`, `INPUT_STD`, `SDF_LOG_EPS`) so every call — training,
    validation, inference — applies the exact same transform.

    Each NNConvLayer uses a DenseNet to map edge attributes to per-edge
    weight matrices, enabling resolution-agnostic operator learning.
    """

    def __init__(
        self,
        width_node: int = 32,
        ker_width: int = 256,
        depth: int = 6,
        ker_in: int = 6,
        in_width: int = 7,
        out_width: int = 6,
    ):
        super().__init__()
        self.depth = depth

        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN, dtype=torch.float32))
        self.register_buffer("input_std", torch.tensor(INPUT_STD, dtype=torch.float32))

        self.fc1 = nn.Linear(in_width, width_node)

        self.convs = nn.ModuleList([
            NNConvLayer(
                in_channels=width_node,
                out_channels=width_node,
                kernel_nn=DenseNet(
                    [ker_in, ker_width, ker_width, width_node * width_node],
                    nonlinearity=nn.ReLU,
                ),
            )
            for _ in range(depth)
        ])

        self.fc2 = nn.Linear(width_node, out_width)

    def normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the hardcoded input transform: z-score for all cols except
        `sdf`, which uses log10(sdf + eps). Input `x` is raw physical units in
        GNO_INPUT_COLS order."""
        z = (x - self.input_mean) / self.input_std
        log_sdf = torch.log10(x[:, SDF_COL:SDF_COL + 1] + SDF_LOG_EPS)
        return torch.cat([z[:, :SDF_COL], log_sdf, z[:, SDF_COL + 1:]], dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = self.normalize_inputs(x)
        x = self.fc1(x)
        for k, conv in enumerate(self.convs):
            x = checkpoint(conv, x, edge_index, edge_attr, use_reentrant=False)
            if k < self.depth - 1:
                x = F.relu(x)
        return self.fc2(x)
