import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing


class NNConvLayer(MessagePassing):
    """Edge-conditioned graph convolution (continuous-kernel convolution).

    x'_i = Θ·x_i + mean_{j∈N(i)} [ h_Θ(e_ij) · x_j ]

    where h_Θ maps edge attributes to an (in_channels × out_channels) weight matrix.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_nn: nn.Module,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_nn = kernel_nn

        self.root = Parameter(torch.empty(in_channels, out_channels))
        self.bias = Parameter(torch.empty(out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.root)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return aggr_out + torch.mm(x, self.root) + self.bias

    def message(self, x_j, edge_attr):
        # kernel_nn maps edge_attr → weight matrix of shape [E, in_channels * out_channels]
        weight = self.kernel_nn(edge_attr).view(-1, self.in_channels, self.out_channels)
        # x_j: [E, in_channels] → [E, 1, in_channels]
        return torch.bmm(x_j.unsqueeze(1), weight).squeeze(1)
