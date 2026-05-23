import torch
import torch.nn as nn


class DenseNet(nn.Module):
    """Simple MLP used as the kernel/edge network inside NNConvLayer."""

    def __init__(self, layer_sizes, nonlinearity=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nonlinearity())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class UnitGaussianNormalizer:
    """Zero-mean, unit-variance normalizer computed per feature dimension."""

    def __init__(self, x, eps=1e-5):
        # x: [N, d] or [N]
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
