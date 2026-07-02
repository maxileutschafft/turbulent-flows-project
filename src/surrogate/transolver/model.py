"""
Self-contained Transolver for unstructured airfoil meshes.

Maps per-node features [N, 7] -> per-node fields [N, 6].

Duplicated and simplified from:
  physicsnemo/models/transolver/transolver.py
  physicsnemo/nn/module/physics_attention.py  (PhysicsAttentionBase + IrregularMesh variant)
  physicsnemo/nn/module/mlp_layers.py         (Mlp)

Simplifications applied:
  - use_te=False always (plain nn.Linear / nn.LayerNorm / torch SDPA)
  - plus=False (no Transolver++ / Gumbel softmax)
  - time_input=False, unified_pos=False, structured_shape=None (unstructured only)
  - No jaxtyping, no physicsnemo Module/ModelMetaData inheritance
  - No embedding argument: all 7 input features are fed directly as the
    functional input (the preprocess MLP projects them to n_hidden).

MIT License (upstream: THUML @ Tsinghua University)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Reuse GNO's input-normalization constants as the single source of truth so
# both surrogates see identically scaled inputs.
from surrogate.gno.model import INPUT_MEAN, INPUT_STD, SDF_COL, SDF_LOG_EPS

# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_ACT_MAP: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "softplus": nn.Softplus,
}


def _get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in _ACT_MAP:
        raise ValueError(f"Unknown activation '{name}'. Options: {list(_ACT_MAP)}")
    return _ACT_MAP[key]()


# ---------------------------------------------------------------------------
# Mlp
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """Two-layer MLP: in_features -> hidden_features -> out_features + activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_features or in_features
        out = out_features or in_features
        layers: list[nn.Module] = [
            nn.Linear(in_features, hidden),
            _get_activation(act),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# PhysicsAttentionIrregularMesh (unstructured variant only)
# ---------------------------------------------------------------------------

class PhysicsAttentionIrregularMesh(nn.Module):
    """
    Physics attention for irregular/unstructured mesh data.

    Projects each token onto learned physics-informed slices via a
    temperature-scaled softmax, applies scaled-dot-product attention
    among the S slice tokens, then scatters back to per-token space.

    Input / output: [B, N, C] -> [B, N, C]

    Parameters
    ----------
    dim : int
        Token feature dimension (must equal n_head * dim_head).
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per attention head (= dim // heads).
    dropout : float
        Dropout on the output projection.
    slice_num : int
        Number of physics slices S.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads

        # Learnable temperature for slice softmax, shape [1, 1, H, 1]
        self.temperature = nn.Parameter(torch.ones(1, 1, heads, 1) * 0.5)

        # Two linear projections: one for slice keys (x_mid), one for values (fx_mid)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)

        # Projects each dim_head vector onto S slice logits
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # QKV projection for attention among slice tokens
        self.qkv_project = nn.Linear(dim_head, 3 * dim_head, bias=False)

        # Output projection: [B, N, inner_dim] -> [B, N, dim]
        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [B, N, C]
        mask : torch.Tensor | None, shape [B, N]
            1 for real nodes, 0 for padded nodes. When provided, padded nodes
            are excluded from the slice aggregation (both the per-slice token
            sum and its normalizer), so real-node outputs are bit-for-bit
            identical to an unpadded single-sample forward. `None` (default)
            disables masking entirely — the original unbatched behavior.

        Returns
        -------
        torch.Tensor, shape [B, N, C]
        """
        H, D = self.heads, self.dim_head

        # --- project to multi-head layout ---
        # [B, N, inner_dim] -> [B, N, H, D]
        x_mid = rearrange(self.in_project_x(x), "B N (H D) -> B N H D", H=H, D=D)
        fx_mid = rearrange(self.in_project_fx(x), "B N (H D) -> B N H D", H=H, D=D)

        # --- compute slice weights ---
        # [B, N, H, D] -> [B, N, H, S]
        slice_logits = self.in_project_slice(x_mid)

        temp = torch.clamp(self.temperature, min=0.5, max=5.0).to(slice_logits.dtype)
        slice_weights = F.softmax(slice_logits / temp, dim=-1)  # [B, N, H, S]

        # Zero out padded nodes so they contribute nothing to the per-slice
        # token sum or its normalizer below (and so their own output is 0).
        # Real nodes keep their exact weights -> identical to the unpadded path.
        if mask is not None:
            slice_weights = slice_weights * mask[:, :, None, None].to(slice_weights.dtype)

        # --- aggregate per-slice tokens ---
        # Normalize over tokens to avoid overflow
        slice_norm = slice_weights.sum(1) + 1e-2  # [B, H, S]
        normed_weights = slice_weights / slice_norm[:, None, :, :]  # [B, N, H, S]

        # (B, H, S, N) @ (B, H, N, D) -> (B, H, S, D)
        slice_token = torch.matmul(
            normed_weights.permute(0, 2, 3, 1),   # [B, H, S, N]
            fx_mid.permute(0, 2, 1, 3),            # [B, H, N, D]
        )  # [B, H, S, D]

        # --- attention among slice tokens ---
        # qkv_project acts on the D axis; slice_token is [B, H, S, D]
        qkv = self.qkv_project(slice_token)                             # [B, H, S, 3D]
        qkv = rearrange(qkv, "B H S (T D) -> B H S T D", T=3, D=D)
        q, k, v = qkv.unbind(3)                                         # each [B, H, S, D]

        out_slice = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # [B, H, S, D]

        # --- scatter back to per-token space ---
        # (B, N, H, S) @ (B, H, S, D) -> (B, N, H, D)
        out_x = torch.einsum("bnhs,bhsd->bnhd", slice_weights, out_slice)

        # [B, N, H, D] -> [B, N, H*D]
        out_x = rearrange(out_x, "B N H D -> B N (H D)")
        out_x = self.out_linear(out_x)
        return self.out_dropout(out_x)


# ---------------------------------------------------------------------------
# TransolverBlock
# ---------------------------------------------------------------------------

class TransolverBlock(nn.Module):
    """
    Transolver transformer block.

    Structure:
      x = x + PhysicsAttn(LayerNorm(x))
      x = x + FFN(LayerNorm(x))
      [if last_layer]: out = LinearHead(LayerNorm(x))

    Input: [B, N, hidden_dim]
    Output: [B, N, hidden_dim] or [B, N, out_dim] for last layer.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
    ) -> None:
        super().__init__()
        self.last_layer = last_layer

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttentionIrregularMesh(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )

        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * mlp_ratio,
            out_features=hidden_dim,
            act=act,
            dropout=dropout,
        )

        if self.last_layer:
            self.ln_out = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        fx : torch.Tensor, shape [B, N, C]
        mask : torch.Tensor | None, shape [B, N]
            Forwarded to the physics-attention slice aggregation (see
            `PhysicsAttentionIrregularMesh.forward`). `None` disables masking.

        Returns
        -------
        torch.Tensor, shape [B, N, C] or [B, N, out_dim] for last_layer=True
        """
        fx = self.attn(self.ln_1(fx), mask=mask) + fx
        fx = self.ffn(self.ln_ffn(fx)) + fx
        if self.last_layer:
            return self.head(self.ln_out(fx))
        return fx


# ---------------------------------------------------------------------------
# Transolver
# ---------------------------------------------------------------------------

class Transolver(nn.Module):
    """
    Transolver neural operator for unstructured airfoil meshes.

    Maps per-node input features directly to per-node output fields.
    No separate embedding input is required — all input columns (including
    x, y coordinates) are fed as the functional input and projected to
    n_hidden via an internal preprocess MLP.

    Constructor
    -----------
    functional_dim : int, default=7
        Number of input feature columns (matches GNO_INPUT_COLS length).
    out_dim : int, default=6
        Number of output fields (matches GNO_TARGET_COLS length).
    n_hidden : int, default=208
        Hidden dimension (must be divisible by n_head).
    n_layers : int, default=4
        Number of TransolverBlock layers.
    n_head : int, default=8
        Number of attention heads.
    mlp_ratio : int, default=4
        FFN hidden size multiplier.
    slice_num : int, default=32
        Number of physics slices.
    dropout : float, default=0.0
        Dropout rate.
    act : str, default="gelu"
        Activation function name.

    forward(x) contract
    -------------------
    x : torch.Tensor
        Shape [N, 7] — single sample, no batch dim (primary training path).
        Also accepts [B, N, 7] for batched inference.
    Returns torch.Tensor
        Shape [N, 6] when input is [N, 7], or [B, N, 6] when input is [B, N, 7].
    """

    def __init__(
        self,
        functional_dim: int = 7,
        out_dim: int = 6,
        n_hidden: int = 208,
        n_layers: int = 4,
        n_head: int = 8,
        mlp_ratio: int = 4,
        slice_num: int = 32,
        dropout: float = 0.0,
        act: str = "gelu",
    ) -> None:
        super().__init__()

        if n_hidden % n_head != 0:
            raise ValueError(
                f"n_hidden must be divisible by n_head, got {n_hidden} % {n_head} = {n_hidden % n_head}"
            )

        self.n_hidden = n_hidden

        # Input normalization (buffers, not params — kept out of the param count).
        # Reuses GNO's constants so both surrogates see identically scaled inputs.
        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN, dtype=torch.float32))
        self.register_buffer("input_std", torch.tensor(INPUT_STD, dtype=torch.float32))

        # Preprocess MLP: functional_dim -> n_hidden*2 -> n_hidden
        # (mirrors blueprint: in_features=functional_dim, hidden=n_hidden*2, out=n_hidden)
        self.preprocess = Mlp(
            in_features=functional_dim,
            hidden_features=n_hidden * 2,
            out_features=n_hidden,
            act=act,
        )

        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                last_layer=(i == n_layers - 1),
                out_dim=out_dim,
                slice_num=slice_num,
            )
            for i in range(n_layers)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the hardcoded input transform: z-score for all cols except
        `sdf`, which uses log10(sdf + eps). Input `x` is raw physical units in
        GNO_INPUT_COLS order.

        Batch-shape-agnostic: works for both `[N, C]` and `[B, N, C]` inputs.
        `input_mean`/`input_std` have shape `[C]` and broadcast against the
        trailing channel axis; the SDF column is indexed with `..., SDF_COL`.
        """
        z = (x - self.input_mean) / self.input_std
        log_sdf = torch.log10(x[..., SDF_COL:SDF_COL + 1] + SDF_LOG_EPS)
        return torch.cat([z[..., :SDF_COL], log_sdf, z[..., SDF_COL + 1:]], dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [N, 7] for a single sample (training path) or [B, N, 7] for a batch.
            Raw physical units in GNO_INPUT_COLS order; normalized internally.
        mask : torch.Tensor | None, shape [B, N]
            Per-node validity mask for padded batches (1 real, 0 padded). When
            meshes of unequal node count are padded to a common N and stacked,
            this excludes the padded nodes from the slice aggregation so each
            real node's output matches an unpadded single-sample forward.
            `None` (default) is the original behavior — no masking.

        Returns
        -------
        torch.Tensor
            [N, 6] or [B, N, 6] mirroring the input leading shape.
        """
        # Normalize raw inputs first (works for both [N, C] and [B, N, C]).
        x = self.normalize_inputs(x)

        unbatch = x.ndim == 2
        if unbatch:
            x = x.unsqueeze(0)  # [1, N, 7]

        # [B, N, C_in] -> [B, N, n_hidden]
        fx = self.preprocess(x)

        for block in self.blocks:
            fx = block(fx, mask=mask)
        # fx: [B, N, out_dim]

        if unbatch:
            fx = fx.squeeze(0)  # [N, out_dim]
        return fx


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = Transolver(
        functional_dim=7,
        out_dim=6,
        n_hidden=208,
        n_layers=4,
        n_head=8,
        mlp_ratio=4,
        slice_num=32,
        dropout=0.0,
        act="gelu",
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"PARAM_COUNT={param_count}")
    assert 1_800_000 <= param_count <= 2_200_000, (
        f"Parameter count {param_count:,} is outside [1.8M, 2.2M]"
    )

    x = torch.randn(2048, 7)
    # SDF column (index SDF_COL) is non-negative in this pipeline: the model
    # applies log10(sdf + eps) internally (matching GNO), so feed |sdf| here.
    x[:, SDF_COL] = x[:, SDF_COL].abs()
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2048, 6), f"Unexpected output shape: {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    print("FORWARD_OK")
