"""
model.py - 2D-adapted DoMINO model (pure PyTorch) with dual-head (volume + surface).

DoMINO combines:
  1. A CNN encoder-decoder (GeoProcessor) that processes a rasterized SDF grid
     (shared between volume and surface heads)
  2. A k-NN local aggregation MLP (LocalPointConv) — one per head
  3. A Fourier-feature MLP (FourierMLP) for node basis functions — one per head
  4. An aggregation MLP that fuses geo, local, basis, and global features — one per head

Volume head:  x [N, 7]  → [N, 6]   (u, v, p, k, omega, nut)
Surface head: surface aux → [M, 1]  (wall pressure p)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from surrogate.domino.datapipe import build_aux


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


def _conv_same(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Conv2d:
    """Conv2d with 'same' padding (works for odd kernel sizes)."""
    pad = (kernel_size - 1) // 2
    return nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad)


# --------------------------------------------------------------------------- #
# Sub-modules
# --------------------------------------------------------------------------- #

class GeoProcessor(nn.Module):
    """
    2D CNN encoder-decoder that processes the rasterised SDF grid.

    Input:  [B, 1, H, W]
    Output: [B, geo_feat_dim, H, W]

    Encoder: 2x MaxPool2d(2) → spatial size is H/4 x W/4 at bottleneck
    Decoder: 2x ConvTranspose2d(stride=2) with skip connections

    Shared between the volume and surface heads.
    """

    def __init__(self, base_filters: int = 32, geo_feat_dim: int = 64, activation: str = "gelu"):
        super().__init__()
        bf = base_filters
        act = activation

        # Encoder stage 1: H x W → H/2 x W/2
        self.enc1 = nn.Sequential(
            _conv_same(1, bf),
            _get_activation(act),
        )
        self.pool1 = nn.MaxPool2d(2)

        # Encoder stage 2: H/2 x W/2 → H/4 x W/4
        self.enc2 = nn.Sequential(
            _conv_same(bf, bf * 2),
            _get_activation(act),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Encoder stage 3 (no pool): H/4 x W/4
        self.enc3 = nn.Sequential(
            _conv_same(bf * 2, bf * 4),
            _get_activation(act),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            _conv_same(bf * 4, bf * 4),
            _get_activation(act),
        )

        # Decoder stage 1: H/4 x W/4 → H/2 x W/2
        # After upsampling bottleneck (bf*4), concat with enc2 skip (bf*2) → bf*4+bf*2=bf*6
        self.up1 = nn.ConvTranspose2d(bf * 4, bf * 2, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            _conv_same(bf * 4, bf * 2),   # input: cat(up1_out=bf*2, skip=bf*2)
            _get_activation(act),
        )

        # Decoder stage 2: H/2 x W/2 → H x W
        # After upsampling (bf*2), concat with enc1 skip (bf) → bf*2+bf=bf*3
        self.up2 = nn.ConvTranspose2d(bf * 2, bf, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            _conv_same(bf * 2, bf),        # input: cat(up2_out=bf, skip=bf)
            _get_activation(act),
        )

        # Final projection → geo_feat_dim channels
        self.proj = nn.Conv2d(bf, geo_feat_dim, kernel_size=1)

    def forward(self, grid_sdf: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(grid_sdf)    # [B, bf, H, W]
        s2 = self.enc2(self.pool1(s1))  # [B, 2*bf, H/2, W/2]
        s3 = self.enc3(self.pool2(s2))  # [B, 4*bf, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(s3)     # [B, 4*bf, H/4, W/4]

        # Decoder stage 1
        u1 = self.up1(b)            # [B, 2*bf, H/2, W/2]
        # Handle potential size mismatch from odd dimensions
        if u1.shape != s2.shape:
            u1 = F.interpolate(u1, size=s2.shape[2:], mode="bilinear", align_corners=True)
        d1 = self.dec1(torch.cat([u1, s2], dim=1))  # [B, 2*bf, H/2, W/2]

        # Decoder stage 2
        u2 = self.up2(d1)           # [B, bf, H, W]
        if u2.shape != s1.shape:
            u2 = F.interpolate(u2, size=s1.shape[2:], mode="bilinear", align_corners=True)
        d2 = self.dec2(torch.cat([u2, s1], dim=1))  # [B, bf, H, W]

        return self.proj(d2)        # [B, geo_feat_dim, H, W]


class LocalPointConv(nn.Module):
    """
    k-NN aggregation MLP.

    Volume head: input per node: [N, k*3]  (k neighbors × [rel_x, rel_y, neighbor_sdf])
    Surface head: input per point: [M, k_surf*3]  (k_surf neighbors × [rel_x, rel_y, neighbor_ds])

    Output: [N or M, local_feat_dim]
    """

    def __init__(
        self,
        k_neighbors: int = 16,
        base_layer: int = 512,
        local_feat_dim: int = 128,
        activation: str = "gelu",
    ):
        super().__init__()
        in_dim = k_neighbors * 3
        hidden = base_layer // 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            _get_activation(activation),
            nn.Linear(hidden, local_feat_dim),
            _get_activation(activation),
        )

    def forward(self, local_in: torch.Tensor) -> torch.Tensor:
        return self.mlp(local_in)  # [N or M, local_feat_dim]


class FourierMLP(nn.Module):
    """
    Neural basis-function MLP with Fourier feature encoding.

    Volume head:  Input [N, 3]  (x, y, sdf)      → [N, basis_dim]
    Surface head: Input [M, 6]  (x, y, sdf≈0, n_x, n_y, log_ds_scaled)
                                                   → [M, basis_dim]

    Fourier encoding: for each of the raw_in input dims and num_modes frequencies,
    appends sin(z*f) and cos(z*f), giving raw_in + raw_in*num_modes*2 total features.

    The `raw_in` arg defaults to 3 (volume path) so existing callers are unchanged.
    """

    def __init__(
        self,
        num_modes: int = 4,
        base_layer: int = 512,
        basis_dim: int = 256,
        activation: str = "gelu",
        raw_in: int = 3,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.raw_in = raw_in
        # Frequencies: exp(linspace(0, pi, num_modes))
        freqs = torch.exp(torch.linspace(0, math.pi, num_modes))
        self.register_buffer("freqs", freqs)  # [num_modes]

        fourier_in = raw_in + raw_in * num_modes * 2  # raw + sin+cos per dim per freq

        self.mlp = nn.Sequential(
            nn.Linear(fourier_in, base_layer),
            _get_activation(activation),
            nn.Linear(base_layer, base_layer),
            _get_activation(activation),
            nn.Linear(base_layer, basis_dim),
            _get_activation(activation),
        )

    def forward(self, node_enc: torch.Tensor) -> torch.Tensor:
        # node_enc: [..., raw_in]  (batch-shape-agnostic)
        leading = node_enc.shape[:-1]           # e.g. () for [N, raw_in] or (B, N) for [B, N, raw_in]
        z = node_enc.unsqueeze(-1) * self.freqs  # [..., raw_in, num_modes]
        fourier = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)  # [..., raw_in, 2*num_modes]
        fourier = fourier.reshape(*leading, -1)                     # [..., raw_in*2*num_modes]
        x = torch.cat([node_enc, fourier], dim=-1)                  # [..., fourier_in]
        return self.mlp(x)                                          # [..., basis_dim]


class AggMLP(nn.Module):
    """
    Aggregation MLP that fuses geo, local, basis, and global features.

    Input:  [N, agg_input_dim]
    Output: [N, out_dim]
    """

    def __init__(
        self,
        agg_input_dim: int,
        base_layer: int = 512,
        out_dim: int = 6,
        activation: str = "gelu",
    ):
        super().__init__()
        act = activation
        self.net = nn.Sequential(
            nn.Linear(agg_input_dim, base_layer),
            nn.LayerNorm(base_layer),
            _get_activation(act),
            nn.Linear(base_layer, base_layer),
            _get_activation(act),
            nn.Linear(base_layer, base_layer),
            _get_activation(act),
            nn.Linear(base_layer, base_layer),
            _get_activation(act),
            nn.Linear(base_layer, out_dim),
        )

    def forward(self, agg_in: torch.Tensor) -> torch.Tensor:
        return self.net(agg_in)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #

class DoMINO(nn.Module):
    """
    2D-adapted DoMINO surrogate model with optional dual-head (volume + surface).

    Volume head:  x [N, 7]  → [N, 6]   (u, v, p, k, omega, nut)
    Surface head: surface aux → [M, 1]  (wall pressure p, z-scored)

    The GeoProcessor CNN is shared between the two heads.

    Args:
        input_features:    Not used directly (kept for API compat)
        out_dim:           Number of volume output channels (default 6)
        base_layer:        Width of volume head MLP hidden layers
        base_layer_surf:   Width of surface head MLP hidden layers (default 416)
        base_filters:      Base channel count for CNN; geo_feat_dim = base_filters * 2
        num_modes:         Number of Fourier frequency modes
        k_neighbors:       k for volume k-NN graph
        k_surf:            k for surface contour k-NN graph (default 8)
        grid_res:          Spatial resolution of SDF grid (H = W = grid_res)
        global_features:   Number of global scalar features (default 4)
        activation:        Activation function name ("gelu", "relu", "silu")
    """

    def __init__(
        self,
        input_features: int = 3,
        out_dim: int = 6,
        base_layer: int = 512,
        base_layer_surf: int = 416,
        base_filters: int = 32,
        num_modes: int = 4,
        k_neighbors: int = 16,
        k_surf: int = 8,
        grid_res: int = 64,
        global_features: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.k_surf = k_surf
        self.grid_res = grid_res
        self.out_dim = out_dim

        # ---- Shared geometry encoder ---------------------------------------- #
        geo_feat_dim = base_filters * 2       # e.g. 36 for base_filters=18

        self.geo_processor = GeoProcessor(
            base_filters=base_filters,
            geo_feat_dim=geo_feat_dim,
            activation=activation,
        )

        # ---- Volume head ---------------------------------------------------- #
        local_feat_dim_vol = base_layer // 4          # e.g. 124 for 496
        basis_dim_vol = base_layer // 2               # e.g. 248
        agg_input_dim_vol = geo_feat_dim + local_feat_dim_vol + basis_dim_vol + global_features
        # = 36 + 124 + 248 + 4 = 412

        self.local_point_conv = LocalPointConv(
            k_neighbors=k_neighbors,
            base_layer=base_layer,
            local_feat_dim=local_feat_dim_vol,
            activation=activation,
        )

        self.fourier_mlp = FourierMLP(
            num_modes=num_modes,
            base_layer=base_layer,
            basis_dim=basis_dim_vol,
            activation=activation,
            raw_in=3,  # volume: (x, y, sdf)
        )

        self.agg_mlp = AggMLP(
            agg_input_dim=agg_input_dim_vol,
            base_layer=base_layer,
            out_dim=out_dim,
            activation=activation,
        )

        # ---- Surface head --------------------------------------------------- #
        local_feat_dim_surf = base_layer_surf // 4    # e.g. 104 for 416
        basis_dim_surf = base_layer_surf // 2         # e.g. 208
        # agg input: geo_feat + local_feat + basis + global_features
        # = 36 + 104 + 208 + 4 = 352
        agg_input_dim_surf = geo_feat_dim + local_feat_dim_surf + basis_dim_surf + global_features

        # Surface LocalPointConv: input [M, k_surf * 3]
        # where the 3 features per neighbour are (rel_x, rel_y, neighbor_ds)
        self.surf_local_point_conv = LocalPointConv(
            k_neighbors=k_surf,
            base_layer=base_layer_surf,
            local_feat_dim=local_feat_dim_surf,
            activation=activation,
        )

        # Surface FourierMLP: raw_in=6 → (x, y, sdf≈0, n_x, n_y, log_ds_scaled)
        self.surf_fourier_mlp = FourierMLP(
            num_modes=num_modes,
            base_layer=base_layer_surf,
            basis_dim=basis_dim_surf,
            activation=activation,
            raw_in=6,
        )

        self.surf_agg_mlp = AggMLP(
            agg_input_dim=agg_input_dim_surf,
            base_layer=base_layer_surf,
            out_dim=1,   # wall pressure p
            activation=activation,
        )

    # --------------------------------------------------------------------- #
    # Forward dispatch
    # --------------------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        aux: dict | None = None,
        *,
        predict_volume: bool = True,
        predict_surface: bool = False,
    ) -> torch.Tensor | dict:
        """
        Args:
            x:                [N, 7] single-sample or [B, N, 7] batched input.
            aux:              precomputed dict from build_aux() / _collate_batch().
                              If None, built from x (single-sample only).
                              When predict_surface=True, must also contain surface
                              aux keys from build_surface_aux().
            predict_volume:   Whether to run the volume head.
            predict_surface:  Whether to run the surface head.

        Returns:
            If only predict_volume (default): volume tensor [N,6] or [B,N,6]
            If only predict_surface:          surface tensor [M,1] or [B,M,1]
            If both:                          dict {"volume": ..., "surface": ...}
        """
        if aux is None:
            aux = build_aux(x, grid_res=self.grid_res, k_neighbors=self.k_neighbors)

        batched = x.ndim == 3
        if batched:
            return self._forward_batched(x, aux, predict_volume=predict_volume, predict_surface=predict_surface)
        else:
            return self._forward_single(x, aux, predict_volume=predict_volume, predict_surface=predict_surface)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _normalize_global_params_single(self, global_params: torch.Tensor, N: int) -> torch.Tensor:
        """Normalise global_params [4] → [N, 4] (single-sample)."""
        gp_raw = global_params  # [4]
        u_norm  = gp_raw[0:1] / 100.0
        v_norm  = gp_raw[1:2] / 100.0
        aoa_norm = gp_raw[2:3] / 45.0
        re_norm = torch.log10(gp_raw[3:4].abs().clamp(min=1.0)) / 7.0
        gp_normed = torch.cat([u_norm, v_norm, aoa_norm, re_norm], dim=0)  # [4]
        return gp_normed.unsqueeze(0).expand(N, -1)                         # [N, 4]

    def _normalize_global_params_batched(self, global_params: torch.Tensor, N: int) -> torch.Tensor:
        """Normalise global_params [B, 4] → [B, N, 4] (batched)."""
        gp_raw = global_params                                                  # [B, 4]
        u_norm  = gp_raw[:, 0:1] / 100.0
        v_norm  = gp_raw[:, 1:2] / 100.0
        aoa_norm = gp_raw[:, 2:3] / 45.0
        re_norm = torch.log10(gp_raw[:, 3:4].abs().clamp(min=1.0)) / 7.0
        gp_normed = torch.cat([u_norm, v_norm, aoa_norm, re_norm], dim=-1)     # [B, 4]
        return gp_normed.unsqueeze(1).expand(-1, N, -1)                         # [B, N, 4]

    # --------------------------------------------------------------------- #
    # Single-sample path
    # --------------------------------------------------------------------- #

    def _forward_single(
        self,
        x: torch.Tensor,
        aux: dict,
        *,
        predict_volume: bool,
        predict_surface: bool,
    ) -> torch.Tensor | dict:
        """Single-sample forward: x [N, 7]."""
        coords = aux["coords"]              # [N, 2]
        sdf    = aux["sdf"]                 # [N, 1]
        global_params = aux["global_params"]  # [4]
        knn_idx    = aux["knn_idx"]         # [N, k]
        knn_rel_pos = aux["knn_rel_pos"]    # [N, k, 2]
        grid_sdf   = aux["grid_sdf"]        # [1, 1, H, W]
        grid_coords_norm = aux["grid_coords_norm"]  # [N, 2]

        N = coords.shape[0]
        k = self.k_neighbors

        # 1. Shared GeoProcessor
        geo_feat_grid = self.geo_processor(grid_sdf)  # [1, geo_feat_dim, H, W]

        vol_out = None
        if predict_volume:
            # 2. Sample geo features at each volume node
            sample_grid = grid_coords_norm.view(1, 1, N, 2)  # [1, 1, N, 2]
            geo_sampled = F.grid_sample(
                geo_feat_grid, sample_grid,
                align_corners=True, mode="bilinear", padding_mode="border",
            )  # [1, geo_feat_dim, 1, N]
            geo_sampled = geo_sampled.squeeze(0).squeeze(1).T  # [N, geo_feat_dim]

            # 3. Local k-NN aggregation
            neighbor_sdf = sdf[knn_idx]                            # [N, k, 1]
            local_in = torch.cat([knn_rel_pos, neighbor_sdf], dim=-1).view(N, k * 3)
            local_enc = self.local_point_conv(local_in)            # [N, local_feat_dim_vol]

            # 4. Fourier basis
            node_enc = torch.cat([coords, sdf], dim=-1)            # [N, 3]
            basis = self.fourier_mlp(node_enc)                     # [N, basis_dim_vol]

            # 5. Aggregation
            gp = self._normalize_global_params_single(global_params, N)  # [N, 4]
            agg_in = torch.cat([geo_sampled, local_enc, basis, gp], dim=-1)
            vol_out = self.agg_mlp(agg_in)                         # [N, out_dim]

        surf_out = None
        if predict_surface:
            surf_out = self._surface_single(aux, geo_feat_grid, global_params)

        return self._pack_output(vol_out, surf_out, predict_volume, predict_surface)

    def _surface_single(
        self,
        aux: dict,
        geo_feat_grid: torch.Tensor,
        global_params: torch.Tensor,
    ) -> torch.Tensor:
        """Run surface head for a single sample. Returns [M, 1]."""
        surf_xy     = aux["surf_xy"]              # [M, 2]
        surf_n      = aux["surf_n"]               # [M, 2]
        surf_ds     = aux["surf_ds"]              # [M, 1]
        surf_grid_coords_norm = aux["surf_grid_coords_norm"]  # [M, 2]
        surf_knn_idx    = aux["surf_knn_idx"]     # [M, k_surf]
        surf_knn_rel_pos = aux["surf_knn_rel_pos"] # [M, k_surf, 2]

        M = surf_xy.shape[0]
        k_s = self.k_surf

        # 1. Sample shared geo features at surface points
        sample_grid = surf_grid_coords_norm.view(1, 1, M, 2)   # [1, 1, M, 2]
        geo_surf = F.grid_sample(
            geo_feat_grid, sample_grid,
            align_corners=True, mode="bilinear", padding_mode="border",
        )  # [1, geo_feat_dim, 1, M]
        geo_surf = geo_surf.squeeze(0).squeeze(1).T  # [M, geo_feat_dim]

        # 2. Surface local k-NN aggregation: input = [rel_x, rel_y, neighbor_ds]
        neighbor_ds = surf_ds[surf_knn_idx]                       # [M, k_s, 1]
        local_in = torch.cat([surf_knn_rel_pos, neighbor_ds], dim=-1).view(M, k_s * 3)
        local_enc = self.surf_local_point_conv(local_in)          # [M, local_feat_dim_surf]

        # 3. Surface Fourier basis: raw_in=6 → (x, y, sdf≈0, n_x, n_y, log_ds_scaled)
        # sdf ≈ 0 at the wall (confirmed: mean |sdf| ≈ 2e-4 for wall nodes).
        # log_ds scaled: subtract log(mean_ds) so values are O(1) and centered near 0.
        # mean_ds ≈ perimeter/M ≈ 2.06/256 ≈ 0.008 for chord-1 airfoil.
        # log(mean_ds) ≈ log(0.008) ≈ -4.8; after shifting, log_ds_scaled is ~O(1).
        sdf_zero = torch.zeros(M, 1, device=surf_xy.device, dtype=surf_xy.dtype)
        mean_ds = surf_ds.mean().clamp(min=1e-8)
        log_ds_scaled = (torch.log(surf_ds.clamp(min=1e-8)) - torch.log(mean_ds))  # [M, 1]
        node_enc = torch.cat([surf_xy, sdf_zero, surf_n, log_ds_scaled], dim=-1)  # [M, 6]
        basis = self.surf_fourier_mlp(node_enc)                   # [M, basis_dim_surf]

        # 4. Aggregation with shared global params
        gp = self._normalize_global_params_single(global_params, M)  # [M, 4]
        agg_in = torch.cat([geo_surf, local_enc, basis, gp], dim=-1)
        surf_out = self.surf_agg_mlp(agg_in)                      # [M, 1]
        return surf_out

    # --------------------------------------------------------------------- #
    # Batched path
    # --------------------------------------------------------------------- #

    def _forward_batched(
        self,
        x: torch.Tensor,
        aux: dict,
        *,
        predict_volume: bool,
        predict_surface: bool,
    ) -> torch.Tensor | dict:
        """Batched forward: x [B, N, 7]."""
        coords = aux["coords"]              # [B, N, 2]
        sdf    = aux["sdf"]                 # [B, N, 1]
        global_params = aux["global_params"]  # [B, 4]
        knn_idx    = aux["knn_idx"]         # [B, N, k]
        knn_rel_pos = aux["knn_rel_pos"]    # [B, N, k, 2]
        grid_sdf   = aux["grid_sdf"]        # [B, 1, H, W]
        grid_coords_norm = aux["grid_coords_norm"]  # [B, N, 2]

        B, N, _ = coords.shape
        k = self.k_neighbors

        # 1. Shared GeoProcessor (Conv2d batches over B naturally)
        geo_feat_grid = self.geo_processor(grid_sdf)  # [B, geo_feat_dim, H, W]

        vol_out = None
        if predict_volume:
            # 2. Sample geo features at each volume node
            sample_grid = grid_coords_norm.view(B, 1, N, 2)  # [B, 1, N, 2]
            geo_sampled = F.grid_sample(
                geo_feat_grid, sample_grid,
                align_corners=True, mode="bilinear", padding_mode="border",
            )  # [B, geo_feat_dim, 1, N]
            geo_sampled = geo_sampled.squeeze(2).permute(0, 2, 1)  # [B, N, geo_feat_dim]

            # 3. Local k-NN aggregation
            idx_flat = knn_idx.reshape(B, N * k, 1)
            neighbor_sdf = torch.gather(sdf, 1, idx_flat.expand(B, N * k, 1))  # [B, N*k, 1]
            neighbor_sdf = neighbor_sdf.reshape(B, N, k, 1)
            local_in = torch.cat([knn_rel_pos, neighbor_sdf], dim=-1).reshape(B, N, k * 3)
            local_enc = self.local_point_conv(local_in)                         # [B, N, local_feat_dim]

            # 4. Fourier basis
            node_enc = torch.cat([coords, sdf], dim=-1)  # [B, N, 3]
            basis = self.fourier_mlp(node_enc)           # [B, N, basis_dim]

            # 5. Aggregation
            gp = self._normalize_global_params_batched(global_params, N)  # [B, N, 4]
            agg_in = torch.cat([geo_sampled, local_enc, basis, gp], dim=-1)
            vol_out = self.agg_mlp(agg_in)                                  # [B, N, out_dim]

        surf_out = None
        if predict_surface:
            surf_out = self._surface_batched(aux, geo_feat_grid, global_params, B)

        return self._pack_output(vol_out, surf_out, predict_volume, predict_surface)

    def _surface_batched(
        self,
        aux: dict,
        geo_feat_grid: torch.Tensor,
        global_params: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """Run surface head for a batched input. Returns [B, M, 1]."""
        surf_xy     = aux["surf_xy"]              # [B, M, 2]
        surf_n      = aux["surf_n"]               # [B, M, 2]
        surf_ds     = aux["surf_ds"]              # [B, M, 1]
        surf_grid_coords_norm = aux["surf_grid_coords_norm"]  # [B, M, 2]
        surf_knn_idx    = aux["surf_knn_idx"]     # [B, M, k_surf]
        surf_knn_rel_pos = aux["surf_knn_rel_pos"] # [B, M, k_surf, 2]

        M = surf_xy.shape[1]
        k_s = self.k_surf

        # 1. Sample shared geo features at surface points
        sample_grid = surf_grid_coords_norm.view(B, 1, M, 2)   # [B, 1, M, 2]
        geo_surf = F.grid_sample(
            geo_feat_grid, sample_grid,
            align_corners=True, mode="bilinear", padding_mode="border",
        )  # [B, geo_feat_dim, 1, M]
        geo_surf = geo_surf.squeeze(2).permute(0, 2, 1)          # [B, M, geo_feat_dim]

        # 2. Surface local k-NN aggregation
        # Gather neighbor ds using torch.gather over the M dimension.
        # surf_knn_idx [B, M, k_s], surf_ds [B, M, 1]
        M_flat = M * k_s
        idx_flat = surf_knn_idx.reshape(B, M_flat, 1)
        neighbor_ds = torch.gather(surf_ds, 1, idx_flat.expand(B, M_flat, 1))  # [B, M*k_s, 1]
        neighbor_ds = neighbor_ds.reshape(B, M, k_s, 1)
        local_in = torch.cat([surf_knn_rel_pos, neighbor_ds], dim=-1).reshape(B, M, k_s * 3)
        local_enc = self.surf_local_point_conv(local_in)                       # [B, M, local_feat_dim_surf]

        # 3. Surface Fourier basis
        sdf_zero = torch.zeros(B, M, 1, device=surf_xy.device, dtype=surf_xy.dtype)
        mean_ds = surf_ds.mean(dim=1, keepdim=True).clamp(min=1e-8)   # [B, 1, 1]
        log_ds_scaled = torch.log(surf_ds.clamp(min=1e-8)) - torch.log(mean_ds)  # [B, M, 1]
        node_enc = torch.cat([surf_xy, sdf_zero, surf_n, log_ds_scaled], dim=-1)  # [B, M, 6]
        basis = self.surf_fourier_mlp(node_enc)                               # [B, M, basis_dim_surf]

        # 4. Aggregation
        gp = self._normalize_global_params_batched(global_params, M)   # [B, M, 4]
        agg_in = torch.cat([geo_surf, local_enc, basis, gp], dim=-1)
        surf_out = self.surf_agg_mlp(agg_in)                            # [B, M, 1]
        return surf_out

    @staticmethod
    def _pack_output(
        vol_out: torch.Tensor | None,
        surf_out: torch.Tensor | None,
        predict_volume: bool,
        predict_surface: bool,
    ) -> torch.Tensor | dict:
        if predict_volume and predict_surface:
            return {"volume": vol_out, "surface": surf_out}
        if predict_volume:
            return vol_out
        if predict_surface:
            return surf_out
        raise ValueError("At least one of predict_volume or predict_surface must be True.")


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    # Recommended config
    model = DoMINO(
        base_filters=18,
        base_layer=496,
        base_layer_surf=416,
        num_modes=4,
        k_neighbors=16,
        k_surf=8,
        grid_res=64,
    )
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"TOTAL_PARAMS={n_params_total:,}")

    # Per-submodule breakdown
    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    n_geo = count_params(model.geo_processor)
    n_vol_lpc = count_params(model.local_point_conv)
    n_vol_fmlp = count_params(model.fourier_mlp)
    n_vol_agg = count_params(model.agg_mlp)
    n_vol = n_vol_lpc + n_vol_fmlp + n_vol_agg

    n_surf_lpc = count_params(model.surf_local_point_conv)
    n_surf_fmlp = count_params(model.surf_fourier_mlp)
    n_surf_agg = count_params(model.surf_agg_mlp)
    n_surf = n_surf_lpc + n_surf_fmlp + n_surf_agg

    print(f"\nGeoProcessor:         {n_geo:>10,}  ({100*n_geo/n_params_total:.1f}%)")
    print(f"Volume head total:    {n_vol:>10,}  ({100*n_vol/n_params_total:.1f}%)")
    print(f"  LocalPointConv:     {n_vol_lpc:>10,}")
    print(f"  FourierMLP(in=3):   {n_vol_fmlp:>10,}")
    print(f"  AggMLP(out=6):      {n_vol_agg:>10,}")
    print(f"Surface head total:   {n_surf:>10,}  ({100*n_surf/n_params_total:.1f}%)")
    print(f"  SurfLocalPointConv: {n_surf_lpc:>10,}")
    print(f"  SurfFourierMLP(in=6):{n_surf_fmlp:>9,}")
    print(f"  SurfAggMLP(out=1):  {n_surf_agg:>10,}")
    print(f"TOTAL:                {n_params_total:>10,}")

    # Synthetic volume-only sanity check (backward compat)
    N = 2000
    x = torch.zeros(N, 7)
    x[:, 0] = torch.rand(N) * 10 - 2
    x[:, 1] = torch.rand(N) * 4 - 2
    x[:, 2] = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 0.5
    x[:, 3] = 50.0; x[:, 4] = 0.0; x[:, 5] = 5.0; x[:, 6] = 1e6

    out = model(x)  # volume-only default
    assert out.shape == (N, 6), f"Bad vol shape: {out.shape}"
    assert torch.isfinite(out).all(), "Volume output has NaN/Inf!"
    print("\nVOLUME_ONLY_FORWARD_OK")

    # Dual-head shape check with synthetic surface aux
    from surrogate.domino.datapipe import build_aux
    aux = build_aux(x)
    M = 256
    k_s = 8

    # Synthetic surface aux tensors
    surf_xy = torch.rand(M, 2)
    surf_n = torch.randn(M, 2); surf_n = surf_n / surf_n.norm(dim=-1, keepdim=True)
    surf_ds = torch.rand(M, 1) * 0.01 + 0.005
    surf_grid_coords_norm = torch.rand(M, 2) * 2 - 1
    surf_knn_idx = torch.randint(0, M, (M, k_s))
    surf_knn_rel_pos = torch.randn(M, k_s, 2) * 0.01
    aux.update({
        "surf_xy": surf_xy, "surf_n": surf_n, "surf_ds": surf_ds,
        "surf_grid_coords_norm": surf_grid_coords_norm,
        "surf_knn_idx": surf_knn_idx, "surf_knn_rel_pos": surf_knn_rel_pos,
    })

    result = model(x, aux=aux, predict_volume=True, predict_surface=True)
    assert isinstance(result, dict), "Expected dict for dual-head"
    assert result["volume"].shape == (N, 6), f"Bad vol shape: {result['volume'].shape}"
    assert result["surface"].shape == (M, 1), f"Bad surf shape: {result['surface'].shape}"
    assert torch.isfinite(result["volume"]).all() and torch.isfinite(result["surface"]).all()
    print("DUAL_HEAD_FORWARD_OK")
    print(f"  volume: {result['volume'].shape}  surface: {result['surface'].shape}")
