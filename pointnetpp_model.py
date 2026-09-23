"""PointNet++ for per-point WSS regression on combined wall+interior point clouds.

PointNet++ (Qi et al., 2017): Hierarchical point cloud feature learning.
Adapted for WSS regression: interior points participate in hierarchical
encoding (SA -> FP), enabling the model to learn du/dn from near-wall
velocity gradients without handcrafted features.

Architecture:
  SA1 (fps->512) -> SA2 (fps->128) -> SA3 (fps->32)
  -> FP3 (32->128) -> FP2 (128->512) -> FP1 (512->N_total) -> output (N_wall, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility functions
# ============================================================

def fps(xyz: torch.Tensor, npoints: int) -> torch.Tensor:
    """Farthest Point Sampling -- select npoints centroids.

    Args:
        xyz: (B, N, 3) point coordinates
        npoints: number of centroids to sample

    Returns:
        idx: (B, npoints) indices of sampled points
    """
    B, N, _ = xyz.shape
    device = xyz.device

    if npoints >= N:
        idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        if npoints > N:
            pad = torch.full((B, npoints - N), -1, dtype=torch.long, device=device)
            idx = torch.cat([idx, pad], dim=1)
        return idx

    idx = torch.zeros(B, npoints, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(npoints):
        idx[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)       # fused, compiler-friendly
        farthest = distance.max(dim=-1)[1]

    return idx


def ball_query(radius: float, k: int, xyz: torch.Tensor,
               query_xyz: torch.Tensor) -> torch.Tensor:
    """Ball query: find up to k neighbors within radius for each query point.

    Args:
        radius: search radius (in normalized coordinate space)
        k: max neighbors
        xyz: (B, N, 3) all point coordinates
        query_xyz: (B, M, 3) query point coordinates (centroids)

    Returns:
        idx: (B, M, k) neighbor indices, padded with first point index
    """
    B, N, _ = xyz.shape
    _, M, _ = query_xyz.shape

    dists = torch.cdist(query_xyz, xyz)  # (B, M, N)
    dists[dists > radius] = 1e10          # in-place, 省去 clone 的内存分配
    _, idx = torch.topk(dists, k, dim=-1, largest=False)

    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: (B, N, C) point features
        idx: (B, M, K) indices

    Returns:
        gathered: (B, M, K, C)
    """
    B, N, C = points.shape
    _, M, K = idx.shape

    idx_clamped = idx.clamp(min=0)
    idx_expanded = idx_clamped.unsqueeze(-1).expand(-1, -1, -1, C)
    points_expanded = points.unsqueeze(1).expand(-1, M, -1, -1)

    gathered = points_expanded.gather(2, idx_expanded)  # (B, M, K, C)
    return gathered


def three_nn_interpolate(xyz1: torch.Tensor, xyz2: torch.Tensor,
                         features2: torch.Tensor) -> torch.Tensor:
    """Inverse-distance weighted interpolation from xyz2->xyz1 using 3 nearest neighbors.

    Args:
        xyz1: (B, N1, 3) target points
        xyz2: (B, N2, 3) source points
        features2: (B, N2, C) source features

    Returns:
        interpolated: (B, N1, C) features at target points
    """
    B, N1, _ = xyz1.shape
    _, N2, _ = xyz2.shape
    C = features2.shape[-1]

    dists = torch.cdist(xyz1, xyz2)  # (B, N1, N2)
    top3_dists, top3_idx = torch.topk(dists, 3, dim=-1, largest=False)  # (B, N1, 3)

    weights = 1.0 / (top3_dists + 1e-10)  # (B, N1, 3)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    idx_expanded = top3_idx.unsqueeze(-1).expand(-1, -1, -1, C)
    features2_expanded = features2.unsqueeze(1).expand(-1, N1, -1, -1)
    gathered = features2_expanded.gather(2, idx_expanded)  # (B, N1, 3, C)

    weights = weights.unsqueeze(-1)  # (B, N1, 3, 1)
    interpolated = (gathered * weights).sum(dim=2)  # (B, N1, C)

    return interpolated


# ============================================================
# PointNet++ Modules
# ============================================================

class _SharedMLP(nn.Module):
    """Shared MLP over points: implemented as stacked Conv1d layers."""

    def __init__(self, channels: list, use_bn: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, bias=False))
            if use_bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            if i < len(channels) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N) -> (B, C_out, N)"""
        return self.mlp(x)


class _SetAbstraction(nn.Module):
    """Set Abstraction: FPS -> Ball Query -> Shared MLP -> Max Pool."""

    def __init__(self, npoints: int, radius: float, k: int,
                 in_channels: int, mlp_channels: list):
        super().__init__()
        self.npoints = npoints
        self.radius = radius
        self.k = k
        self.mlp = _SharedMLP([in_channels] + mlp_channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = fps(xyz, self.npoints)  # (B, M)
        M = fps_idx.shape[1]
        fps_idx_clamped = fps_idx.clamp(min=0)
        new_xyz = torch.gather(xyz, 1,
                               fps_idx_clamped.unsqueeze(-1).expand(-1, -1, 3))  # (B, M, 3)

        # Ball query
        group_idx = ball_query(self.radius, self.k, xyz, new_xyz)  # (B, M, K)

        # Gather grouped points
        grouped_xyz = index_points(xyz, group_idx)  # (B, M, K, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # (B, M, K, 3)

        if features is not None:
            grouped_features = index_points(features, group_idx)  # (B, M, K, C_in)
            grouped = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
        else:
            grouped = grouped_xyz_norm

        C_in_group = grouped.shape[-1]
        grouped = grouped.reshape(B * M, self.k, C_in_group)
        grouped = grouped.permute(0, 2, 1)  # (B*M, C, K)

        grouped = self.mlp(grouped)  # (B*M, C_out, K)
        new_features, _ = grouped.max(dim=-1)  # (B*M, C_out)

        C_out = new_features.shape[1]
        new_features = new_features.reshape(B, M, C_out)  # (B, M, C_out)

        return new_xyz, new_features


class _FeaturePropagation(nn.Module):
    """Feature Propagation: Interpolate + Skip Connection + MLP."""

    def __init__(self, in_channels: int, skip_channels: int,
                 mlp_channels: list):
        super().__init__()
        self.mlp = _SharedMLP([in_channels + skip_channels] + mlp_channels)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                features1: torch.Tensor, features2: torch.Tensor):
        # Interpolate features2 from N2->N1
        interpolated = three_nn_interpolate(xyz1, xyz2, features2)  # (B, N1, C2)

        if features1 is not None:
            concat = torch.cat([interpolated, features1], dim=-1)
        else:
            concat = interpolated

        B, N1, C_in = concat.shape
        concat = concat.permute(0, 2, 1)  # (B, C_in, N1)
        out = self.mlp(concat)  # (B, C_out, N1)
        out = out.permute(0, 2, 1)  # (B, N1, C_out)

        return out


# ============================================================
# Main Model
# ============================================================

class PointNetPP(nn.Module):
    """PointNet++ for per-point WSS regression on combined wall+interior point clouds.

    Interior points participate in hierarchical encoding (SA -> FP), enabling
    the model to learn du/dn from near-wall velocity gradients without
    handcrafted features.

    Args:
        in_features: per-point features EXCLUDING coords
            - coords_only: in_features=1 (is_wall flag)
            - coords_features: in_features=3 (|v|, p, is_wall)
        sa_npoints: centroids per SA level
        sa_radii: ball query radii
        sa_mlps: shared MLP channels per SA level
        fp_mlps: shared MLP channels per FP level
    """

    def __init__(self, in_features: int = 0,
                 sa_npoints: list = None,
                 sa_radii: list = None,
                 sa_mlps: list = None,
                 fp_mlps: list = None):
        super().__init__()

        if sa_npoints is None:
            sa_npoints = [512, 128, 32]
        if sa_radii is None:
            sa_radii = [0.1, 0.2, 0.4]
        if sa_mlps is None:
            sa_mlps = [
                [64, 64, 128],
                [128, 128, 256],
                [256, 256, 512],
            ]
        if fp_mlps is None:
            fp_mlps = [
                [256, 256],
                [256, 128],
                [128, 64, 1],
            ]

        self.in_features = in_features
        self.coords = None
        self._xyz_norm = None

        # ---- Set Abstraction layers ----
        sa1_in = 3 + (in_features if in_features > 0 else 0)
        self.sa1 = _SetAbstraction(sa_npoints[0], sa_radii[0], 32,
                                    sa1_in, sa_mlps[0])

        self.sa2 = _SetAbstraction(sa_npoints[1], sa_radii[1], 32,
                                    3 + sa_mlps[0][-1], sa_mlps[1])

        self.sa3 = _SetAbstraction(sa_npoints[2], sa_radii[2], 32,
                                    3 + sa_mlps[1][-1], sa_mlps[2])

        # ---- Feature Propagation layers ----
        sa_out = [sa_mlps[0][-1], sa_mlps[1][-1], sa_mlps[2][-1]]

        self.fp3 = _FeaturePropagation(sa_out[2], sa_out[1], fp_mlps[0])
        self.fp2 = _FeaturePropagation(fp_mlps[0][-1], sa_out[0], fp_mlps[1])
        self.fp1 = _FeaturePropagation(fp_mlps[1][-1],
                                        in_features if in_features > 0 else 0,
                                        fp_mlps[2])

    def set_coords(self, coords: torch.Tensor):
        """Store ALL point coordinates (wall + interior) and normalize.

        Args:
            coords: (N_total, 3) combined wall + interior coordinates
        """
        center = coords.mean(dim=0, keepdim=True)
        centered = coords - center
        scale = centered.norm(dim=1).max()
        if scale < 1e-8:
            scale = torch.tensor(1.0, device=coords.device)
        self._scale = scale
        self._xyz_norm = centered / scale

    def forward(self, combined_features: torch.Tensor,
                wall_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass returning WSS at wall points.

        Args:
            combined_features: (N_total, 3+in_features) point features
                First 3 columns = [x, y, z], rest = per-point features.
            wall_mask: (N_total,) bool, True for wall points.

        Returns:
            wss: (N_wall, 1) predicted WSS
        """
        if self._xyz_norm is None:
            raise RuntimeError("set_coords() must be called before forward()")

        B = 1
        N = self._xyz_norm.shape[0]
        device = self._xyz_norm.device

        xyz = self._xyz_norm.unsqueeze(0)  # (1, N_total, 3)

        # Extract per-point features (non-coord columns)
        if self.in_features > 0 and combined_features.shape[1] > 3:
            point_feat = combined_features[:, 3:].unsqueeze(0)  # (1, N, in_features)
        else:
            point_feat = None

        # ---- Hierarchical encoding (all points) ----
        xyz1, feat1 = self.sa1(xyz, point_feat)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)

        # ---- Feature propagation (decode back to all points) ----
        fp3_out = self.fp3(xyz2, xyz3, feat2, feat3)
        fp2_out = self.fp2(xyz1, xyz2, feat1, fp3_out)
        fp1_out = self.fp1(xyz, xyz1, point_feat, fp2_out)  # (1, N_total, 1)

        wss_all = fp1_out.squeeze(0)  # (N_total, 1)

        if wall_mask is not None:
            return wss_all[wall_mask]  # (N_wall, 1)
        return wss_all

    def forward_coupled(self, combined_features: torch.Tensor,
                        velocity_real: torch.Tensor = None,
                        pressure_real: torch.Tensor = None,
                        wall_mask: torch.Tensor = None):
        """Compatible interface with DirectWSSPredictor.forward_coupled."""
        wss = self.forward(combined_features, wall_mask=wall_mask)
        magnitude = torch.abs(wss)
        sign_prob = (torch.sign(wss) + 1.0) / 2.0
        return wss, magnitude, sign_prob
