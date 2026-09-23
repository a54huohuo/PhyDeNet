"""DGCNN (Dynamic Graph CNN) for per-point WSS regression.

Wang et al., 2019 — "Dynamic Graph CNN for Learning on Point Clouds"
https://arxiv.org/abs/1801.07829

EdgeConv: for each point, computes edge features [h_i, h_j - h_i] over a
k-NN graph, then applies a shared MLP + max-pool to aggregate neighborhood
information.

Memory note: For large point clouds (~16K points), the k-NN graph is
precomputed from coordinates (static graph) to avoid O(N²) pairwise distance
computations in feature space per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
import numpy as np


def build_knn_graph(coords: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Build k-NN graph from coordinates.

    Uses GPU-native torch.cdist + topk when coords are on GPU (avoids CPU-GPU
    transfer). Falls back to scipy cKDTree for CPU tensors or very large N
    where cdist's O(N²) memory would exceed ~2 GB.

    Args:
        coords: (N, 3) coordinates (CPU or GPU)
        k: number of neighbors (including self)

    Returns:
        knn_idx: (N, k) indices of k nearest neighbors (first col = self)
    """
    N = coords.shape[0]

    # Threshold: cdist(N,N) at float32 = N²×4 bytes; keep under ~1.5 GB
    if coords.is_cuda and N <= 18000:
        # GPU-native path: single cdist + topk, no CPU round-trip
        with torch.no_grad():
            dists = torch.cdist(coords, coords)           # (N, N)
            _, knn_idx = torch.topk(dists, k, dim=-1, largest=False)
        return knn_idx

    # CPU path: scipy cKDTree (O(N log N), works for any N)
    coords_np = coords.detach().cpu().numpy().astype(np.float64)
    tree = cKDTree(coords_np)
    _, knn_idx = tree.query(coords_np, k=k)
    return torch.LongTensor(knn_idx)


def index_neighbors(features: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbor features.

    Args:
        features: (N, C) point features
        knn_idx: (N, k) neighbor indices

    Returns:
        neighbor_feat: (N, k, C) gathered features
    """
    N, C = features.shape
    k = knn_idx.shape[1]
    # features[knn_idx]: (N, k, C)
    return features[knn_idx.view(-1)].view(N, k, C)


class _EdgeConv(nn.Module):
    """Single EdgeConv layer.

    For each edge (i, j): h_ij = MLP([h_i, h_j - h_i])
    Output: max_j(h_ij)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        # Input: [h_i, h_j - h_i] = 2 * in_channels
        self.mlp = nn.Sequential(
            nn.Conv1d(2 * in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in) point features
            knn_idx: (N, k) precomputed neighbor indices

        Returns:
            out: (N, C_out)
        """
        N, C = x.shape
        k = knn_idx.shape[1]

        # Gather neighbors: (N, k, C)
        x_neighbors = x[knn_idx.view(-1)].view(N, k, C)

        # Self features expanded: (N, k, C)
        x_self = x.unsqueeze(1).expand(-1, k, -1)

        # Edge features: [h_i, h_j - h_i]: (N, k, 2*C)
        edge_feat = torch.cat([x_self, x_neighbors - x_self], dim=-1)

        # Reshape for Conv1d: (N, 2*C, k)
        edge_feat = edge_feat.permute(0, 2, 1)

        # Shared MLP: (N, C_out, k)
        edge_feat = self.mlp(edge_feat)

        # Max-pool over neighbors: (N, C_out)
        out, _ = edge_feat.max(dim=-1)

        return out


class DGCNN(nn.Module):
    """DGCNN adapted for per-point WSS regression.

    Uses static k-NN graph (precomputed from coordinates) with multiple
    EdgeConv layers. Per-layer features are concatenated (DenseNet-style
    skip connections), then processed by a final MLP.

    Supports combined wall+interior point clouds: interior points participate
    in message passing but are not directly supervised. This allows the model
    to learn ∂u/∂n from near-wall velocity profiles.
    """

    def __init__(self, in_features: int = 0, k: int = 20,
                 conv_channels: list = None):
        """
        Args:
            in_features: input feature dimension (EXCLUDING coords)
                - combined mode (coords_only): in_features should include is_wall flag
                - combined mode (coords_features): in_features = |v| + p + is_wall flag
            k: number of nearest neighbors
            conv_channels: output channels per EdgeConv layer
        """
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 64, 128, 256]

        self.in_features = in_features
        self.k = k
        self.conv_channels = conv_channels
        self.coords = None
        self._knn_idx = None

        # Input dim: coords(3) + features(in_features)
        total_in = 3 + max(in_features, 0)

        # EdgeConv layers
        self.conv_layers = nn.ModuleList()
        in_ch = total_in
        for out_ch in conv_channels:
            self.conv_layers.append(_EdgeConv(in_ch, out_ch))
            in_ch = out_ch

        # Concatenated feature dim
        concat_dim = sum(conv_channels)

        # Final regression MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def set_coords(self, coords: torch.Tensor):
        """Store ALL point coordinates (wall + interior) and precompute k-NN graph.

        Also precomputes coordinate normalization params to avoid redundant
        mean/std computation in every forward pass.

        Args:
            coords: (N_total, 3) combined wall + interior coordinates
        """
        self.coords = coords
        self._knn_idx = build_knn_graph(coords, k=self.k)
        if coords.is_cuda:
            self._knn_idx = self._knn_idx.to(coords.device)

        # Precompute normalization params (avoid recomputing every forward)
        self._center = coords.mean(dim=0, keepdim=True)
        centered = coords - self._center
        self._scale = centered.norm(dim=1).max().clamp(min=1e-8)

    def forward(self, combined_features: torch.Tensor,
                wall_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            combined_features: (N_total, in_features) combined point features
                Features should include coords as first 3 columns:
                [x, y, z, ...additional features...]
            wall_mask: (N_total,) bool, True for wall points. If None, return all.

        Returns:
            wss: (N_wall, 1) predicted WSS at wall points
        """
        if self._knn_idx is None:
            raise RuntimeError("set_coords() must be called before forward()")

        N = combined_features.shape[0]
        device = combined_features.device

        # Use coords from features (first 3 columns) or from stored coords
        if self.coords is not None:
            coords = self.coords
        else:
            coords = combined_features[:, :3]

        # Normalize coords to unit sphere (precomputed params from set_coords)
        if hasattr(self, '_center'):
            xyz_norm = (coords - self._center) / self._scale
        else:
            center = coords.mean(dim=0, keepdim=True)
            centered = coords - center
            scale = centered.norm(dim=1).max().clamp(min=1e-8)
            xyz_norm = centered / scale

        # Build input: normalized coords + additional features
        if combined_features.shape[1] > 3:
            extra_feat = combined_features[:, 3:]  # non-coord features
        else:
            extra_feat = None

        if extra_feat is not None:
            x = torch.cat([xyz_norm, extra_feat], dim=1)
        else:
            x = xyz_norm

        # EdgeConv layers with skip concatenation
        layer_outputs = []
        for conv in self.conv_layers:
            x = conv(x, self._knn_idx)
            layer_outputs.append(x)

        # Concatenate all layer outputs
        x_cat = torch.cat(layer_outputs, dim=1)

        # Final MLP
        wss_all = self.final_mlp(x_cat)  # (N_total, 1)

        if wall_mask is not None:
            return wss_all[wall_mask]  # (N_wall, 1)
        return wss_all

    def forward_coupled(self, combined_features: torch.Tensor,
                        velocity_real: torch.Tensor = None,
                        pressure_real: torch.Tensor = None,
                        wall_mask: torch.Tensor = None):
        """Compatible interface with DirectWSSPredictor.forward_coupled.

        For point cloud models, velocity_real and pressure_real are embedded
        in combined_features. These params are kept for interface compatibility.
        """
        wss = self.forward(combined_features, wall_mask=wall_mask)
        magnitude = torch.abs(wss)
        sign_prob = (torch.sign(wss) + 1.0) / 2.0
        return wss, magnitude, sign_prob
