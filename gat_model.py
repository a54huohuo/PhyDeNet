"""GAT (Graph Attention Network) for per-point WSS regression.

Veličković et al., 2018 — "Graph Attention Networks"
https://arxiv.org/abs/1710.10903

Builds a static k-NN graph from combined wall+interior coordinates, then applies
multi-head graph attention layers to aggregate neighborhood information
for per-point WSS prediction. Interior points participate in message passing
but are not directly supervised.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
import numpy as np


def build_knn_graph(coords: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Build k-NN graph (excluding self).

    Uses GPU-native torch.cdist + topk when coords are on GPU (avoids CPU-GPU
    transfer). Falls back to scipy cKDTree for CPU tensors or very large N.

    Args:
        coords: (N, 3)
        k: number of neighbors (k-1 excluding self, returned as k-1)

    Returns:
        edge_index: (2, N * (k-1)) — [src, dst] edge pairs
    """
    N = coords.shape[0]
    k_excl = k - 1  # exclude self

    if coords.is_cuda and N <= 18000:
        # GPU-native path: cdist + topk, no CPU round-trip
        with torch.no_grad():
            dists = torch.cdist(coords, coords)           # (N, N)
            # k includes self; keep k neighbors, drop first (self)
            _, knn_idx = torch.topk(dists, k, dim=-1, largest=False)
            knn_idx = knn_idx[:, 1:]                       # (N, k-1), exclude self
    else:
        # CPU path: scipy cKDTree (O(N log N))
        coords_np = coords.detach().cpu().numpy().astype(np.float64)
        tree = cKDTree(coords_np)
        _, knn_idx = tree.query(coords_np, k=k)            # (N, k), col 0 = self
        knn_idx = knn_idx[:, 1:]                           # exclude self: (N, k-1)
        knn_idx = torch.LongTensor(knn_idx)

    # Build edge_index: [src, dst] pairs
    dst = torch.arange(N, device=knn_idx.device).unsqueeze(1).expand(-1, k_excl).reshape(-1)
    src = knn_idx.reshape(-1)

    edge_index = torch.stack([src, dst], dim=0)  # (2, E)
    return edge_index


class _GATConv(nn.Module):
    """Single GAT layer with multi-head attention."""

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 4, dropout: float = 0.2,
                 concat: bool = True):
        """
        Args:
            in_channels: input feature dim
            out_channels: output feature dim PER HEAD
            heads: number of attention heads
            dropout: attention dropout
            concat: if True, concat heads → heads*out_channels;
                    if False (last layer), average heads → out_channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat

        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

        if concat:
            self.bias = nn.Parameter(torch.zeros(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) [src, dst] edge pairs

        Returns:
            out: (N, heads*out) if concat else (N, out)
        """
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        Wh = self.W(x)  # (N, H * F')
        Wh = Wh.view(N, self.heads, self.out_channels)  # (N, H, F')

        Wh_src = Wh[src]  # (E, H, F')
        Wh_dst = Wh[dst]  # (E, H, F')

        alpha_src = (Wh_src * self.att_src).sum(dim=-1)  # (E, H)
        alpha_dst = (Wh_dst * self.att_dst).sum(dim=-1)  # (E, H)
        e = F.leaky_relu(alpha_src + alpha_dst, 0.2)  # (E, H)

        e_max = torch.zeros(N, self.heads, device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, self.heads),
                              e, reduce='amax', include_self=False)
        e_exp = torch.exp(e - e_max[dst])  # (E, H)

        e_sum = torch.zeros(N, self.heads, device=x.device)
        e_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.heads), e_exp)

        alpha = e_exp / (e_sum[dst] + 1e-8)  # (E, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        alpha_expanded = alpha.unsqueeze(-1)  # (E, H, 1)
        weighted = Wh_src * alpha_expanded  # (E, H, F')

        out = torch.zeros(N, self.heads, self.out_channels, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1)
                         .expand(-1, self.heads, self.out_channels), weighted)

        if self.concat:
            out = out.reshape(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return out


class GAT(nn.Module):
    """GAT for per-point WSS regression on combined wall+interior point clouds.

    Interior points participate in message passing via k-NN graph attention,
    enabling the model to learn ∂u/∂n from near-wall velocity profiles
    without handcrafted features.
    """

    def __init__(self, in_features: int = 0, k: int = 16,
                 hidden_dim: int = 64, heads: int = 4,
                 num_layers: int = 3):
        """
        Args:
            in_features: input feature dim EXCLUDING coords
                - combined (coords_only): in_features includes is_wall flag (1)
                - combined (coords_features): in_features = |v| + p + is_wall (3)
            k: number of neighbors for k-NN graph
            hidden_dim: hidden dimension per attention head
            heads: number of attention heads
            num_layers: number of GAT layers
        """
        super().__init__()
        self.in_features = in_features
        self.k = k
        self.coords = None
        self._edge_index = None

        # Input dim: coords(3) + features(in_features)
        total_in = 3 + max(in_features, 0)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        in_ch = total_in

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_ch = hidden_dim
            n_heads = heads if not is_last else 1
            concat = not is_last
            self.gat_layers.append(
                _GATConv(in_ch, out_ch, heads=n_heads, concat=concat, dropout=0.2)
            )
            if concat:
                in_ch = out_ch * n_heads
            else:
                in_ch = out_ch

        # Final regression MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(in_ch, 128),
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
        self._edge_index = build_knn_graph(coords, k=self.k + 1)
        if coords.is_cuda:
            self._edge_index = self._edge_index.to(coords.device)

        # Precompute normalization params (avoid recomputing every forward)
        self._center = coords.mean(dim=0, keepdim=True)
        centered = coords - self._center
        self._scale = centered.norm(dim=1).max().clamp(min=1e-8)

    def forward(self, combined_features: torch.Tensor,
                wall_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            combined_features: (N_total, in_features+3) combined point features
                First 3 columns = coordinates [x, y, z].
            wall_mask: (N_total,) bool, True for wall points.

        Returns:
            wss: (N_wall, 1) predicted WSS
        """
        if self._edge_index is None:
            raise RuntimeError("set_coords() must be called before forward()")

        device = combined_features.device

        if self.coords is not None:
            coords = self.coords
        else:
            coords = combined_features[:, :3]

        # Normalize coords (precomputed params from set_coords)
        if hasattr(self, '_center'):
            xyz_norm = (coords - self._center) / self._scale
        else:
            center = coords.mean(dim=0, keepdim=True)
            centered = coords - center
            scale = centered.norm(dim=1).max().clamp(min=1e-8)
            xyz_norm = centered / scale

        # Build input: normalized coords + additional features
        if combined_features.shape[1] > 3:
            extra_feat = combined_features[:, 3:]
            x = torch.cat([xyz_norm, extra_feat], dim=1)
        else:
            x = xyz_norm

        # GAT layers
        for gat in self.gat_layers:
            x = gat(x, self._edge_index)
            x = F.elu(x)

        # Final MLP
        wss_all = self.final_mlp(x)  # (N_total, 1)

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
