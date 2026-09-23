"""Vanilla PointNet for per-point WSS regression.

Qi et al., 2017 — "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
https://arxiv.org/abs/1606.05843

This is a LIGHTWEIGHT alternative to PointNet++: no ball query, no FPS, no
feature propagation, no hierarchical structure.  The architecture is O(N)
in the number of points — shared MLP per-point + global max pooling —
making it 50–100× faster than PointNet++ while still serving as a valid
end-to-end point cloud baseline.

Two variants (selectable via `in_features`):
  - `pointnet_xyz`:   in_features=0 → (x, y, z) only, pure geometric black-box
  - `pointnet`:       in_features=2 → (x, y, z, velocity, pressure), physics-aware

Wall-only mode:
  By default the model operates on wall points only (set via `set_coords`),
  avoiding the interior point overhead that makes DGCNN / GAT / PointNet++
  prohibitively slow.  If interior points are desired, pass the full combined
  coordinate tensor to `set_coords`.

Architecture:
  Input (N, 3+d_in) ─┬─ SharedMLP [64, 128, 256] → local_feat (N, 256)
                      │
                      └─ Global Max Pooling ─────────→ global_feat (256,)
                                                       │
                              Concat ← [local_feat, tiled(global_feat)]
                                                       │
                              Final MLP [512, 256, 128] → output (N, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SharedMLP1d(nn.Module):
    """1×1 Conv-based shared MLP (PointNet-style per-point feature extraction)."""

    def __init__(self, channels: list, use_bn: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1,
                                    bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            if i < len(channels) - 2:  # no ReLU+dropout after final layer
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N) → (B, C', N)"""
        return self.net(x)


class PointNet(nn.Module):
    """Vanilla PointNet adapted for per-point WSS regression.

    Segmentation-style architecture:
      1. Shared MLP → per-point local features
      2. Global max pooling → global feature
      3. Concat [local + tiled(global)] → per-point MLP → per-point WSS
    """

    def __init__(self, in_features: int = 0,
                 local_channels: list = None,
                 head_channels: list = None,
                 dropout: float = 0.2,
                 use_combined: bool = False):
        """
        Args:
            in_features: number of per-point scalar features BEYOND xyz.
                         0 = coords only (3 input dims).
                         2 = coords + velocity + pressure (5 input dims).
                         3 = coords + |v| + p + is_wall (6 input dims, combined cloud).
            local_channels: shared MLP channel list (excluding input dim).
            head_channels:   final MLP channel list.
            dropout:         dropout rate.
            use_combined:    if True, uses wall+interior combined point cloud;
                             forward_coupled accepts wall_mask to extract wall-only
                             predictions for supervised training.
        """
        super().__init__()

        self.in_features = in_features
        self.use_combined = use_combined
        input_dim = 3 + in_features  # xyz + optional scalars

        if local_channels is None:
            local_channels = [64, 128, 256]
        if head_channels is None:
            head_channels = [512, 256, 128]

        # Shared MLP for per-point feature extraction
        self.local_mlp = _SharedMLP1d(
            [input_dim] + local_channels,
            use_bn=True, dropout=dropout
        )
        local_out = local_channels[-1]  # 256

        # Final prediction head: [local + global] → per-point WSS
        head_in = local_out + local_out  # local_feat (256) + tiled global (256)
        head_dims = [head_in] + head_channels + [1]
        fc_layers = []
        for i in range(len(head_dims) - 1):
            fc_layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
            if i < len(head_dims) - 2:
                fc_layers.append(nn.BatchNorm1d(head_dims[i + 1]))
                fc_layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    fc_layers.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*fc_layers)

        # Stored coordinates (set per case via set_coords)
        self.register_buffer('_coords', torch.empty(0, 3), persistent=False)
        self._feat_cache = None  # non-trainable per-point features

    def set_coords(self, coords):
        """Set wall-point coordinates for the current case.

        Args:
            coords: np.ndarray or torch.Tensor of shape (N_wall, 3)
        """
        if isinstance(coords, torch.Tensor):
            pass
        else:
            coords = torch.from_numpy(np.asarray(coords, dtype=np.float32))
        self._coords = coords

    def set_features(self, features):
        """Optionally set per-point features (velocity, pressure).

        Args:
            features: np.ndarray or torch.Tensor of shape (N_wall, in_features)
        """
        if isinstance(features, torch.Tensor):
            self._feat_cache = features
        else:
            self._feat_cache = torch.from_numpy(
                np.asarray(features, dtype=np.float32)
            )

    def forward(self, features=None, velocity_real=None, pressure_real=None):
        """Forward pass.  Coordinates are used from set_coords(); optional
        per-point scalar features from arguments or set_features().

        Compatible with the trainer/evaluator calling convention:
            model(wall_features, wall_velocity, wall_pressure)

        For combined point cloud mode (use_combined=True), the `features`
        argument contains combined_features[:, 3:] — per-point scalars
        beyond xyz (e.g. is_wall, |v|, p).  The xyz coords are read from
        self._coords (set by set_coords prior to forward).

        The `features` argument (51-dim engineered features) is IGNORED in
        wall-only mode — this is intentionally a black-box model that
        operates on raw coordinates + scalars only.
        """
        if self._coords.numel() == 0:
            raise RuntimeError(
                "PointNet.set_coords() must be called before forward()"
            )

        device = self._coords.device
        N = self._coords.shape[0]

        # Build input tensor: (N, 3 + in_features)
        if self.use_combined:
            # Combined point cloud: `features` = combined_features where
            # columns 0:3 = xyz (duplicated in self._coords) and
            # columns 3:  = per-point scalars (is_wall, |v|, p, ...).
            # Extract only the scalar columns past xyz.
            if features is not None and features.shape[1] >= 3 + self.in_features:
                extra = features[:, 3:3 + self.in_features].to(device).float()
                if extra.shape[0] != N:
                    extra = extra[:N]
            else:
                extra = torch.zeros(N, self.in_features, device=device)
            pts = torch.cat([self._coords.to(device).float(), extra], dim=1)
        elif self.in_features == 0:
            pts = self._coords  # (N, 3)
        elif self.in_features == 2:
            # Use velocity + pressure provided by trainer
            if velocity_real is not None and pressure_real is not None:
                vel = velocity_real.to(device).float().view(-1, 1)
                pres = pressure_real.to(device).float().view(-1, 1)
                if vel.shape[0] != N:
                    vel = vel[:N]
                    pres = pres[:N]
            elif self._feat_cache is not None:
                fc = self._feat_cache.to(device).float()
                vel = fc[:, 0:1]
                pres = fc[:, 1:2]
            else:
                vel = torch.zeros(N, 1, device=device)
                pres = torch.zeros(N, 1, device=device)
            pts = torch.cat([self._coords, vel, pres], dim=1)  # (N, 5)
        else:
            # Generic: use features from set_features or argument
            if self._feat_cache is not None:
                fc = self._feat_cache.to(device).float()
                pts = torch.cat([self._coords, fc[:, :self.in_features]], dim=1)
            else:
                extra = torch.zeros(N, self.in_features, device=device)
                pts = torch.cat([self._coords, extra], dim=1)

        # Shared MLP per-point: (N, C_in) → (1, C_in, N) → (1, C_out, N) → (N, C_out)
        x = pts.unsqueeze(0).transpose(1, 2)  # (1, C_in, N)
        local_feat = self.local_mlp(x)          # (1, C_out, N)
        local_feat = local_feat.squeeze(0).transpose(0, 1)  # (N, C_out)

        # Global max pooling
        global_feat = local_feat.max(dim=0)[0]  # (C_out,)
        global_feat = global_feat.unsqueeze(0).expand(N, -1)  # (N, C_out)

        # Concat [local, global] → per-point prediction
        combined = torch.cat([local_feat, global_feat], dim=1)  # (N, 2*C_out)
        wss = self.head(combined)  # (N, 1)

        return wss

    def forward_coupled(self, features=None, velocity_real=None, pressure_real=None,
                         wall_mask=None):
        """Compatibility with trainer interface — returns (wss, magnitude, sign_prob).

        When use_combined=True and wall_mask is provided, predictions are sliced
        to wall points only for supervised training.  The model still performs
        a full forward pass over all (wall+interior) points so interior points
        contribute to the global max-pooling feature.

        Since this is a single-head black-box model, magnitude and sign_prob are
        derived from the raw wss prediction (no sign-magnitude decoupling).
        """
        wss_all = self.forward(features, velocity_real, pressure_real)

        if self.use_combined and wall_mask is not None:
            wss = wss_all[wall_mask]
        else:
            wss = wss_all

        magnitude = torch.abs(wss)
        sign_prob = (torch.sign(wss) + 1.0) / 2.0  # hard threshold, not sigmoid
        return wss, magnitude, sign_prob


import numpy as np  # noqa: E402 — for set_coords numpy fallback
