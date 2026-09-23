"""神经网络模型定义"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import EXP_CONFIG, DEVICE

_HAS_COMPILE = hasattr(torch, 'compile')


def maybe_compile(model: nn.Module, enabled: bool = True) -> nn.Module:
    """torch.compile 包装，加速训练/推理（PyTorch >= 2.0）

    DirectWSSPredictor / CoupledSingleHeadPredictor：forward 中存在条件分支和
    动态切片，fullgraph=False 下会产生大量 graph break，反而更慢 → 默认关闭。

    PointNet++ / DGCNN / GAT：纯数学运算，无动态分支，compile 可融合大量小
    kernel 并启用 CUDA graph 捕获（reduce-overhead），显著减少 kernel launch
    开销 → 默认开启。
    """
    # 点云模型无需分支，compile 收益最大
    _POINT_CLOUD_TYPES = {'pointnetpp', 'dgcnn', 'gat'}

    if not enabled or not _HAS_COMPILE:
        return model

    # 检查是否为点云模型（通过 model_type 属性或类名推断）
    model_type = getattr(model, '_model_type', None)
    is_point_cloud = model_type in _POINT_CLOUD_TYPES

    try:
        import triton  # noqa: F401
        _has_triton = True
    except ImportError:
        _has_triton = False

    try:
        if _has_triton:
            if is_point_cloud:
                # reduce-overhead: CUDA graph 捕获无分支区域，batch=1 下收益显著
                # fullgraph=False: 允许在 BatchNorm 等处安全 graph break
                return torch.compile(model, mode='reduce-overhead', fullgraph=False)
            else:
                return torch.compile(model, mode='reduce-overhead', fullgraph=False)
        else:
            return torch.compile(model, backend='eager', fullgraph=False)
    except Exception:
        return model


class WSSNetwork(nn.Module):
    """通用WSS预测MLP"""

    def __init__(self, layers: list = [17, 64, 128, 128, 64, 1],
                 dropout: float = 0.2):
        super(WSSNetwork, self).__init__()
        self.dropout_rate = dropout

        dims = layers
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(dims[i+1]))
            self.dropouts.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(dims[-1], 1)
        self.activation = nn.SiLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, bn, dropout in zip(self.hidden_layers,
                                        self.batch_norms,
                                        self.dropouts):
            x = linear(x)
            x = bn(x)
            x = self.activation(x)
            x = dropout(x)
        return self.output_layer(x)  # 解耦头输出线性值（可为负）


class DirectWSSPredictor(nn.Module):
    """
    符号-幅值解耦耦合网络

    幅值分支：v/d/wss_ref/n [+ geom] + v,p → 预测 |WSS| (Softplus确保非负)
    符号分支：分离几何特征 + 预测幅值 → 分类符号 (Sigmoid)
    耦合输出：wss = magnitude * (2*sign_prob - 1)

    Flags:
        use_sign_branch:        False → 移除符号分支，wss = magnitude（恒正）
        use_sep_geom:           False → 符号输入仅含幅值（1维），不含12维分离几何
        use_geometry_features:  False → 幅值网络不含4维几何特征(18维而非22维)
        use_velocity_pressure_only: True → 幅值网络仅v+p(2维)，flow_only消融
    """

    def __init__(self, wss_layers: list = [22, 128, 128, 64, 1],
                 sign_layers: list = [13, 64, 32, 1],
                 use_sign_branch: bool = True,
                 use_sep_geom: bool = True,
                 detach_magnitude: bool = True,
                 use_velocity_pressure_only: bool = False,
                 use_geometry_features: bool = True):
        super().__init__()

        self.use_sign_branch = use_sign_branch
        self.use_sep_geom = use_sep_geom
        self.detach_magnitude = detach_magnitude
        self.use_velocity_pressure_only = use_velocity_pressure_only
        self.use_geometry_features = use_geometry_features

        # 幅值网络输入维度
        if use_velocity_pressure_only:
            wss_layers[0] = 2   # 仅 velocity(1) + pressure(1)（无几何/法向量/WSS参考）
        elif not use_geometry_features:
            wss_layers[0] = 18  # 6(v,d) + 1(wss_phys) + 9(n) + 2(v,p)，无4维geom
        else:
            wss_layers[0] = 22  # + 4(geom)
        self.wss_net = WSSNetwork(wss_layers)

        if use_sign_branch:
            sign_layers[0] = 13 if use_sep_geom else 1
            self.sign_net = self._build_mlp(sign_layers, final_activation='sigmoid')
        else:
            self.sign_net = None

        self.gate_thresh = nn.Parameter(torch.tensor(0.4))
        self.gate_k = nn.Parameter(torch.tensor(10.0))

    def _build_mlp(self, layers: list, final_activation: str = None):
        modules = []
        for i in range(len(layers) - 2):
            modules += [
                nn.Linear(layers[i], layers[i + 1]),
                nn.BatchNorm1d(layers[i + 1]),
                nn.SiLU()
            ]
        modules.append(nn.Linear(layers[-2], layers[-1]))
        if final_activation == 'softplus':
            modules.append(nn.Softplus(beta=5.0))
        elif final_activation == 'sigmoid':
            modules.append(nn.Sigmoid())
        return nn.Sequential(*modules)

    def _parse_base_features(self, features: torch.Tensor,
                             velocity_real: torch.Tensor,
                             pressure_real: torch.Tensor) -> torch.Tensor:
        if self.use_velocity_pressure_only:
            # flow_only 消融：仅使用壁面速度幅值和压力（2维）
            # 无几何特征、无法向量、无 WSS 物理参考
            return torch.cat([velocity_real, pressure_real], dim=1)  # 1 + 1 = 2

        v1 = features[:, 4:5];   d1 = features[:, 5:6]
        v2 = features[:, 6:7];   d2 = features[:, 7:8]
        v3 = features[:, 8:9];   d3 = features[:, 9:10]
        wss_phys = features[:, 10:11]
        n1 = features[:, 11:14]
        n2 = features[:, 14:17]
        n3 = features[:, 17:20]

        # 几何特征 (point density proxy, curvedness, shape index, anisotropy)
        # 仅在 use_geometry_features=True 时纳入幅值网络输入
        if self.use_geometry_features:
            n_feats = features.shape[1]
            if n_feats >= 24:
                geom = features[:, 20:24]
            else:
                geom = torch.zeros((features.shape[0], 4), device=features.device)
            return torch.cat([
                v1, d1, v2, d2, v3, d3,      # 6
                wss_phys,                      # 1
                n1, n2, n3,                    # 9
                geom,                          # 4
                velocity_real, pressure_real   # 2
            ], dim=1)                          # = 22
        else:
            # 无几何特征：18维
            return torch.cat([
                v1, d1, v2, d2, v3, d3,      # 6
                wss_phys,                      # 1
                n1, n2, n3,                    # 9
                velocity_real, pressure_real   # 2
            ], dim=1)                          # = 18

    def forward(self, features: torch.Tensor,
                velocity_real: torch.Tensor,
                pressure_real: torch.Tensor) -> torch.Tensor:
        base_input = self._parse_base_features(features, velocity_real, pressure_real)
        magnitude = torch.abs(self.wss_net(base_input))

        if self.use_sign_branch:
            if self.use_sep_geom:
                # 保护：特征维度不足51列时用零填充（与 _parse_base_features 的 geom 保护一致）
                n_feats = features.shape[1]
                if n_feats >= 51:
                    sep_features = features[:, 39:51]
                else:
                    sep_features = torch.zeros((features.shape[0], 12), device=features.device)
            else:
                sep_features = torch.zeros((features.shape[0], 0), device=features.device)
            sign_input = torch.cat([sep_features, magnitude.detach() if self.detach_magnitude else magnitude], dim=-1)
            sign_prob = self.sign_net(sign_input)
        else:
            sign_prob = torch.ones_like(magnitude)  # 移除符号分支 → wss 恒正

        # gate = torch.sigmoid((magnitude - self.gate_thresh) * self.gate_k)
        # adjusted_sign = gate * 1.0 + (1.0 - gate) * (2.0 * sign_prob - 1.0)
        wss = magnitude * (2.0 * sign_prob - 1.0)
        return wss

    def forward_coupled(self, features, velocity_real, pressure_real):
        """训练专用：返回 (wss, magnitude, sign_prob)"""
        base_input = self._parse_base_features(features, velocity_real, pressure_real)
        magnitude = torch.abs(self.wss_net(base_input))

        if self.use_sign_branch:
            if self.use_sep_geom:
                # 保护：特征维度不足51列时用零填充
                n_feats = features.shape[1]
                if n_feats >= 51:
                    sep_features = features[:, 39:51]
                else:
                    sep_features = torch.zeros((features.shape[0], 12), device=features.device)
            else:
                sep_features = torch.zeros((features.shape[0], 0), device=features.device)
            sign_input = torch.cat([sep_features, magnitude.detach() if self.detach_magnitude else magnitude], dim=-1)
            sign_prob = self.sign_net(sign_input)
        else:
            sign_prob = torch.ones_like(magnitude)

        wss = magnitude * (2.0 * sign_prob - 1.0)
        return wss, magnitude, sign_prob


class CoupledSingleHeadPredictor(nn.Module):
    """
    单头MLP：直接预测WSS（正负均可），无符号/幅值解耦
    用于 coupled_head / pure_mlp_xyz / no_geometry_mlp 对比实验
    """

    def __init__(self, input_dim: int, hidden_layers: list = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [input_dim, 256, 256, 256, 256, 1]
        else:
            hidden_layers = [input_dim] + hidden_layers + [1]
        self.net = WSSNetwork(hidden_layers)

    def forward(self, features, velocity_real=None, pressure_real=None):
        """标准接口，兼容 trainer/evaluator 调用"""
        return self.net(features)

    def forward_coupled(self, features, velocity_real=None, pressure_real=None):
        """兼容接口：返回 (wss, magnitude, sign_prob) — magnitude/sign_prob 为 dummy"""
        wss = self.net(features)
        magnitude = torch.abs(wss)
        sign_prob = (torch.sign(wss) + 1) / 2  # 从 wss 反推 sign_prob
        return wss, magnitude, sign_prob


def create_model(input_dim: int = 24) -> nn.Module:
    """模型工厂：根据 EXP_CONFIG 创建对应模型"""
    config = EXP_CONFIG

    # PointNet (vanilla) 分支 — 仅壁面点，O(N) 轻量级
    #   pointnet_xyz:  (x,y,z) only   → in_features=0
    #   pointnet:      (x,y,z,v,p)    → in_features=2
    if config.get('model_type') == 'pointnet':
        from pointnet_model import PointNet
        in_features = 0 if config['pointnet_input'] == 'coords_only' else 2
        model = PointNet(in_features=in_features, use_combined=False)
        model._model_type = 'pointnet'
        return model

    # PointNet 组合点云 (壁面+内部) — 与 PointNet++/DGCNN/GAT 同输入条件对比
    #   pointnet_combined_xyz:  (x,y,z,is_wall) only → in_features=1
    #   pointnet_combined:      (x,y,z,|v|,p,is_wall) → in_features=3
    if config.get('model_type') == 'pointnet_combined':
        from pointnet_model import PointNet
        in_features = 1 if config['pointnet_input'] == 'coords_only' else 3
        model = PointNet(in_features=in_features, use_combined=True)
        model._model_type = 'pointnet_combined'
        return model

    # PointNet++ 分支（支持组合点云：壁面+内部点）
    if config.get('model_type') == 'pointnetpp':
        from pointnetpp_model import PointNetPP
        in_features = 1 if config['pointnetpp_input'] == 'coords_only' else 3
        model = PointNetPP(in_features=in_features)
        model._model_type = 'pointnetpp'
        return model

    # DGCNN 分支（支持组合点云：壁面+内部点，通过 k-NN 图传播内部流场信息）
    if config.get('model_type') == 'dgcnn':
        from dgcnn_model import DGCNN
        # in_features = 非坐标特征数
        #   coords_only: [x,y,z] + [is_wall] → 1
        #   coords_features: [x,y,z] + [|v|, p, is_wall] → 3
        in_features = 1 if config['gnn_input'] == 'coords_only' else 3
        model = DGCNN(in_features=in_features)
        model._model_type = 'dgcnn'
        return model

    # GAT 分支（支持组合点云）
    if config.get('model_type') == 'gat':
        from gat_model import GAT
        in_features = 1 if config['gnn_input'] == 'coords_only' else 3
        model = GAT(in_features=in_features)
        model._model_type = 'gat'
        return model

    if config['decouple_head']:
        return DirectWSSPredictor(
            wss_layers=[24, 256, 256, 256, 256, 1],
            sign_layers=[13, 256, 128, 64, 1],
            use_sign_branch=config['use_sign_branch'],
            use_sep_geom=config['use_sep_geom'],
            detach_magnitude=config.get('detach_magnitude', True),
            use_velocity_pressure_only=config.get('use_velocity_pressure_only', False),
            use_geometry_features=config.get('use_geometry_features', True),
        )
    else:
        return CoupledSingleHeadPredictor(
            input_dim=input_dim,
            hidden_layers=[256, 256, 256, 256],
        )
