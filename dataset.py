"""数据集类：加载和预处理颈动脉血流数据"""

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis
from typing import Tuple, Dict
from sklearn.decomposition import PCA  # 仅用于局部邻域，不是全局
from config import DEVICE, DataConfig, FLUID_PARAMS, EXP_CONFIG
from data_splitter import AdaptiveWSSSplitter
import os
import pickle

class CarotidDataset:
    """
    颈动脉血流数据集
    数据格式: x, y, z, velocity, pressure, wss
    """
    
    def __init__(self, data_file: str, split_mode: str = 'train',
                 z_split_ratio: float = 0.5,
                 split_strategy: str = 'adaptive_wss_bidirectional'):
        """
        加载颈动脉数据
        
        Args:
            data_file: CSV文件路径
            split_mode: 'train' 或 'val'
            z_split_ratio: Z轴切分比例
            split_strategy: 切分策略
        """
        # 读取数据
        self.data = pd.read_csv(data_file)
        print(f"数据形状: {self.data.shape}")
        print(f"数据列: {self.data.columns.tolist()}")
        
        self.split_mode = split_mode
        
        # 提取坐标和物理量
        coords = self.data[['x', 'y', 'z']].values.astype(np.float32)
        self.velocity = self.data['velocity'].values.astype(np.float32).reshape(-1, 1)
        self.pressure = self.data['pressure'].values.astype(np.float32).reshape(-1, 1)
        self.wss = self.data['wss'].values.astype(np.float32).reshape(-1, 1)
        
        # 去量纲化
        self.features_characteristics = self._compute_characteristic_scales()
        self.velocity /= self.features_characteristics['velocity']
        self.pressure /= self.features_characteristics['pressure']
        self.wss /= self.features_characteristics['wss']
        
        # 数据切分
        coords, velocity_split, pressure_split, wss_split = self._split_data(
            coords, self.velocity, self.pressure, self.wss,
            split_strategy, z_split_ratio
        )
        
        # 识别壁面和内部点
        self._identify_wall_and_interior(coords, velocity_split, pressure_split, wss_split)
        
        # 计算几何特征
        self.compute_wall_distance()
        self.compute_wall_distance_and_normal(DataConfig.BOUNDARY_LAYER_THRESHOLD)
        self.compute_wss_physics()
        
        # 计算局部几何特征
        self.k = DataConfig.K_NEIGHBORS_WALL
        self.local_geom_features, self.wall_neighbor_indices = \
            self.compute_local_geometry(self.wall_coords)
        
        # 构建输入特征
        self.build_inputs()

        # ========== 新增：流动方向特征 ==========
        self.data_file = data_file
        
        # 1. 计算/加载局部流动方向
        self.compute_local_flow_direction(
            k_neighbors=10,
            data_file=data_file
        )
        
        # 2. 计算WSS-流动方向对齐
        self.compute_wss_direction_alignment()
        
        # 3. 计算衍生特征
        self.compute_flow_derived_features()
        
        # 4. 构建增强输入（39维）
        self.build_inputs_with_flow()
        # ==========================================
        
        # 标准化处理
        self._apply_scaling()
        
        # 流体参数
        self.mu = FLUID_PARAMS['mu']
        self.rho = FLUID_PARAMS['rho']

    def _compute_characteristic_scales(self) -> Dict[str, float]:
        """计算特征长度（无量纲化参数）"""
        print(f"\n{'='*60}")
        print("自动计算特征长度")
        
        characteristics = {}
        
        # 速度特征长度（使用95%分位数）
        U_max = float(np.max(self.velocity))
        U_95 = float(np.percentile(self.velocity, 95))
        U_char = U_95 if U_95 > 0 else U_max
        
        characteristics['velocity'] = U_char
        characteristics['velocity_max'] = U_max
        
        # 压力特征长度
        P_max = float(np.max(self.pressure))
        P_95 = float(np.percentile(self.pressure, 95))
        P_char = P_95 if P_95 > 0 else P_max
        
        characteristics['pressure'] = P_char
        characteristics['pressure_max'] = P_max
        
        # WSS特征长度
        tau_max = float(np.max(np.abs(self.wss[self.wss > 0])))
        tau_95 = float(np.percentile(np.abs(self.wss[self.wss > 0]), 95))
        
        characteristics['wss'] = tau_95
        characteristics['wss_max'] = tau_max
        
        print(f"  速度: U_95 = {U_char:.4f} m/s")
        print(f"  压力: P_95 = {P_char:.4f} Pa")
        print(f"  WSS:  τ_95 = {tau_95:.4f} Pa")
        print(f"{'='*60}\n")
        
        return characteristics

    def _split_data(self, coords, velocity, pressure, wss, 
                    split_strategy, z_split_ratio):
        """根据策略切分数据"""
        
        if split_strategy == 'adaptive_wss_bidirectional':
            splitter = AdaptiveWSSSplitter(coords=coords, wss_values=wss)
            best_result = splitter.find_optimal_split(prefer='middle', verbose=True)
            
            self.z_threshold = best_result['z_threshold']
            self.split_direction = best_result['direction']
            self.split_info = splitter.get_split_info()
            
            print(f"\n切分方案: 方向={self.split_direction}, Z={self.z_threshold:.4f}")
            
            # 应用切分
            if self.split_direction == AdaptiveWSSSplitter.DIRECTION_BOTTOM_UP:
                if self.split_mode == 'train':
                    mask = coords[:, 2] <= self.z_threshold
                    print(f"【训练集】Z <= {self.z_threshold:.4f} (下部)")
                else:
                    mask = coords[:, 2] > self.z_threshold
                    print(f"【验证集】Z > {self.z_threshold:.4f} (上部)")
            else:
                if self.split_mode == 'train':
                    mask = coords[:, 2] >= self.z_threshold
                    print(f"【训练集】Z >= {self.z_threshold:.4f} (上部)")
                else:
                    mask = coords[:, 2] < self.z_threshold
                    print(f"【验证集】Z < {self.z_threshold:.4f} (下部)")
        else:
            # 传统Z比例切分
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
            self.z_threshold = z_min + (z_max - z_min) * z_split_ratio
            
            if self.split_mode == 'train':
                mask = coords[:, 2] <= self.z_threshold
            else:
                mask = coords[:, 2] > self.z_threshold
        
        # 应用掩码
        coords = coords[mask]
        self.data = self.data[mask]
        
        print(f"原始数据: {len(mask)} 点 -> 当前split: {mask.sum()} 点")
        
        return coords, velocity[mask], pressure[mask], wss[mask]

    def _identify_wall_and_interior(self, coords, velocity, pressure, wss):
        """识别壁面点和内部点"""
        wall_mask = np.abs(velocity).flatten() < 1e-6
        interior_mask = ~wall_mask
        
        self.wall_coords = coords[wall_mask]
        self.interior_coords = coords[interior_mask]
        
        # 壁面点标签
        self.wall_wss = wss[wall_mask]
        self.wall_velocity = velocity[wall_mask]
        self.wall_pressure = pressure[wall_mask]
        
        # 内部点标签
        self.interior_velocity = velocity[interior_mask]
        self.interior_pressure = pressure[interior_mask]
        
        print(f"\n壁面点数: {len(self.wall_coords)}")
        print(f"内部点数: {len(self.interior_coords)}")

    def compute_wall_distance(self):
        """计算每个内部点到最近壁面的距离"""
        if len(self.wall_coords) > 0:
            wall_tree = cKDTree(self.wall_coords)
            interior_distances, _ = wall_tree.query(self.interior_coords)
            self.interior_wall_distance = interior_distances.reshape(-1, 1)
            self.wall_distance = np.zeros((len(self.wall_coords), 1), dtype=np.float32)
        else:
            self.interior_wall_distance = np.zeros((len(self.interior_coords), 1), dtype=np.float32)
            self.wall_distance = np.zeros((len(self.wall_coords), 1), dtype=np.float32)
        
        print(f"  边界层距离范围: [{self.interior_wall_distance.min():.4f}, "
              f"{self.interior_wall_distance.max():.4f}] mm")

    def compute_wall_distance_and_normal(self, boundary_layer_threshold: float = 3.0,
                                          n_neighbors: int = 3):
        """计算壁面距离和法向量"""
        if len(self.wall_coords) == 0:
            raise ValueError("壁面点为空")
        
        self.n_neighbors = n_neighbors
        
        # 内部点的壁面距离和法向
        wall_tree = cKDTree(self.wall_coords)
        interior_distances, nearest_idx = wall_tree.query(self.interior_coords, k=1)
        
        self.interior_wall_distance = interior_distances.reshape(-1, 1).astype(np.float32)
        self.nearest_wall_idx = nearest_idx
        
        nearest_wall_points = self.wall_coords[nearest_idx]
        direction_to_wall = nearest_wall_points - self.interior_coords
        norms = np.linalg.norm(direction_to_wall, axis=1, keepdims=True) + 1e-10
        self.interior_wall_normal = (direction_to_wall / norms).astype(np.float32)
        
        # 壁面点法向 - 从邻近内部点估计
        interior_tree = cKDTree(self.interior_coords)
        self.interior_tree = interior_tree
        dists_to_interior, nearest_interior_idx = interior_tree.query(
            self.wall_coords, k=n_neighbors
        )
        
        self.wall_nearest_interior_idx = nearest_interior_idx
        self.wall_nearest_interior_dist = dists_to_interior.astype(np.float32)
        
        # 获取最近内部点的速度和坐标
        self.wall_nearest_interior_velocity = np.zeros((len(self.wall_coords), n_neighbors), 
                                                        dtype=np.float32)
        self.wall_nearest_interior_coords = np.zeros((len(self.wall_coords), n_neighbors, 3),
                                                      dtype=np.float32)
        
        for i in range(n_neighbors):
            idx = nearest_interior_idx[:, i]
            self.wall_nearest_interior_velocity[:, i] = self.interior_velocity[idx, 0]
            self.wall_nearest_interior_coords[:, i, :] = self.interior_coords[idx]
        
        # 计算壁面点法向
        nearest_interior_points = self.interior_coords[nearest_interior_idx[:, 0]]
        direction_from_interior = self.wall_coords - nearest_interior_points
        norms_wall = np.linalg.norm(direction_from_interior, axis=1, keepdims=True) + 1e-10
        self.wall_normal = (direction_from_interior / norms_wall).astype(np.float32)
        
        # 边界层标记
        self.is_boundary_layer = self.interior_wall_distance < boundary_layer_threshold
        
        print(f"  边界层内点数: {self.is_boundary_layer.sum()} "
              f"({100*self.is_boundary_layer.sum()/len(self.interior_coords):.1f}%)")

    def compute_separation_geometry_features(self, radius: float = 3.0, k: int = 30):
        """
        计算用于回流/分离检测的纯几何形态学特征（不依赖速度方向符号）
        输出: (n_wall, 12)
        优化版：单次批量KDTree查询 + batch线性代数，消除Python逐点循环开销
        """
        n_wall = len(self.wall_coords)
        sep_features = np.zeros((n_wall, 12), dtype=np.float32)

        if not hasattr(self, 'interior_tree'):
            self.interior_tree = cKDTree(self.interior_coords)
        if not hasattr(self, 'wall_tree'):
            self.wall_tree = cKDTree(self.wall_coords)

        # ===== 批量查询：所有壁面点一次性完成 =====
        idists, iidx = self.interior_tree.query(self.wall_coords, k=k, distance_upper_bound=radius)
        imask = idists < radius  # (n_wall, k)
        n_ivalid = imask.sum(axis=1)  # (n_wall,)
        has_enough = n_ivalid >= 5

        # 批量提取邻居速度和坐标（用NaN填充无效位，后续用mask过滤）
        vel_all = np.take(self.interior_velocity.flatten(), np.clip(iidx, 0, len(self.interior_velocity) - 1))
        vel_all[~imask] = np.nan  # (n_wall, k)

        # ---- 1. 速度剖面非线性度 (3维) - batch polyfit ----
        for i in np.where(has_enough)[0]:
            d = idists[i, imask[i]]
            v = vel_all[i, imask[i]]
            d_max = d.max()
            if d_max < 1e-10:
                continue
            d_norm = d / d_max
            v_max = v.max()
            if v_max < 1e-10:
                continue
            v_norm = v / v_max
            try:
                # 二次多项式拟合: v = a*d² + b*d + c
                A = np.column_stack([d_norm**2, d_norm, np.ones_like(d_norm)])
                coeffs, _res, _rank, _s = np.linalg.lstsq(A, v_norm, rcond=None)
                a, b, c = coeffs
                v_pred_lin = a * d_norm + c
                linearity = 1 - np.mean(np.abs(v_norm - v_pred_lin)) / (np.std(v_norm) + 1e-8)
                sep_features[i, 0] = np.clip(linearity, 0, 1)
                sep_features[i, 1] = abs(b)
                sep_features[i, 2] = b
            except Exception:
                pass

        # ---- 2. 近壁点云各向异性 (2维) ----
        for i in np.where(n_ivalid >= 6)[0]:
            pts = self.interior_coords[iidx[i, imask[i]]] - self.wall_coords[i]
            cov = np.cov(pts.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals.sort()
            eigvals = eigvals[::-1]
            s = eigvals.sum()
            if s < 1e-10 or eigvals[0] < 1e-10:
                continue
            eigvals = eigvals / s
            sep_features[i, 3] = 1 - (eigvals[1] + eigvals[2]) / (2 * eigvals[0])
            sep_features[i, 4] = (eigvals[1] - eigvals[2]) / (eigvals[1] + 1e-8)

        # ---- 3. 速度分布统计异常 (3维) ----
        for i in np.where(has_enough)[0]:
            v = vel_all[i, imask[i]]
            if len(v) < 5 or np.std(v) < 1e-10:
                continue
            sep_features[i, 5] = skew(v)
            sep_features[i, 6] = kurtosis(v)
            if len(v) > 2:
                sep_features[i, 7] = np.corrcoef(v, idists[i, imask[i]])[0, 1]

        # ===== 壁面曲率：批量查询 =====
        wdists_all, widx_all = self.wall_tree.query(self.wall_coords, k=min(15, n_wall))
        wmask_all = wdists_all < radius * 2.0
        n_wvalid = wmask_all.sum(axis=1)

        # ---- 4. 壁面局部曲率Hessian (3维) ----
        for i in np.where(n_wvalid >= 6)[0]:
            nb = self.wall_coords[widx_all[i, wmask_all[i]]]
            centered = nb - self.wall_coords[i]
            cov = np.cov(centered.T)
            eigvals = np.sort(np.linalg.eigvalsh(cov))
            if len(eigvals) >= 2:
                sep_features[i, 8] = eigvals[0] * eigvals[1]
                mean_curv = (eigvals[0] + eigvals[1]) / 2.0
                sep_features[i, 9] = mean_curv
                sep_features[i, 10] = np.sign(mean_curv) * np.log1p(abs(mean_curv))

        # ---- 5. 局部截面扩张比 (1维) ----
        wext_dists, wext_idx = self.wall_tree.query(self.wall_coords, k=min(50, n_wall),
                                                     distance_upper_bound=radius * 2)
        wext_mask = wext_dists < radius * 2
        n_wext_valid = wext_mask.sum(axis=1)

        for i in np.where(n_wext_valid >= 10)[0]:
            nb = self.wall_coords[wext_idx[i, wext_mask[i]]]
            centered = nb - self.wall_coords[i]
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            tube_axis = eigvecs[:, -1]
            up = self.wall_coords[i] - tube_axis * 3.0
            down = self.wall_coords[i] + tube_axis * 3.0
            up_d, _ = self.wall_tree.query(up, k=30, distance_upper_bound=radius * 2)
            down_d, _ = self.wall_tree.query(down, k=30, distance_upper_bound=radius * 2)
            up_c = (up_d < radius * 2).sum()
            down_c = (down_d < radius * 2).sum()
            sep_features[i, 11] = up_c / (down_c + 1e-8)

        self.wall_separation_geometry_features = sep_features
        print(f"\n  分离几何特征: {sep_features.shape[1]}维")
        print(f"    - 剖面非线性: 3维 | 点云各向异性: 2维 | 速度分布异常: 3维")
        print(f"    - 壁面Hessian: 3维 | 局部扩张比: 1维")
        return sep_features

    def compute_local_geometry(self, wall_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算壁面局部几何特征（曲率等）"""
        n_wall = len(wall_coords)
        tree = cKDTree(wall_coords)
        distances, indices = tree.query(wall_coords, k=self.k + 1)
        
        neighbor_indices = indices[:, 1:]  # 去掉自身
        neighbor_distances = distances[:, 1:]
        
        # 计算特征
        features = {}
        features['density'] = neighbor_distances.mean(axis=1, keepdims=True)
        
        # 协方差分析
        shape_indices = np.zeros((n_wall, 3), dtype=np.float32)
        
        for i in range(n_wall):
            neighbors = wall_coords[neighbor_indices[i]]
            centered = neighbors - wall_coords[i]
            cov = centered.T @ centered / self.k
            
            eigenvalues, _ = np.linalg.eigh(cov)
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            
            if eigenvalues[2] > 1e-6:
                e0, e1, e2 = eigenvalues / eigenvalues[2]
                curvedness = np.sqrt(e0**2 + e1**2 + e2**2)
                shape_index = (2/np.pi) * np.arctan2(e1 - e0, e2 - e1)
                shape_indices[i] = [curvedness, shape_index, e0/e2]
        
        features['curvedness'] = shape_indices[:, 0:1]
        features['shape_index'] = shape_indices[:, 1:2]
        features['anisotropy'] = shape_indices[:, 2:3]
        
        feature_matrix = np.hstack([
            features['density'],
            features['curvedness'],
            features['shape_index'],
            features['anisotropy'],
        ])
        self.wall_tree = tree

        return feature_matrix, neighbor_indices

    def compute_wss_physics(self, mu: float = 0.00345):
        """从标量速度数据计算物理WSS = μ · ∂u/∂n"""
        print(f"\n  ===== 物理计算 WSS =====")
        
        n_wall = len(self.wall_coords)
        self.wss_physics = np.zeros((n_wall, 1), dtype=np.float32)
        
        # 建立反向映射
        wall_to_interiors = [[] for _ in range(n_wall)]
        for interior_idx, wall_idx in enumerate(self.nearest_wall_idx):
            wall_to_interiors[wall_idx].append(interior_idx)
        
        for wall_idx in range(n_wall):
            fans = wall_to_interiors[wall_idx]
            
            if len(fans) == 0:
                # 孤儿点
                dists = np.linalg.norm(self.interior_coords - self.wall_coords[wall_idx], axis=1)
                nearest_int = np.argmin(dists)
                d = dists[nearest_int]
                v = self.interior_velocity[nearest_int, 0]
                if d > 0:
                    self.wss_physics[wall_idx, 0] = mu * (v / d)
            else:
                # 使用粉丝点平均
                gradients = []
                for int_idx in fans:
                    d = self.interior_wall_distance[int_idx, 0]
                    v = self.interior_velocity[int_idx, 0]
                    if d > 0.001:
                        gradients.append(v / d)
                
                if gradients:
                    self.wss_physics[wall_idx, 0] = mu * np.mean(gradients)
        
        # 验证
        correlation = np.corrcoef(
            np.abs(self.wss_physics.flatten()), 
            np.abs(self.wall_wss.flatten())
        )[0, 1]
        
        print(f"  物理WSS vs CFD标签 相关性: {correlation:.4f}")
        print(f"  ==========================\n")
        
        return self.wss_physics

    def build_inputs(self):
        """构建输入特征：基础11维 + 局部几何4维 = 15维"""
        self.wall_velocity_directions = self._compute_velocity_directions()
        
        # 壁面点特征
        wall_features_list = [
            self.wall_distance,           # 1
            self.wall_normal,             # 3
        ]
        
        # 3个最近邻的速度和距离
        for i in range(self.n_neighbors):
            wall_features_list.append(self.wall_nearest_interior_velocity[:, i:i+1])
            wall_features_list.append(self.wall_nearest_interior_dist[:, i:i+1])
        
        wall_features_list.append(self.wss_physics)  # 1: 物理WSS
        
        # 新增：3个最近邻的速度方向（单位矢量）
        for i in range(self.n_neighbors):
            wall_features_list.append(self.wall_velocity_directions[:, i*3:(i+1)*3])   # 3*3=9维

        # 基础特征 (11维) + 速度方向 (9维) = 20维 + 局部几何 (4维) = 24维
        wall_base = np.hstack(wall_features_list)
        self.wall_features = np.hstack([wall_base, self.local_geom_features])
        
        print(f"\n  输入特征: {self.wall_features.shape[1]}维")
        print(f"    - 基础特征: 11维")
        print(f"    - 速度方向: 9维 (3点 × 3方向)")
        print(f"    - 局部几何: 4维")
        
        # 内部点特征（占位，保持维度一致）
        interior_features_list = [
            self.interior_wall_distance,
            self.interior_wall_normal,
        ]
        
        for i in range(self.n_neighbors):
            interior_features_list.append(self.interior_velocity)
            interior_features_list.append(np.zeros((len(self.interior_coords), 1)))        

        interior_features_list.append(np.zeros((len(self.interior_coords), 1))) # wss_physics占位
        # 内部点速度方向占位
        for i in range(self.n_neighbors):
            interior_features_list.append(np.zeros((len(self.interior_coords), 3)))
        interior_features_list.append(np.zeros((len(self.interior_coords), 4), dtype=np.float32)) # 局部几何占位       

        self.interior_features = np.hstack(interior_features_list)
        
        # 合并
        self.all_features = np.vstack([self.wall_features, self.interior_features])
        
        # 标准化后的引用
        self.wall_features_scaled = self.wall_features
        self.interior_features_scaled = self.interior_features

    def build_inputs_with_flow(self):
        """
        构建包含流动方向特征的完整输入
        """
        # 原有特征
        base_features = self.wall_features  # 24维
        flow_features = self.wall_flow_derived_features  # 15维
        
        # 新增流动衍生特征
        if not hasattr(self, 'wall_flow_derived_features'):
            self.compute_flow_derived_features()
        
        flow_features = self.wall_flow_derived_features  # 15维
        if not hasattr(self, 'wall_separation_geometry_features'):
            self.compute_separation_geometry_features(radius=3.0, k=30)
        sep_features = self.wall_separation_geometry_features  # 12维
        # 合并
        self.wall_features_enhanced = np.hstack([base_features, flow_features, sep_features])
        
        print(f"\n  增强后输入特征: {self.wall_features_enhanced.shape[1]}维")
        print(f"    - 基础特征: 24维")
        print(f"    - 流动衍生: 15维")
        print(f"    - 分离几何: 12维")

        # 更新所有引用（关键！）
        self.wall_features = self.wall_features_enhanced
        self.wall_features_scaled = self.wall_features_enhanced
        # ====== 关键修复：interior_features 也需要39维 ======
        n_interior = len(self.interior_coords)
        
        # interior 的流动方向特征占位（15维零填充）
        interior_flow_placeholder = np.zeros((n_interior, 15), dtype=np.float32)
        interior_sep_placeholder = np.zeros((n_interior, 12), dtype=np.float32)

        # 扩展 interior_features 到39维
        self.interior_features_enhanced = np.hstack([
            self.interior_features,           # 24维原始
            interior_flow_placeholder,          # 15维占位
            interior_sep_placeholder            # 12维占位
        ])
        
        # 更新 interior 引用
        self.interior_features = self.interior_features_enhanced
        self.interior_features_scaled = self.interior_features_enhanced
        
        # 重新合并 all_features
        self.all_features = np.vstack([
            self.wall_features_enhanced, 
            self.interior_features_enhanced
        ])
        return self.wall_features_enhanced

    def _compute_velocity_directions(self) -> np.ndarray:
        """
        计算壁面点到其3个最近内部速度点的单位方向矢量
        
        Returns:
            directions: (n_wall, 9) 数组，每行 [dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3]
                    每个 (dx,dy,dz) 是从壁面点指向速度点的单位矢量
        """
        n_wall = len(self.wall_coords)
        directions = np.zeros((n_wall, 9), dtype=np.float32)
        
        for i in range(self.n_neighbors):
            # 获取第i个最近内部点
            nearest_coords = self.wall_nearest_interior_coords[:, i, :]  # (n_wall, 3)
            
            # 从壁面点指向内部点的矢量
            vec = nearest_coords - self.wall_coords  # (n_wall, 3)
            
            # 归一化为单位矢量
            norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10
            unit_vec = vec / norms
            
            # 存入对应位置. d单位质量模长太小，不利于学习，用原变量试试
            directions[:, i*3:(i+1)*3] = vec
        
        # 验证
        print(f"\n  速度方向特征统计:")
        for i in range(self.n_neighbors):
            d = directions[:, i*3:(i+1)*3]
            print(f"    V{i+1}方向: 模长=[{np.linalg.norm(d, axis=1).min():.3f}, "
                f"{np.linalg.norm(d, axis=1).max():.3f}] (应≈1.0)")
        
        return directions

    def _apply_scaling(self):
        """应用标准化（此处数据已去量纲，无需额外标准化）"""
        self.wall_wss_scaled = self.wall_wss
        self.wall_velocity_scaled = self.wall_velocity
        self.wall_pressure_scaled = self.wall_pressure
        self.interior_velocity_scaled = self.interior_velocity
        self.interior_pressure_scaled = self.interior_pressure

    def _get_flow_cache_path(self, data_file: str, split_mode: str) -> str:
        """生成流动方向缓存文件路径"""
        # 基于原始数据文件路径生成缓存路径
        data_dir = os.path.dirname(data_file)
        base_name = os.path.splitext(os.path.basename(data_file))[0]
        cache_file = f"{base_name}_{split_mode}_flow_direction.pkl"
        return os.path.join(data_dir, cache_file)
    
    def save_flow_direction(self, data_file: str, split_mode: str):
        """
        保存计算好的流动方向结果到数据文件夹
        
        Args:
            data_file: 原始CSV文件路径（用于确定保存位置）
            split_mode: 'train' 或 'val'
        """
        cache_path = self._get_flow_cache_path(data_file, split_mode)
        
        flow_data = {
            'wall_flow_direction': getattr(self, 'wall_flow_direction', None),
            'wall_flow_quality': getattr(self, 'wall_flow_quality', None),
            'interior_flow_direction': getattr(self, 'interior_flow_direction', None),
            'interior_flow_quality': getattr(self, 'interior_flow_quality', None),
            'wss_flow_alignment': getattr(self, 'wss_flow_alignment', None),
            'wss_sign': getattr(self, 'wss_sign', None),
            # 保存元信息用于验证
            'n_wall': len(self.wall_coords) if hasattr(self, 'wall_coords') else 0,
            'n_interior': len(self.interior_coords) if hasattr(self, 'interior_coords') else 0,
            'data_file': data_file,
            'split_mode': split_mode,
            'compute_time': pd.Timestamp.now().isoformat()
        }
        
        # 检查是否有有效数据
        if flow_data['wall_flow_direction'] is None:
            print(f"⚠️ 无流动方向数据可保存")
            return
        
        with open(cache_path, 'wb') as f:
            pickle.dump(flow_data, f)
        
        print(f"✓ 流动方向数据已保存: {cache_path}")
        print(f"  文件大小: {os.path.getsize(cache_path) / 1024:.1f} KB")


    def load_flow_direction(self, data_file: str, split_mode: str) -> bool:
        """
        从数据文件夹加载流动方向结果
        
        Args:
            data_file: 原始CSV文件路径
            split_mode: 'train' 或 'val'
        
        Returns:
            bool: 是否成功加载
        """
        cache_path = self._get_flow_cache_path(data_file, split_mode)
        
        if not os.path.exists(cache_path):
            print(f"  未找到缓存文件: {cache_path}")
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                flow_data = pickle.load(f)
            
            # 验证数据一致性
            current_n_wall = len(self.wall_coords) if hasattr(self, 'wall_coords') else 0
            cached_n_wall = flow_data.get('n_wall', 0)
            
            if current_n_wall != cached_n_wall and current_n_wall > 0:
                print(f"⚠️ 缓存数据点数不匹配: 当前={current_n_wall}, 缓存={cached_n_wall}")
                print(f"  重新计算流动方向...")
                return False
            
            # 恢复数据
            if flow_data['wall_flow_direction'] is not None:
                self.wall_flow_direction = flow_data['wall_flow_direction']
            if flow_data['wall_flow_quality'] is not None:
                self.wall_flow_quality = flow_data['wall_flow_quality']
            if flow_data['interior_flow_direction'] is not None:
                self.interior_flow_direction = flow_data['interior_flow_direction']
            if flow_data['interior_flow_quality'] is not None:
                self.interior_flow_quality = flow_data['interior_flow_quality']
            if flow_data['wss_flow_alignment'] is not None:
                self.wss_flow_alignment = flow_data['wss_flow_alignment']
            if flow_data['wss_sign'] is not None:
                self.wss_sign = flow_data['wss_sign']
            
            print(f"✓ 已加载流动方向缓存: {cache_path}")
            print(f"  计算时间: {flow_data.get('compute_time', 'unknown')}")
            return True
            
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
            return False


    def compute_local_flow_direction(self, k_neighbors: int = 10,
                                    boundary_layer_only: bool = True,
                                  data_file: str = None,
                                  force_recompute: bool = False):
        """
        计算局部流动方向（不依赖全局PCA）
        优化版：单次批量KDTree查询，消除逐点循环
        """

        # 尝试加载缓存
        if not force_recompute and data_file is not None:
            split_mode = getattr(self, 'split_mode', 'unknown')
            if self.load_flow_direction(data_file, split_mode):
                return self.wall_flow_direction, self.wall_flow_quality

        print(f"\n{'='*60}")
        print("计算局部流动方向（无全局PCA）")
        print(f"{'='*60}")

        coords = np.vstack([self.wall_coords, self.interior_coords])
        velocity = np.vstack([self.wall_velocity, self.interior_velocity])

        n_points = len(coords)
        n_wall = len(self.wall_coords)
        flow_direction = np.zeros((n_points, 3), dtype=np.float32)
        flow_quality = np.zeros(n_points, dtype=np.float32)

        tree = cKDTree(coords)

        if boundary_layer_only:
            bl_mask = np.zeros(n_points, dtype=bool)
            bl_mask[:n_wall] = True
            bl_mask[n_wall:] = self.interior_wall_distance.flatten() < 5.0
            compute_indices = np.where(bl_mask)[0]
        else:
            compute_indices = np.arange(n_points)

        print(f"  计算点数: {len(compute_indices)}/{n_points}")

        interior_tree = getattr(self, 'interior_tree', None)
        if interior_tree is None and len(self.interior_coords) > 0:
            interior_tree = cKDTree(self.interior_coords)

        # ===== 批量查询：所有计算点一次性查询k近邻 =====
        compute_coords = coords[compute_indices]
        k_actual = min(k_neighbors, n_points)
        batch_dists, batch_neighbors = tree.query(compute_coords, k=k_actual)

        degenerate_count = 0

        for batch_i, global_idx in enumerate(compute_indices):
            neighbors = batch_neighbors[batch_i]
            k = len(neighbors)
            neighbor_coords = coords[neighbors]
            centered = neighbor_coords - coords[global_idx]
            vel_proxy = centered  # (k, 3)

            vel_std = np.std(vel_proxy, axis=0)
            if np.max(vel_std) < 1e-10 or k < 3:
                degenerate_count += 1
                coord_cov = np.cov(centered.T)
                if coord_cov.ndim >= 2 and coord_cov.shape[0] >= 2:
                    try:
                        ceigvals, ceigvecs = np.linalg.eigh(coord_cov)
                        local_dir = ceigvecs[:, np.argmax(ceigvals)]
                    except Exception:
                        if global_idx < n_wall and interior_tree is not None:
                            _, nearest_int = interior_tree.query(coords[global_idx], k=1)
                            nearest_int = nearest_int.item() if hasattr(nearest_int, 'item') else nearest_int
                            direction = self.interior_coords[nearest_int] - coords[global_idx]
                            norm = np.linalg.norm(direction)
                            local_dir = direction / norm if norm > 1e-6 else np.array([1, 0, 0], dtype=np.float32)
                        else:
                            local_dir = np.array([1, 0, 0], dtype=np.float32)
                else:
                    local_dir = np.array([1, 0, 0], dtype=np.float32)
                flow_quality[global_idx] = 0.1
            else:
                vel_cov = np.cov(vel_proxy.T)
                if vel_cov.ndim < 2 or vel_cov.shape[0] < 2:
                    mean_vel = np.mean(vel_proxy, axis=0)
                    norm = np.linalg.norm(mean_vel)
                    local_dir = mean_vel / norm if norm > 1e-6 else np.array([1, 0, 0], dtype=np.float32)
                    flow_quality[global_idx] = 0.3
                else:
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(vel_cov)
                        max_eig_idx = np.argmax(eigenvalues)
                        local_dir = eigenvectors[:, max_eig_idx]
                        flow_quality[global_idx] = eigenvalues[max_eig_idx] / (eigenvalues.sum() + 1e-10)
                    except np.linalg.LinAlgError:
                        mean_vel = np.mean(vel_proxy, axis=0)
                        norm = np.linalg.norm(mean_vel)
                        local_dir = mean_vel / norm if norm > 1e-6 else np.array([1, 0, 0], dtype=np.float32)
                        flow_quality[global_idx] = 0.2

            # 方向符号对齐
            current_vel = velocity[global_idx].flatten()
            vel_norm_val = np.linalg.norm(current_vel)

            if current_vel.shape[0] == 3 and vel_norm_val > 1e-6:
                if np.dot(local_dir, current_vel) < 0:
                    local_dir = -local_dir
            elif current_vel.shape[0] == 1 and vel_norm_val > 1e-6:
                coord_diff = neighbor_coords[-1] - neighbor_coords[0]
                diff_norm = np.linalg.norm(coord_diff)
                if diff_norm > 1e-6:
                    coord_dir = coord_diff / diff_norm
                    if np.dot(local_dir, coord_dir) < 0:
                        local_dir = -local_dir
                if global_idx < n_wall and interior_tree is not None:
                    _, nearest_int_idx = interior_tree.query(coords[global_idx], k=1)
                    nearest_int_idx = nearest_int_idx.item() if hasattr(nearest_int_idx, 'item') else nearest_int_idx
                    direction_to_interior = self.interior_coords[nearest_int_idx] - coords[global_idx]
                    dir_norm = np.linalg.norm(direction_to_interior)
                    if dir_norm > 1e-6:
                        direction_to_interior = direction_to_interior / dir_norm
                        if np.dot(local_dir, direction_to_interior) < 0:
                            local_dir = -local_dir

            flow_direction[global_idx] = local_dir

        if degenerate_count > 0:
            print(f"  退化点数: {degenerate_count}/{len(compute_indices)}")

        self.wall_flow_direction = flow_direction[:n_wall]
        self.interior_flow_direction = flow_direction[n_wall:]
        self.wall_flow_quality = flow_quality[:n_wall]
        self.interior_flow_quality = flow_quality[n_wall:]

        norms = np.linalg.norm(self.wall_flow_direction, axis=1)
        zero_mask = norms < 1e-6
        if zero_mask.sum() > 0:
            print(f"  ⚠️ {zero_mask.sum()} 个壁面点方向模长为0，设为默认方向")
            self.wall_flow_direction[zero_mask] = np.array([1, 0, 0])

        print(f"  壁面流动方向质量: {self.wall_flow_quality.mean():.3f} ± {self.wall_flow_quality.std():.3f}")
        print(f"  方向模长范围: [{np.linalg.norm(self.wall_flow_direction, axis=1).min():.3f}, "
            f"{np.linalg.norm(self.wall_flow_direction, axis=1).max():.3f}]")

        if data_file is not None:
            split_mode = getattr(self, 'split_mode', 'unknown')
            self.save_flow_direction(data_file, split_mode)

        return self.wall_flow_direction, self.wall_flow_quality


    # 在 compute_local_flow_direction 中扩展
    def compute_velocity_gradient_features(self, k_neighbors=10):
        """
        计算局部速度梯度特征（不依赖法向量）
        """
        coords = np.vstack([self.wall_coords, self.interior_coords])
        velocity = np.vstack([self.wall_velocity, self.interior_velocity])
        n_wall = len(self.wall_coords)
        
        tree = cKDTree(coords)
        
        # 只计算壁面点
        grad_features = np.zeros((n_wall, 6), dtype=np.float32)
        
        for i in range(n_wall):
            # 查询k近邻（包含壁面点和内部点）
            distances, neighbors = tree.query(self.wall_coords[i], k=k_neighbors)
            
            # 邻域坐标和速度
            neighbor_coords = coords[neighbors]
            neighbor_vel = velocity[neighbors].flatten()
            
            # 坐标差分矩阵 (k-1, 3)
            dx = neighbor_coords[1:] - neighbor_coords[0]
            dv = neighbor_vel[1:] - neighbor_vel[0]
            
            if len(dx) > 3:
                # 最小二乘估计速度梯度: dv = dx · grad_u
                # grad_u 是 (3,) 的梯度向量
                grad_u, residuals, rank, s = np.linalg.lstsq(dx, dv, rcond=None)
                
                # 特征1: 梯度模长 |∇u|
                grad_magnitude = np.linalg.norm(grad_u)
                
                # 特征2: 梯度方向与流动方向夹角
                flow_dir = self.wall_flow_direction[i]
                angle_grad_flow = np.arccos(np.clip(
                    np.dot(grad_u, flow_dir) / (grad_magnitude + 1e-10), -1, 1
                )) * 180 / np.pi
                
                # 特征3: 梯度方向与最近内部点方向夹角
                nearest_int_dir = self.interior_coords[self.nearest_wall_idx[i]] - self.wall_coords[i]
                nearest_int_dir = nearest_int_dir / (np.linalg.norm(nearest_int_dir) + 1e-10)
                angle_grad_normal = np.arccos(np.clip(
                    np.dot(grad_u, nearest_int_dir) / (grad_magnitude + 1e-10), -1, 1
                )) * 180 / np.pi
                
                # 特征4: 速度梯度散度（近似）∂u/∂x + ∂u/∂y + ∂u/∂z
                # 这里用标量速度的梯度模长近似
                
                # 特征5: 局部速度变化率 std(dv)/mean(|v|)
                vel_variation = np.std(dv) / (np.mean(np.abs(neighbor_vel)) + 1e-10)
                
                # 特征6: 近壁速度衰减率 (v_center - v_wall) / distance
                # 用最近内部点速度近似
                v_wall = 0  # 壁面无滑移
                v_near = neighbor_vel[1] if len(neighbor_vel) > 1 else 0
                d_near = distances[1] if len(distances) > 1 else 1e-3
                decay_rate = (v_near - v_wall) / d_near
                
                grad_features[i] = [
                    grad_magnitude,
                    angle_grad_flow,
                    angle_grad_normal,
                    vel_variation,
                    decay_rate,
                    residuals[0] if len(residuals) > 0 else 0
                ]
        
        self.wall_grad_features = grad_features
        return grad_features

    def compute_velocity_structure_features(self):
        """
        计算速度结构特征（利用已有的n1,n2,n3方向信息）
        """
        n_wall = len(self.wall_coords)
        
        # 已有特征：v1,v2,v3速度和d1,d2,d3距离
        # 已有特征：n1,n2,n3方向（从壁面指向内部点的矢量）
        
        # 新特征1: 速度方向与局部流动方向的一致性
        v_align = np.zeros((n_wall, 3), dtype=np.float32)
        for i in range(3):
            # n_i 是方向向量（已归一化或原始坐标差）
            ni = self.wall_velocity_directions[:, i*3:(i+1)*3]  # (n_wall, 3)
            # 归一化
            ni_norm = np.linalg.norm(ni, axis=1, keepdims=True) + 1e-10
            ni_unit = ni / ni_norm
            
            # 与流动方向的点积
            v_align[:, i] = np.sum(ni_unit * self.wall_flow_direction, axis=1)
        
        # 新特征2: 速度衰减的非线性度
        # 泊肃叶流: v ~ d (线性), 实际可能非线性
        v1 = self.wall_features[:, 4]  # 根据build_inputs的索引
        v2 = self.wall_features[:, 6]
        v3 = self.wall_features[:, 8]
        d1 = self.wall_features[:, 5]
        d2 = self.wall_features[:, 7]
        d3 = self.wall_features[:, 9]
        
        # 速度-距离的线性度（泊肃叶流应为线性）
        linearity = np.zeros(n_wall, dtype=np.float32)
        for i in range(n_wall):
            # 拟合 v = a*d + b
            d_points = np.array([d1[i], d2[i], d3[i]])
            v_points = np.array([v1[i], v2[i], v3[i]])
            if np.std(d_points) > 1e-6:
                a, b = np.polyfit(d_points, v_points, 1)
                v_pred = a * d_points + b
                linearity[i] = 1 - np.mean(np.abs(v_points - v_pred)) / (np.mean(np.abs(v_points)) + 1e-10)
        
        # 新特征3: 速度梯度方向一致性（判断分离/附着）
        # 如果v1,v2,v3方向与flow_dir的夹角变化大 → 流动复杂
        angle_variation = np.std(v_align, axis=1)
        
        # 新特征4: 近壁速度梯度符号（判断WSS符号的关键）
        # WSS = μ * ∂v/∂n, 如果v随距离增加 → WSS>0, 减小 → WSS<0
        # 用(v1-v2)/(d1-d2)近似
        v_gradient = (v1 - v2) / (d1 - d2 + 1e-10)
        
        features = np.hstack([
            v_align,           # 3维: v1,v2,v3与flow_dir对齐
            linearity.reshape(-1, 1),      # 1维: 速度-距离线性度
            angle_variation.reshape(-1, 1), # 1维: 方向变化率
            v_gradient.reshape(-1, 1),      # 1维: 近壁速度梯度
        ])
        
        self.wall_velocity_structure_features = features
        return features
    

    def compute_flow_derived_features(self, k_neighbors: int = 10):
        """
        计算所有流动方向衍生特征（作为神经网络输入）
        
        不依赖壁面法向量，使用最近内部点方向作为代理
        """
        print(f"\n{'='*60}")
        print("计算流动方向衍生特征")
        print(f"{'='*60}")
        
        n_wall = len(self.wall_coords)
        
        # 1. 确保流动方向已计算
        if not hasattr(self, 'wall_flow_direction'):
            self.compute_local_flow_direction(k_neighbors=k_neighbors)
        
        # 2. 计算最近内部点方向（作为法向代理）
        nearest_int_dir = np.zeros((n_wall, 3), dtype=np.float32)
        for i in range(n_wall):
            if hasattr(self, 'nearest_wall_idx') and i < len(self.nearest_wall_idx):
                int_idx = self.nearest_wall_idx[i]
                if int_idx < len(self.interior_coords):
                    direction = self.interior_coords[int_idx] - self.wall_coords[i]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        nearest_int_dir[i] = direction / norm
        
        # 3. 计算alignment（使用最近内部点方向代理法向）
        alignment = np.sum(self.wall_flow_direction * nearest_int_dir, axis=1)
        
        # 4. 提取已有速度特征
        v1 = self.wall_features[:, 4]
        v2 = self.wall_features[:, 6]
        v3 = self.wall_features[:, 8]
        d1 = self.wall_features[:, 5]
        d2 = self.wall_features[:, 7]
        d3 = self.wall_features[:, 9]
        
        # 5. 速度方向与流动方向对齐（利用已有的n1,n2,n3）
        v_align = np.zeros((n_wall, 3), dtype=np.float32)
        for i in range(3):
            ni = self.wall_velocity_directions[:, i*3:(i+1)*3]
            ni_norm = np.linalg.norm(ni, axis=1, keepdims=True) + 1e-10
            ni_unit = ni / ni_norm
            v_align[:, i] = np.sum(ni_unit * self.wall_flow_direction, axis=1)
        
        # 6. 速度-距离线性度（泊肃叶偏离度）- 向量化batch polyfit
        d_points = np.column_stack([d1, d2, d3])  # (n_wall, 3)
        v_points = np.column_stack([v1, v2, v3])  # (n_wall, 3)
        d_std = np.std(d_points, axis=1)
        v_mean_abs = np.mean(np.abs(v_points), axis=1)

        linearity = np.zeros(n_wall, dtype=np.float32)
        valid = (d_std > 1e-6) & (v_mean_abs > 1e-10)

        # 向量化一阶多项式拟合: v = a*d + b, 最小二乘法
        for i in np.where(valid)[0]:
            A = np.column_stack([d_points[i], np.ones(3)])
            coeffs, _res, _rank, _s = np.linalg.lstsq(A, v_points[i], rcond=None)
            v_pred = coeffs[0] * d_points[i] + coeffs[1]
            linearity[i] = 1 - np.mean(np.abs(v_points[i] - v_pred)) / (v_mean_abs[i] + 1e-10)
        
        # 7. 速度梯度近似（关键WSS符号预测特征）
        v_gradient = (v1 - v2) / (d1 - d2 + 1e-10)
        
        # 8. 速度方向变化率（判断流动复杂性）
        angle_variation = np.std(v_align, axis=1)
        
        # 9. 组合所有新特征（严格15维）
        new_features = np.hstack([
            self.wall_flow_direction,           # 3
            alignment.reshape(-1, 1),             # 1
            np.abs(alignment).reshape(-1, 1),   # 1
            self.wall_flow_quality.reshape(-1, 1),  # 1
            v_align,                              # 3
            linearity.reshape(-1, 1),             # 1
            v_gradient.reshape(-1, 1),            # 1
            angle_variation.reshape(-1, 1),        # 1
            np.zeros((n_wall, 3), dtype=np.float32)  # 3 (占位扩展)
        ])
        
        self.wall_flow_derived_features = new_features
        print(f"  流动衍生特征: {new_features.shape[1]}维")
        print(f"    - 流动方向: 3维")
        print(f"    - 对齐度(代理): 2维")
        print(f"    - 方向质量: 1维")
        print(f"    - 速度-流动对齐: 3维")
        print(f"    - 速度结构: 3维")
        
        return new_features


    def compute_wss_direction_alignment(self, data_file: str = None,
                                    force_recompute: bool = False):
        """
        计算WSS符号与局部流动方向的对齐度
        """
        # 检查是否已有alignment缓存
        if not force_recompute and hasattr(self, 'wss_flow_alignment'):
            print(f"  使用已缓存的alignment数据")
            return {
                'alignment': self.wss_flow_alignment,
                'wss_sign': self.wss_sign,
                'normal_mean': float(np.mean(self.wss_flow_alignment[self.wss_sign > 0])),
                'reverse_mean': float(np.mean(self.wss_flow_alignment[self.wss_sign < 0])),
            }


        print(f"\n{'='*60}")
        print("WSS方向与流动方向对齐分析")
        print(f"{'='*60}")
        
        n_wall = len(self.wall_coords)
        wss_sign = np.sign(self.wall_wss.flatten())
        
        # ========== 核心修改：利用已计算的局部流动方向 ==========
        # 不再从标量速度构造方向，直接使用 compute_local_flow_direction 的结果
        
        if not hasattr(self, 'wall_flow_direction'):
            print("  先计算局部流动方向...")
            self.compute_local_flow_direction()
        
        # 对齐度 = 流动方向 · 壁面法向
        # 层流中：流动应平行壁面（法向≈0），但近壁速度梯度产生WSS
        # WSS>0：速度沿法向增加（壁面0→内部正）→ 流动方向应有远离壁面分量
        # WSS<0：速度沿法向减小（或反向）→ 流动方向可能指向壁面（回流）
        
        alignment = np.sum(self.wall_flow_direction * self.wall_normal, axis=1)
        
        # 但更好的指标：检查流动方向是否与WSS符号一致
        # WSS = μ * (∂u/∂n)，若流动方向与法向同向且WSS>0 → 正常
        # 若流动方向与法向反向且WSS<0 → 回流
        
        # 分类统计
        pos_mask = wss_sign > 0
        neg_mask = wss_sign < 0
        
        print(f"\n  WSS>0点数: {pos_mask.sum()} ({pos_mask.mean():.1%})")
        print(f"  WSS<0点数: {neg_mask.sum()} ({neg_mask.mean():.1%})")
        
        print(f"\n  WSS>0区域:")
        print(f"    平均对齐度: {alignment[pos_mask].mean():.4f}")
        print(f"    (正值=流动有远离壁面分量，负值=有指向壁面分量)")
        
        print(f"\n  WSS<0区域:")
        print(f"    平均对齐度: {alignment[neg_mask].mean():.4f}")
        print(f"    (负值强烈暗示回流)")
        
        # 统计检验
        from scipy.stats import ttest_ind, mannwhitneyu
        # 检查组是否为空
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        
        if n_pos > 0 and n_neg > 0:
            # 两组均有数据，执行统计检验
            t_stat, t_p = ttest_ind(alignment[pos_mask], alignment[neg_mask])
            u_stat, u_p = mannwhitneyu(alignment[pos_mask], alignment[neg_mask], 
                                        alternative='two-sided')
            
            print(f"\n  统计检验:")
            print(f"    t-test: t={t_stat:.4f}, p={t_p:.2e}")
            print(f"    Mann-Whitney U: p={u_p:.2e}")
            
            # 显著性判断
            if t_p < 0.001:
                sig_level = "***"
            elif t_p < 0.01:
                sig_level = "**"
            elif t_p < 0.05:
                sig_level = "*"
            else:
                sig_level = "ns"
            print(f"    显著性: {sig_level}")
            
        elif n_pos == 0:
            # 无正WSS点
            t_stat, t_p, u_stat, u_p = None, float('nan'), None, float('nan')
            print(f"\n  ⚠️ 统计检验跳过: WSS>0 点数为0（全为负WSS或零）")
            print(f"    无法比较两组差异")
            
        elif n_neg == 0:
            # 无负WSS点
            t_stat, t_p, u_stat, u_p = None, float('nan'), None, float('nan')
            print(f"\n  ⚠️ 统计检验跳过: WSS<0 点数为0（全为正WSS）")
            print(f"    这是正常层流预期结果，无需回流分析")
        
        # 回流判断：WSS<0 且 alignment < -0.3（流动明显指向壁面）
        recirculation_threshold = -0.3
        recirculation_mask = neg_mask & (alignment < recirculation_threshold)
        
        print(f"\n  疑似回流（WSS<0 & alignment<{recirculation_threshold}）: "
            f"{recirculation_mask.sum()} ({recirculation_mask.sum()/n_wall:.1%})")
        
        self.wss_flow_alignment = alignment
        self.wss_sign = wss_sign
        
        # 保存（通过compute_local_flow_direction的保存机制）
        if data_file is not None and hasattr(self, 'wall_flow_direction'):
            split_mode = getattr(self, 'split_mode', 'unknown')
            self.save_flow_direction(data_file, split_mode)

        return {
            'alignment': alignment,
            'wss_sign': wss_sign,
            'normal_mean': float(alignment[pos_mask].mean()),
            'reverse_mean': float(alignment[neg_mask].mean()),
            't_test_p': float(t_p),
            'mann_whitney_p': float(u_p),
            'recirculation_mask': recirculation_mask,
            'significant': t_p < 0.05
        }

    def build_raw_neighbor_features(self, k: int = 10) -> np.ndarray:
        """为每个壁面点构建原始 K 近邻特征（不手工提取 v/d/n，让 MLP 自己学）。

        对每个壁面点，取 K 个最近内部点，构建：
          [wall_x, wall_y, wall_z, wall_|v|(=0), wall_p,
           dx_1, dy_1, dz_1, |v_1|, p_1,
           dx_2, dy_2, dz_2, |v_2|, p_2,
           ...
           dx_K, dy_K, dz_K, |v_K|, p_K]

        共 5 + K*5 维。与手工特征(22维)对比，验证手工特征是否等价于 NN 自主学习。

        Returns:
            raw_features: (N_wall, 5 + K*5) 原始近邻特征
        """
        n_wall = len(self.wall_coords)
        interior_tree = getattr(self, 'interior_tree', None)
        if interior_tree is None:
            interior_tree = cKDTree(self.interior_coords)

        # 批量查询每个壁面点的 K 个最近内部点
        dists, idx = interior_tree.query(self.wall_coords, k=k)

        # 内部点特征: 坐标 + 速度 + 压力
        interior_coords = self.interior_coords[idx]          # (N_wall, K, 3)
        interior_vel = self.interior_velocity_scaled[idx]    # (N_wall, K, 1)
        interior_pres = self.interior_pressure_scaled[idx]    # (N_wall, K, 1)

        # 相对坐标 (dx, dy, dz)
        wall_coords_expanded = self.wall_coords[:, np.newaxis, :]  # (N_wall, 1, 3)
        relative_coords = interior_coords - wall_coords_expanded    # (N_wall, K, 3)

        # 壁面自身特征: xyz + |v|(=0) + p
        wall_feat = np.hstack([
            self.wall_coords,                    # (N_wall, 3)
            self.wall_velocity_scaled,           # (N_wall, 1) — 约等于 0
            self.wall_pressure_scaled,           # (N_wall, 1)
        ]).astype(np.float32)                    # (N_wall, 5)

        # 拼接所有近邻特征
        neighbor_feats = np.hstack([
            relative_coords.reshape(n_wall, k * 3),   # K*3: dx,dy,dz
            interior_vel.reshape(n_wall, k * 1),       # K*1: |v|
            interior_pres.reshape(n_wall, k * 1),      # K*1: p
        ]).astype(np.float32)                          # (N_wall, K*5)

        raw_features = np.hstack([wall_feat, neighbor_feats])  # (N_wall, 5 + K*5)

        print(f"  [raw_neighbor] K={k}, 特征维={raw_features.shape[1]} "
              f"(壁面5 + 近邻{k}x5 = {5+k*5})")
        return raw_features

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """返回PyTorch张量（根据实验配置选择特征子集）"""
        if EXP_CONFIG.get('use_raw_neighbors', False):
            # raw_neighbor_mlp: 原始K近邻特征 → MLP自主学习 ∂u/∂n
            k = EXP_CONFIG.get('raw_neighbor_k', 10)
            raw_feat = self.build_raw_neighbor_features(k=k)
            wall_features_to_use = raw_feat
            interior_features_to_use = np.zeros((len(self.interior_coords), raw_feat.shape[1]), dtype=np.float32)
            print(f"  [raw_neighbor_mlp] 原始近邻特征: {wall_features_to_use.shape[1]}维 (K={k})")
        elif EXP_CONFIG['use_coords_only']:
            # pure_mlp_xyz / dgcnn_xyz / gat_xyz: 仅坐标
            wall_features_to_use = self.wall_coords.astype(np.float32)
            interior_features_to_use = self.interior_coords.astype(np.float32)
            print(f"  [coords_only] 使用坐标特征: {wall_features_to_use.shape[1]}维")
        elif EXP_CONFIG.get('use_velocity_pressure_only', False):
            # flow_only: 仅速度幅值和压力（2维：|v|, p）
            vp = np.hstack([
                self.wall_velocity_scaled,
                self.wall_pressure_scaled,
            ]).astype(np.float32)
            wall_features_to_use = vp
            interior_features_to_use = self.interior_features_scaled[:, :20]
            print(f"  [flow_only] 仅速度+压力: {wall_features_to_use.shape[1]}维")
        elif not EXP_CONFIG['use_geometry_features']:
            base = getattr(self, 'wall_features', self.wall_features_scaled)
            wall_features_to_use = base[:, :20]
            interior_features_to_use = self.interior_features_scaled[:, :20]
            print(f"  [no_geometry_mlp] 无几何特征: {wall_features_to_use.shape[1]}维")
        else:
            # 默认：完整增强特征
            wall_features_to_use = getattr(self, 'wall_features_enhanced', self.wall_features_scaled)
            interior_features_to_use = self.interior_features_scaled

        return {
            'wall_coords': torch.FloatTensor(self.wall_coords).to(DEVICE),
            'interior_coords': torch.FloatTensor(self.interior_coords).to(DEVICE),
            'wall_features': torch.FloatTensor(wall_features_to_use).to(DEVICE),
            'interior_features': torch.FloatTensor(interior_features_to_use).to(DEVICE),
            'wall_wss': torch.FloatTensor(self.wall_wss_scaled).to(DEVICE),
            'wall_velocity': torch.FloatTensor(self.wall_velocity_scaled).to(DEVICE),
            'wall_pressure': torch.FloatTensor(self.wall_pressure_scaled).to(DEVICE),
            'interior_velocity': torch.FloatTensor(self.interior_velocity_scaled).to(DEVICE),
            'interior_pressure': torch.FloatTensor(self.interior_pressure_scaled).to(DEVICE),
        }

    def get_combined_tensors(self) -> Dict[str, torch.Tensor]:
        """返回壁面+内部组合点云张量（用于 DGCNN/GAT 等点云方法）

        点云方法需要看到内部点的速度/压力信息才能学习 ∂u/∂n|_wall，
        因此将壁面和内部点合并为一个点云，通过 k-NN 图传播信息。

        Returns:
            combined_coords: (N_wall + N_interior, 3) 组合坐标
            combined_features: (N_wall + N_interior, F) 组合特征
                - gnn_input='coords_only': [x_norm, y_norm, z_norm, is_wall]
                - gnn_input='coords_features': [x_norm, y_norm, z_norm, |v|, p, is_wall]
            wall_mask: (N_total,) 布尔 mask，True=壁面点
            wall_wss: (N_wall, 1) 壁面 WSS 标签（仅壁面点有监督）
        """
        n_wall = len(self.wall_coords)
        n_interior = len(self.interior_coords)

        # 合并坐标
        all_coords = np.vstack([self.wall_coords, self.interior_coords]).astype(np.float32)

        # 构建每点特征
        is_wall_flag = np.zeros((n_wall + n_interior, 1), dtype=np.float32)
        is_wall_flag[:n_wall] = 1.0  # 壁面=1, 内部=0

        # 兼容 DGCNN/GAT 的 'gnn_input' 和 PointNet++ 的 'pointnetpp_input'
        gnn_input = EXP_CONFIG.get('gnn_input') or EXP_CONFIG.get('pointnetpp_input', 'coords_only')

        if gnn_input == 'coords_only':
            # 仅坐标 + is_wall 标记（4维），不含速度/压力
            # 速度始终为0（无滑移），压力无额外信息 → 4维
            combined_features = np.hstack([all_coords, is_wall_flag]).astype(np.float32)
            extra = '(x,y,z,is_wall) — 纯几何'
        else:
            # coords_features: 坐标 + 速度 + 压力 + is_wall（6维）
            vel_all = np.vstack([
                self.wall_velocity_scaled,      # 壁面速度 ≈ 0（无滑移）
                self.interior_velocity_scaled,   # 内部速度（非零！关键信息）
            ]).astype(np.float32)
            pres_all = np.vstack([
                self.wall_pressure_scaled,
                self.interior_pressure_scaled,
            ]).astype(np.float32)
            combined_features = np.hstack([all_coords, vel_all, pres_all, is_wall_flag]).astype(np.float32)
            extra = '(x,y,z,|v|,p,is_wall) — 含内部流场'

        # wall_mask 和 标签
        wall_mask = np.zeros(n_wall + n_interior, dtype=bool)
        wall_mask[:n_wall] = True
        wall_wss = self.wall_wss_scaled.astype(np.float32)

        print(f"  [GNN combined] 壁面={n_wall} + 内部={n_interior} = {n_wall+n_interior} 点, "
              f"特征={combined_features.shape[1]}维 {extra}")

        return {
            'combined_coords': torch.FloatTensor(all_coords).to(DEVICE),
            'combined_features': torch.FloatTensor(combined_features).to(DEVICE),
            'wall_mask': torch.BoolTensor(wall_mask).to(DEVICE),
            'wall_wss': torch.FloatTensor(wall_wss).to(DEVICE),
            'wall_velocity': torch.FloatTensor(self.wall_velocity_scaled).to(DEVICE),
            'wall_pressure': torch.FloatTensor(self.wall_pressure_scaled).to(DEVICE),
        }