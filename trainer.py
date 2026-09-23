"""训练器"""

import torch
import numpy as np
from scipy.spatial import cKDTree

from config import DEVICE, SAVE_DIR, TrainConfig, EXP_CONFIG
from utils import AdaptiveGradientClipper, MetricsLogger
from losses import GradientConstraint, PhysicsInformedLoss, CoupledWSSLoss
import matplotlib.pyplot as plt

# 混合精度训练
_USE_AMP = torch.cuda.is_available()
_amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

try:
    _GradScaler = torch.amp.GradScaler
    _autocast_ctx = lambda dtype=_amp_dtype: torch.amp.autocast('cuda', dtype=dtype)
except AttributeError:
    _GradScaler = torch.cuda.amp.GradScaler
    _autocast_ctx = lambda dtype=_amp_dtype: torch.cuda.amp.autocast(dtype=dtype)

class DirectWSSTrainer:
    """WSS预测网络训练器"""
    
    def __init__(self, model, dataset, save_dir: str = SAVE_DIR):
        self.model = model.to(DEVICE)
        self.dataset = dataset
        self.data = dataset.get_tensors()
        self.save_dir = save_dir

        # 实验标志
        self.use_sign_branch = EXP_CONFIG['use_sign_branch']
        self.use_physics_loss = EXP_CONFIG['use_physics_loss']
        self.decouple_head = EXP_CONFIG['decouple_head']
        self.is_ml_baseline = EXP_CONFIG['ml_algorithm'] is not None

        # 梯度裁剪
        self.grad_clipper = AdaptiveGradientClipper(
            clip_factor=TrainConfig.GRAD_CLIP_FACTOR
        )
        
        # 预计算邻居（用于梯度约束）
        self._precompute_neighbors()
        self.velocity_gradient_constraint = PhysicsInformedLoss(lambda_decay=1)

        # 替换为耦合损失
        # 损失函数：无符号分支/单头模型时用简单MSE
        if self.use_sign_branch and self.decouple_head:
            self.coupled_loss_fn = CoupledWSSLoss(
                lambda_recon=1.0,
                lambda_mag=0.5,
                lambda_sign=0.5,
                neg_weight=5.0,
            ).to(DEVICE)
        else:
            self.coupled_loss_fn = None  # 使用简单 MSE
        
        # 兼容 DirectWSSPredictor (wss_net) / CoupledSingleHeadPredictor (net) / PointNet++
        if hasattr(model, 'wss_net'):
            self._main_net = model.wss_net
        elif hasattr(model, 'net'):
            self._main_net = model.net
        else:
            self._main_net = model  # PointNet++ 等无子网络的模型

        # PointNet++ 需要每 case 更新坐标
        if hasattr(self.model, 'set_coords'):
            self.model.set_coords(self.data['wall_coords'])
        param_groups = [
            {'params': self._main_net.parameters(), 'lr': TrainConfig.LR_INITIAL, 'weight_decay': 1e-5}
        ]
        if hasattr(model, 'sign_net') and model.sign_net is not None:
            param_groups.append(
                {'params': model.sign_net.parameters(), 'lr': TrainConfig.LR_INITIAL * 2.0, 'weight_decay': 1e-4}
            )
        # Optimizer: 按 EXP_CONFIG['optimizer_type'] 分支 (默认 adamw, B 组用 adam)
        _opt_type = EXP_CONFIG.get('optimizer_type', 'adamw')
        if _opt_type == 'adam':
            self.optimizer = torch.optim.Adam(
                param_groups, betas=(0.9, 0.999), eps=1e-8)
        else:
            self.optimizer = torch.optim.AdamW(
                param_groups, betas=(0.9, 0.999), eps=1e-8)
        print(f"  [Optimizer] type={_opt_type}, scheduler={EXP_CONFIG.get('scheduler_type','warmup_cosine')}")
        # 符号网络学习率×2、权重衰减×10：鼓励符号分支快速收敛
        
        # 学习率调度：预热 + 余弦退火 (可被 EXP_CONFIG['scheduler_type'] 覆盖为恒定)
        _sched_type = EXP_CONFIG.get('scheduler_type', TrainConfig.SCHEDULER_TYPE)

        if _sched_type == 'constant':
            # B 组: 恒定 lr, 无 warmup 无 cosine
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=0)
        else:
            # A 组 (默认): Warmup → CosineAnnealing
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=TrainConfig.LR_WARMUP_STEPS
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=9000,
                        eta_min=TrainConfig.LR_MIN
                    )
                ],
                milestones=[TrainConfig.LR_WARMUP_STEPS]
            )
        
        # 损失权重
        self.lambda_wss = TrainConfig.LAMBDA_WSS
        self.lambda_phys = TrainConfig.LAMBDA_PHYS
        
        # 记录
        self.loss_history = {'total': [], 'wss': [], 'mag': [], 'sign': [], 'phys': [], 'velocity': [],
                             'sign_acc': [], 'neg_recall': [], 'neg_precision': [], 'neg_f1': [], 'neg_mag': [], 'sign_violation': [], 'sign_violation_pos': [],
                             'tp': [], 'tn': [], 'fp': [], 'fn': [], 'r2_train': []}
        self.metrics_logger = MetricsLogger(save_dir)
        self.best_r2 = -float('inf')
        self.force_plot = True  # 强制绘制标志
        self.best_epoch = 0

        # 混合精度
        self.scaler = _GradScaler('cuda') if _USE_AMP else None
        
        # ====== 新增：关键变量梯度历史记录 ======
        self.gradient_history = {
            'v1': [], 'd1': [], 'v2': [], 'd2': [], 'v3': [], 'd3': [],
            'wss_ref': [],
            'n1': [], 'n2': [], 'n3': [],
            'density': [], 'curvedness': [], 'shape_index': [], 'anisotropy': [],
            'sep_linearity': [], 'sep_inflection': [], 'sep_expansion': []
        }
        self.gradient_history_std = {k: [] for k in self.gradient_history.keys()}  # 标准差
        self.gradient_record_interval = 50  # 每50 epoch记录一次梯度
    
    def _record_gradients(self, model, dataset, epoch):
        """
        记录关键变量的梯度统计信息

        Args:
            model: 当前模型
            dataset: 数据集
            epoch: 当前epoch数
        """
        # PointNet++ 等非特征通道模型不支持此分析
        if hasattr(model, 'set_coords'):
            return
        was_training = model.training
        
        try:
            features = dataset.get_tensors()['wall_features'].clone().detach().requires_grad_(True)
            n_feats = features.shape[1]
            if n_feats < 20:
                return  # 仅完整特征布局支持梯度分析（>=20维）

            with torch.enable_grad():
                wss_pred = model(features,
                            torch.zeros_like(features[:, 0:1]),
                            torch.zeros_like(features[:, 0:1]))
                grads = torch.autograd.grad(wss_pred.sum(), features,
                                        create_graph=False, retain_graph=False)[0]

            grad_data = {
                'v1': torch.abs(grads[:, 4]).cpu().numpy(),
                'd1': torch.abs(grads[:, 5]).cpu().numpy(),
                'v2': torch.abs(grads[:, 6]).cpu().numpy(),
                'd2': torch.abs(grads[:, 7]).cpu().numpy(),
                'v3': torch.abs(grads[:, 8]).cpu().numpy(),
                'd3': torch.abs(grads[:, 9]).cpu().numpy(),
                'wss_ref': torch.abs(grads[:, 10]).cpu().numpy(),
                'n1': torch.abs(grads[:, 11:14]).mean(dim=1).cpu().numpy(),  # n1三个分量均值
                'n2': torch.abs(grads[:, 14:17]).mean(dim=1).cpu().numpy(),
                'n3': torch.abs(grads[:, 17:20]).mean(dim=1).cpu().numpy(),
                'density': torch.abs(grads[:, 20]).cpu().numpy(),
                'curvedness': torch.abs(grads[:, 21]).cpu().numpy(),
                'shape_index': torch.abs(grads[:, 22]).cpu().numpy(),
                'anisotropy': torch.abs(grads[:, 23]).cpu().numpy(),
                'sep_linearity': torch.abs(grads[:, 39]).cpu().numpy(),
                'sep_inflection': torch.abs(grads[:, 40]).cpu().numpy(),
                'sep_expansion': torch.abs(grads[:, 50]).cpu().numpy(),
            }
            
            # 记录均值和标准差
            for key, values in grad_data.items():
                self.gradient_history[key].append(float(np.mean(values)))
                self.gradient_history_std[key].append(float(np.std(values)))
            
        except Exception as e:
            print(f"⚠️ 记录梯度时出错: {e}")
        finally:
            if was_training and not model.training:
                model.train()
            if features.grad is not None:
                features.grad.zero_()

    def plot_gradient_evolution(self, save_path: str = None):
        """
        绘制关键变量梯度随epoch变化的曲线图
        
        包含：
        1. 总图：所有变量的梯度均值演化（归一化对比）
        2. 14个子图：每个变量的梯度均值+标准差带
        """
        import os
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'gradient_evolution.png')
        
        # 准备数据
        epochs = np.arange(len(self.gradient_history['v1'])) * self.gradient_record_interval
        
        if len(epochs) == 0:
            print("⚠️ 无梯度记录数据，跳过绘制")
            return
        
        # 变量分组（按物理意义）
        velocity_group = ['v1', 'v2', 'v3']
        distance_group = ['d1', 'd2', 'd3']
        direction_group = ['n1', 'n2', 'n3']
        geometry_group = ['density', 'curvedness', 'shape_index', 'anisotropy']
        physics_group = ['wss_ref']
        
        all_groups = velocity_group + distance_group + direction_group + physics_group + geometry_group
        
        # ========== 图1: 总图 - 所有变量归一化对比 ==========
        fig_total, ax_total = plt.subplots(figsize=(14, 8))
        
        # 归一化到[0,1]以便对比
        for key in all_groups:
            values = np.array(self.gradient_history[key])
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
            else:
                normalized = values
            ax_total.plot(epochs, normalized, label=key, linewidth=1.5, alpha=0.8)
        
        ax_total.set_xlabel('Epoch', fontsize=12)
        ax_total.set_ylabel('Normalized Gradient Importance', fontsize=12)
        ax_total.set_title('Gradient Evolution of All Features (Normalized)', fontsize=14, fontweight='bold')
        ax_total.legend(loc='upper left', ncol=3, fontsize=9)
        ax_total.grid(True, alpha=0.3)
        
        # 添加分组背景色带
        ax_total.axvspan(epochs[0], epochs[-1], alpha=0.05, color='blue', label='Velocity')
        
        plt.tight_layout()
        save_path_total = save_path.replace('.png', '_total.png')
        plt.savefig(save_path_total, dpi=300, bbox_inches='tight')
        print(f"✓ 梯度演化总图已保存: {save_path_total}")
        plt.close(fig_total)
        
        # ========== 图2: 14个子图 - 每个变量详细演化 ==========
        n_vars = len(all_groups)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols  # 向上取整
        
        fig_detail, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten() if n_vars > 1 else [axes]
        
        colors = {
            'velocity': '#1f77b4',  # 蓝色
            'distance': '#2ca02c',    # 绿色
            'direction': '#ff7f0e',  # 橙色
            'physics': '#d62728',    # 红色
            'geometry': '#9467bd'    # 紫色
        }
        
        for idx, key in enumerate(all_groups):
            ax = axes[idx]
            
            values = np.array(self.gradient_history[key])
            stds = np.array(self.gradient_history_std[key])
            
            # 确定颜色组
            if key in velocity_group:
                color = colors['velocity']
                group_name = 'Velocity'
            elif key in distance_group:
                color = colors['distance']
                group_name = 'Distance'
            elif key in direction_group:
                color = colors['direction']
                group_name = 'Direction'
            elif key in physics_group:
                color = colors['physics']
                group_name = 'Physics'
            else:
                color = colors['geometry']
                group_name = 'Geometry'
            
            # 绘制均值线
            ax.plot(epochs, values, color=color, linewidth=2, label=f'{key} (mean)')
            
            # 绘制标准差带
            ax.fill_between(epochs, 
                           values - stds, 
                           values + stds, 
                           color=color, alpha=0.2, label=f'{key} (±std)')
            
            # 标注初始值和最终值
            ax.scatter([epochs[0]], [values[0]], color='green', s=50, zorder=5, marker='o')
            ax.scatter([epochs[-1]], [values[-1]], color='red', s=50, zorder=5, marker='s')
            ax.annotate(f'{values[0]:.2e}', xy=(epochs[0], values[0]), 
                       xytext=(10, 10), textcoords='offset points', fontsize=8, color='green')
            ax.annotate(f'{values[-1]:.2e}', xy=(epochs[-1], values[-1]), 
                       xytext=(10, -15), textcoords='offset points', fontsize=8, color='red')
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Gradient Magnitude', fontsize=10)
            ax.set_title(f'{key} [{group_name}]', fontsize=11, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        # 隐藏多余的子图
        for idx in range(n_vars, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Gradient Evolution of Individual Features\n'
                    'Green circle=Initial, Red square=Final', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path_detail = save_path.replace('.png', '_detail.png')
        plt.savefig(save_path_detail, dpi=300, bbox_inches='tight')
        print(f"✓ 梯度演化详细子图已保存: {save_path_detail}")
        plt.close(fig_detail)
        
        # ========== 图3: 分组对比图（物理意义分组） ==========
        fig_group, axes_group = plt.subplots(2, 3, figsize=(18, 12))
        axes_group = axes_group.flatten()
        
        group_configs = [
            (velocity_group, 'Velocity Gradients (v1,v2,v3)', colors['velocity']),
            (distance_group, 'Distance Gradients (d1,d2,d3)', colors['distance']),
            (direction_group, 'Direction Gradients (n1,n2,n3)', colors['direction']),
            (physics_group, 'Physics Reference (wss_ref)', colors['physics']),
            (geometry_group, 'Geometry Gradients (density,curvedness,shape_index,anisotropy)', colors['geometry']),
        ]
        
        for idx, (group_vars, title, color) in enumerate(group_configs):
            ax = axes_group[idx]
            
            for key in group_vars:
                values = np.array(self.gradient_history[key])
                ax.plot(epochs, values, label=key, linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Gradient Magnitude', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_yscale('log')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # 隐藏最后一个空子图
        axes_group[-1].axis('off')
        
        plt.suptitle('Gradient Evolution by Physical Groups', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path_group = save_path.replace('.png', '_group.png')
        plt.savefig(save_path_group, dpi=300, bbox_inches='tight')
        print(f"✓ 梯度演化分组图已保存: {save_path_group}")
        plt.close(fig_group)
        
        # 保存原始数据到CSV
        self._save_gradient_csv(epochs)
    
    def _save_gradient_csv(self, epochs):
        """保存梯度历史到CSV"""
        import csv
        import os
        
        csv_path = os.path.join(self.save_dir, 'gradient_evolution.csv')
        
        # 构建表头
        header = ['epoch']
        for key in self.gradient_history.keys():
            header.append(f'{key}_mean')
            header.append(f'{key}_std')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for i in range(len(epochs)):
                row = [epochs[i]]
                for key in self.gradient_history.keys():
                    row.append(self.gradient_history[key][i])
                    row.append(self.gradient_history_std[key][i])
                writer.writerow(row)
        
        print(f"✓ 梯度演化数据已保存: {csv_path}")

    def _precompute_neighbors(self):
        """预计算邻居关系"""
        coords = self.dataset.wall_coords
        tree = cKDTree(coords)
        kneighbors = 6
        _, indices = tree.query(coords, k=kneighbors)
        self.neighbor_indices = torch.LongTensor(indices[:, 1:]).to(DEVICE)
        
        # 梯度约束
        self.gradient_constraint = GradientConstraint(
            weight=0.05, 
            k_neighbors=kneighbors
        )
        self.gradient_constraint.precompute(
            coords, 
            self.dataset.wall_wss, 
            DEVICE
        )

    def compute_physics_loss(self, wss_pred: torch.Tensor, epoch: int) -> torch.Tensor:
        """计算物理约束损失"""
        return self.gradient_constraint(wss_pred)

    def _plot_velocity_constraint_violation(self, model, dataset, save_dir, epoch, force_plot=False, name="train"):
        """
        绘制v1/v2/v3重要性的空间分布比例图

        Args:
            model: 当前模型
            dataset: 数据集
            save_dir: 保存目录
            epoch: 当前epoch数
            force_plot: 是否强制绘制（用于每个dataset的初始分析）
        """
        # PointNet++ 等非特征通道模型不支持此分析
        if hasattr(model, 'set_coords'):
            return 0.0
            
        print(f"\n{'='*60}")
        print(f"【Epoch {epoch}】绘制V1/V2/V3梯度比例空间分布...")
        print(f"{'='*60}")
        
        model.eval()
        
        try:
            # 计算各点梯度
            features = dataset.get_tensors()['wall_features'].clone().detach().requires_grad_(True)
            n_feats = features.shape[1]
            if n_feats < 20:
                return 0.0  # 仅完整特征布局支持此分析

            # 使用零速度/压力输入，仅观察几何特征对WSS的影响
            wss_pred = model(features,
                             torch.zeros_like(features[:, 0:1]),
                             torch.zeros_like(features[:, 0:1]))
            
            grads = torch.autograd.grad(wss_pred.sum(), features, create_graph=False)[0]
            
            grad_v1 = torch.abs(grads[:, 4]).cpu().numpy()
            grad_v2 = torch.abs(grads[:, 6]).cpu().numpy()
            grad_v3 = torch.abs(grads[:, 8]).cpu().numpy()
            
            # 计算比例 (v1应该远大于v2，v2应该远大于v3)
            ratio_v1_v2 = grad_v1 / (grad_v2 + 1e-8)
            ratio_v2_v3 = grad_v2 / (grad_v3 + 1e-8)
            
            # # 提取 D1, D2, D3 距离特征 (索引 5, 7, 9)
            # d1 = torch.abs(grads[:, 5]).cpu().numpy()
            # d2 = torch.abs(grads[:, 7]).cpu().numpy()
            # d3 = torch.abs(grads[:, 9]).cpu().numpy()
            # 提取 D1, D2, D3 距离特征 (索引 5, 7, 9)
            d1 = features[:, 5].detach().cpu().numpy()
            d2 = features[:, 7].detach().cpu().numpy()
            d3 = features[:, 9].detach().cpu().numpy()
            coords = dataset.wall_coords

            # ========== 图1: V1/V2 和 V2/V3 比例图 (2x2) ==========
            # 绘制空间分布 - 2行2列布局
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            
            # 1. v1/v2比例空间分布 (XY平面)
            ax = axes[0, 0]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=ratio_v1_v2, 
                           cmap='RdYlBu_r', s=15, vmin=0, vmax=10, alpha=0.7)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'V1/V2 Gradient Ratio (Epoch {epoch})')
            cbar1 = plt.colorbar(sc, ax=ax, label='Ratio')
            cbar1.ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)  # 标记ratio=1的阈值
            
            # 2. v2/v3比例空间分布 (XY平面)
            ax = axes[0, 1]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=ratio_v2_v3, 
                           cmap='RdYlBu_r', s=15, vmin=0, vmax=10, alpha=0.7)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'V2/V3 Gradient Ratio (Epoch {epoch})')
            cbar2 = plt.colorbar(sc, ax=ax, label='Ratio')
            cbar2.ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
            
            # 3. v1/v2比例空间分布 (XZ平面)
            ax = axes[1, 0]
            sc = ax.scatter(coords[:, 0], coords[:, 2], c=ratio_v1_v2, 
                           cmap='RdYlBu_r', s=15, vmin=0, vmax=10, alpha=0.7)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title(f'V1/V2 Ratio (XZ View)')
            plt.colorbar(sc, ax=ax, label='Ratio')
            
            # 4. v2/v3比例空间分布 (XZ平面)
            ax = axes[1, 1]
            sc = ax.scatter(coords[:, 0], coords[:, 2], c=ratio_v2_v3, 
                           cmap='RdYlBu_r', s=15, vmin=0, vmax=10, alpha=0.7)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title(f'V2/V3 Ratio (XZ View)')
            plt.colorbar(sc, ax=ax, label='Ratio')
            
            plt.suptitle(f'Velocity Gradient Ratio Analysis - {dataset.split_mode} set\n'
                        f'(Expected: V1/V2 >> 1, V2/V3 >> 1)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            save_path = f'{save_dir}/velocity_ratio_epoch{epoch}_{name}_{dataset.split_mode}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 梯度比例分析图已保存: {save_path}")
            plt.close(fig)
            
            # ========== 图2: D1/D2/D3 距离分布 (3行 x 3列 = 9子图) ==========
            fig2, axes2 = plt.subplots(3, 3, figsize=(18, 18))
            
            # 距离数据列表和名称
            distances = [d1, d2, d3]
            dist_names = ['D1 (Nearest)', 'D2 (2nd)', 'D3 (3rd)']
            # 统一colorbar范围
            d_max = max(np.max(d1), np.max(d2), np.max(d3))
            d_min = min(np.min(d1), np.min(d2), np.min(d3))
            
            # 视角配置
            views = [
                (0, 1, 'X', 'Y'),  # XY
                (0, 2, 'X', 'Z'),  # XZ
                (1, 2, 'Y', 'Z'),  # YZ
            ]
            
            for row_idx, (dist_data, dist_name) in enumerate(zip(distances, dist_names)):
                for col_idx, (i, j, xlabel, ylabel) in enumerate(views):
                    ax = axes2[row_idx, col_idx]
                    
                    sc = ax.scatter(coords[:, i], coords[:, j], 
                                   c=dist_data, 
                                   cmap='viridis', 
                                   s=15, 
                                   vmin=d_min, 
                                   vmax=d_max,
                                   alpha=0.7)
                    
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_title(f'{dist_name} ({xlabel}{ylabel})')
                    ax.set_aspect('equal', adjustable='box')
                    
                    # 每个子图添加colorbar
                    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Distance (mm)', rotation=270, labelpad=15)
            
            plt.suptitle(f'Wall Distance Distribution Analysis - {dataset.split_mode} set\n'
                        f'D1/D2/D3: Distance to 1st/2nd/3rd nearest interior point', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            save_path2 = f'{save_dir}/distance_distribution_epoch{epoch}_{dataset.split_mode}.png'
            plt.savefig(save_path2, dpi=300, bbox_inches='tight')
            print(f"✓ 距离分布分析图已保存: {save_path2}")
            plt.close(fig2)

            # 额外保存详细统计信息
            # self._log_velocity_constraint_stats(ratio_v1_v2, ratio_v2_v3, epoch, dataset.split_mode)
            
            violation_ratio = float(np.mean((ratio_v1_v2 < 1) | (ratio_v2_v3 < 1)))
            
        except Exception as e:
            print(f"⚠️ 绘制velocity比例分析图时出错: {e}")
            import traceback
            traceback.print_exc()
            violation_ratio = 0.0
        
        finally:
            model.train()
        
        return violation_ratio

    def _log_velocity_constraint_stats(self, ratio_v1_v2, ratio_v2_v3, violation, epoch):
        """记录velocity约束违反的统计信息到日志"""
        stats = {
            'epoch': epoch,
            'violation_ratio': float(violation.mean()),
            'ratio_v1_v2_mean': float(np.mean(ratio_v1_v2)),
            'ratio_v1_v2_std': float(np.std(ratio_v1_v2)),
            'ratio_v2_v3_mean': float(np.mean(ratio_v2_v3)),
            'ratio_v2_v3_std': float(np.std(ratio_v2_v3)),
            'n_violation_points': int(violation.sum()),
            'total_points': len(violation)
        }
        
        # 保存到CSV
        import csv
        import os
        stats_path = os.path.join(self.save_dir, 'velocity_constraint_stats.csv')
        file_exists = os.path.exists(stats_path)
        
        with open(stats_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)
        
        # 打印统计信息
        print(f"\n  Velocity Constraint Statistics @ Epoch {epoch}:")
        print(f"    违反比例: {stats['violation_ratio']:.2%}")
        print(f"    v1/v2 比例: {stats['ratio_v1_v2_mean']:.2f} ± {stats['ratio_v1_v2_std']:.2f}")
        print(f"    v2/v3 比例: {stats['ratio_v2_v3_mean']:.2f} ± {stats['ratio_v2_v3_std']:.2f}")
        print(f"    违反点数: {stats['n_violation_points']}/{stats['total_points']}")


    def train(self, epochs: int = 10000, print_every: int = 500,
              current_epochs: int = 0):
        """训练循环"""
        print("="*60)
        print("训练WSS预测网络")
        print("="*60)

        # 组合点云模型（PointNet++/DGCNN/GAT）：使用壁面+内部组合点云，仅壁面点有监督
        # PointNet (vanilla) 有 set_coords() 但仅用壁面点，排除在此分支外
        is_gnn = getattr(self.model, '_model_type', None) in (
            'pointnetpp', 'dgcnn', 'gat', 'pointnet_combined')

        if is_gnn:
            self.data = self.dataset.get_combined_tensors()
            self.model.set_coords(self.data['combined_coords'])
            wall_features = self.data['combined_features']
            wall_wss_true = self.data['wall_wss']
            wall_velocity_real = self.data.get('wall_velocity')
            wall_pressure_real = self.data.get('wall_pressure')
            wall_mask = self.data['wall_mask']
            print(f"训练数据: {len(wall_features)} 点 (壁面+内部), "
                  f"壁面={wall_mask.sum().item()} 点")
        else:
            wall_features = self.data['wall_features']
            wall_wss_true = self.data['wall_wss']
            wall_velocity_real = self.data['wall_velocity']
            wall_pressure_real = self.data['wall_pressure']
            wall_mask = None
            # 轻量模型（PointNet vanilla 等）：每次进入 train() 恢复壁面坐标
            # （evaluator 可能已切换到验证集坐标）
            if hasattr(self.model, 'set_coords'):
                self.model.set_coords(self.data['wall_coords'])
            print(f"训练数据: {len(wall_features)} 壁面点")


        # ====== 新增：每个dataset开始时强制绘制初始比例图 ======
        if current_epochs == 0:  # 仅在训练开始时绘制一次
            print(f"\n{'='*60}")
            print("【Dataset初始分析】绘制V1/V2/V3梯度比例空间分布...")
            print(f"{'='*60}")
            self._plot_velocity_constraint_violation(
                self.model, self.dataset, self.save_dir, 
                current_epochs, force_plot=True
            )
        # =========================================================

        for epoch in range(epochs):
            total_epoch = current_epochs + epoch + 1
            # ===== 阶梯式硬约束：每500轮变化一次 =====
            stage = min(total_epoch // 500, 6)          # 0,1,2,3,4,5,6（共7个台阶）
            ramp = stage / 6.0                          # 0 → 0.167 → 0.333 → ... → 1.0
            hard_margin = 0.15 * ramp                      # 0 -> 0.15

            # 阈值从 0.5/0.5 逐渐变为 0.35/0.65
            # 负样本侧更严格（因为负样本是主要矛盾）
            neg_thresh = 0.50 - hard_margin * 1.0          # 0.50 -> 0.35
            pos_thresh = 0.50 + hard_margin * 0.67         # 0.50 -> 0.60

            self.optimizer.zero_grad()

            # ===== 混合精度 autocast =====
            with (_autocast_ctx() if _USE_AMP else torch.enable_grad()):
                # 前向传播（GNN模型传递 wall_mask 以仅返回壁面预测）
                if is_gnn:
                    wss_pred, magnitude, sign_prob = self.model.forward_coupled(
                        wall_features, wall_velocity_real, wall_pressure_real,
                        wall_mask=wall_mask)
                else:
                    wss_pred, magnitude, sign_prob = self.model.forward_coupled(
                        wall_features, wall_velocity_real, wall_pressure_real
                    )

                # 损失计算：解耦头用耦合损失，单头/无符号用MSE
                if self.coupled_loss_fn is not None:
                    coupled_loss, metrics = self.coupled_loss_fn(
                        wss_pred, magnitude, sign_prob, wall_wss_true, neg_thresh, pos_thresh
                    )
                else:
                    recon_loss = torch.nn.functional.mse_loss(wss_pred, wall_wss_true)
                    coupled_loss = recon_loss
                    with torch.no_grad():
                        sign_true = (wall_wss_true > 0).float()
                        sign_pred = (wss_pred > 0).float()
                        _sa = (sign_pred == sign_true).float().mean().item()
                        _neg_mask = (sign_true <= 0.5)
                        _nr = ((sign_pred == sign_true)[_neg_mask]).float().mean().item() if _neg_mask.sum() > 0 else 1.0
                        _np_mask = (sign_pred <= 0.5)
                        _np = ((sign_pred == sign_true)[_np_mask]).float().mean().item() if _np_mask.sum() > 0 else 0.0
                        _nf1 = 2*_nr*_np/(_nr+_np) if (_nr+_np)>0 else 0.0
                        _tp = int(((sign_true>0.5)&(sign_pred>0.5)).sum().item())
                        _tn = int(((sign_true<=0.5)&(sign_pred<=0.5)).sum().item())
                        _fp = int(((sign_true<=0.5)&(sign_pred>0.5)).sum().item())
                        _fn = int(((sign_true>0.5)&(sign_pred<=0.5)).sum().item())
                        _pos_mask = (sign_true > 0.5)
                        _pa = ((sign_pred==sign_true)[_pos_mask]).float().mean().item() if _pos_mask.sum()>0 else 0.0
                    metrics = {
                        'recon': recon_loss.item(), 'mag': 0.0, 'sign': 0.0,
                        'sign_acc': _sa, 'pos_acc': _pa,
                        'neg_recall': _nr, 'neg_precision': _np, 'neg_f1': _nf1,
                        'neg_mag': 0.0, 'sign_violation': 0.0, 'sign_violation_pos': 0.0,
                        'confusion': {'tp': _tp, 'tn': _tn, 'fp': _fp, 'fn': _fn}
                    }

                # 物理损失
                phys_loss = self.compute_physics_loss(wss_pred, epoch)

                if self.use_physics_loss:
                    velocity_loss = self.velocity_gradient_constraint(
                        torch.cat([wall_features, wall_velocity_real, wall_pressure_real], dim=1), self.model
                    )
                else:
                    velocity_loss = torch.tensor(0.0, device=DEVICE)

                # 总损失
                total_loss = coupled_loss + self.lambda_phys * phys_loss + velocity_loss

            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_threshold = self.grad_clipper(self._main_net.parameters())
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                clip_threshold = self.grad_clipper(self._main_net.parameters())
                self.optimizer.step()
            self.scheduler.step()
            
            # 记录
            self.loss_history['total'].append(total_loss.item())
            self.loss_history['wss'].append(metrics['recon'])
            self.loss_history['mag'].append(metrics['mag'])
            self.loss_history['sign'].append(metrics['sign'])
            self.loss_history['phys'].append(self.lambda_phys * phys_loss.item())
            self.loss_history['velocity'].append(velocity_loss.item())
            self.loss_history['sign_acc'].append(metrics['sign_acc'])
            self.loss_history['neg_recall'].append(metrics['neg_recall'])
            self.loss_history['neg_precision'].append(metrics['neg_precision'])
            self.loss_history['neg_f1'].append(metrics['neg_f1'])
            self.loss_history['neg_mag'].append(metrics['neg_mag'])
            self.loss_history['sign_violation'].append(metrics['sign_violation'])
            self.loss_history['sign_violation_pos'].append(metrics['sign_violation_pos'])
            self.loss_history['tp'].append(metrics['confusion']['tp'])
            self.loss_history['tn'].append(metrics['confusion']['tn'])
            self.loss_history['fp'].append(metrics['confusion']['fp'])
            self.loss_history['fn'].append(metrics['confusion']['fn'])

            # ====== 新增：定期记录梯度 ======
            if (epoch + 1) % self.gradient_record_interval == 0:
                self._record_gradients(self.model, self.dataset, current_epochs + epoch + 1)

            # 打印
            if (epoch + 1) % print_every == 0:
                with torch.no_grad():
                    r2 = 1 - torch.sum((wall_wss_true - wss_pred)**2) / \
                         torch.sum((wall_wss_true - wall_wss_true.mean())**2)
                    self.loss_history['r2_train'].append(r2.item())

                    # WSS幅值 R²：衡量模型对|WSS|的回归能力，排除符号预测干扰
                    r2_mag = 1 - torch.sum((wall_wss_true.abs() - wss_pred.abs())**2) / \
                             torch.sum((wall_wss_true.abs() - wall_wss_true.abs().mean())**2)

                # 跟踪训练集最佳 R²
                if r2.item() > self.best_r2:
                    self.best_r2 = r2.item()
                    self.best_epoch = current_epochs + epoch + 1

                if sign_prob is not None:
                    print(f"sign_prob: mean={sign_prob.mean():.3f}, std={sign_prob.std():.3f}")
                    if (wall_wss_true>0).sum()>0 and (wall_wss_true<0).sum()>0:
                        print(f"sign_prob (真正): {sign_prob[wall_wss_true>0].mean():.3f}")
                        print(f"sign_prob (真负): {sign_prob[wall_wss_true<0].mean():.3f}")
                        print(f"sign_prob 区分度: {(sign_prob[wall_wss_true>0].mean() - sign_prob[wall_wss_true<0].mean()):.3f}")

                print(f"Epoch [{epoch+1}/{epochs}] | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                print(f"  Loss: Total={total_loss.item():.4e} "
                      f"Recon={metrics['recon']:.4e} "
                      f"Mag={metrics['mag']:.4e} "
                      f"Sign={metrics['sign']:.4e}")
                print(f"  R²: {r2.item():.4f} | R²_mag: {r2_mag.item():.4f}")
                print(f"  Sign-Acc: {metrics['sign_acc']:.3f} | "
                      f"Pos-Acc: {metrics['pos_acc']:.3f} | "
                      f"Neg-Recall: {metrics['neg_recall']:.3f} | "
                      f"Neg-Prec: {metrics['neg_precision']:.3f} | "
                      f"Neg-F1: {metrics['neg_f1']:.3f}")
                print(f"  Confusion: TP={metrics['confusion']['tp']} "
                      f"TN={metrics['confusion']['tn']} "
                      f"FP={metrics['confusion']['fp']} "
                      f"FN={metrics['confusion']['fn']}")

                # 记录到CSV（完整指标，供论文数据分析）
                self.metrics_logger.log_epoch(
                    epoch=current_epochs + epoch + 1,
                    losses={
                        'total': total_loss.item(),
                        'wss': metrics['recon'],
                        'mag': metrics['mag'],
                        'sign': metrics['sign'],
                        'phys': self.lambda_phys * phys_loss.item(),
                        'velocity': velocity_loss.item(),
                        'neg_mag': metrics['neg_mag'],
                        'sign_violation': metrics['sign_violation'],
                        'sign_violation_pos': metrics['sign_violation_pos'],
                    },
                    lr=self.optimizer.param_groups[0]['lr'],
                    r2_train=r2.item(),
                    r2_mag_train=r2_mag.item(),
                    sign_acc=metrics['sign_acc'],
                    pos_acc=metrics['pos_acc'],
                    neg_recall=metrics['neg_recall'],
                    neg_precision=metrics['neg_precision'],
                    neg_f1=metrics['neg_f1'],
                    tp=metrics['confusion']['tp'],
                    tn=metrics['confusion']['tn'],
                    fp=metrics['confusion']['fp'],
                    fn=metrics['confusion']['fn'],
                    best_r2=self.best_r2,
                    best_epoch=self.best_epoch,
                    notes=(f"sign_acc={metrics['sign_acc']:.3f},"
                           f"neg_recall={metrics['neg_recall']:.3f},"
                           f"neg_f1={metrics['neg_f1']:.3f},"
                           f"r2_mag={r2_mag.item():.4f}")
                )

                # ====== 关键修改：当velocity_loss > 0时绘制约束违反图 ======
                # 只在第一次检测到velocity_loss > 0时绘制，避免重复生成过多图片
                # if velocity_loss.item() > 0:
                #     violation_ratio = self._plot_velocity_constraint_violation(
                #         self.model, self.dataset, self.save_dir, 
                #         current_epochs + epoch + 1, force_plot=self.force_plot
                #     )
                #     self.force_plot = False  # 绘制一次后重置标志

         # ====== 训练结束后绘制梯度演化图 ======
        print(f"\n{'='*60}")
        print("【绘制梯度演化曲线】")
        print(f"{'='*60}")
        self.plot_gradient_evolution()
        # =============================================
        self.plot_loss_curves()
        print("训练完成!")

    def plot_loss_curves(self, save_path: str = None, window_size: int = 50):
        """
        绘制训练损失曲线和符号分类监控（10子图完整版）
        
        Args:
            save_path: 保存路径，默认为self.save_dir/loss_curves.png
            window_size: 平滑窗口大小
        """
        import os
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'loss_curves.png')
        
        fig = plt.figure(figsize=(18, 20))
        
        colors = {
            'total': '#1f77b4', 'recon': '#2ca02c', 'mag': '#17becf',
            'sign': '#9467bd', 'phys': '#d62728', 'velocity': '#ff7f0e',
            'sign_acc': '#2ca02c', 'neg_recall': '#d62728',
            'neg_precision': '#ff7f0e', 'neg_f1': '#9467bd', 'pos_acc': '#1f77b4', 'neg_mag': '#c7c7c7', 'sign_violation': '#8c564b', 'sign_violation_pos': '#e377c2'
        }
        
        epochs = np.arange(1, len(self.loss_history['total']) + 1)
        n_epochs = len(epochs)            
        
        # 1. 总损失（log尺度）
        ax1 = plt.subplot(5, 2, 1)
        ax1.plot(epochs, self.loss_history['total'], color=colors['total'], lw=1.5, alpha=0.8, label='Total')
        ax1.plot(epochs, self.loss_history['wss'], color=colors['recon'], lw=1.2, alpha=0.7, label='Recon')
        ax1.plot(epochs, self.loss_history['mag'], color=colors['mag'], lw=1.2, alpha=0.7, label='Magnitude')
        ax1.plot(epochs, self.loss_history['sign'], color=colors['sign'], lw=1.2, alpha=0.7, label='Sign')
        ax1.plot(epochs, self.loss_history['phys'], color=colors['phys'], lw=1.2, alpha=0.7, label='Physics')
        ax1.plot(epochs, self.loss_history['velocity'], color=colors['velocity'], lw=1.2, alpha=0.7, label='Velocity')
        ax1.plot(epochs, self.loss_history['neg_mag'], color=colors['neg_mag'], lw=1.2, alpha=0.7, label='Negative Magnitude')
        ax1.plot(epochs, self.loss_history['sign_violation'], color=colors['sign_violation'], lw=1.2, alpha=0.7, label='Sign Violation')
        ax1.plot(epochs, self.loss_history['sign_violation_pos'], color=colors['sign_violation_pos'], lw=1.2, alpha=0.7, label='Sign Violation (Positive)')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (log scale)', fontsize=11)
        ax1.set_yscale('log')
        ax1.set_title('Training Loss Curves (All Components)', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3, which='both')
        
        final_text = (f"Final: Total={self.loss_history['total'][-1]:.2e} | "
                    f"Recon={self.loss_history['wss'][-1]:.2e} | "
                    f"Mag={self.loss_history['mag'][-1]:.2e} | "
                    f"Sign={self.loss_history['sign'][-1]:.2e} | "
                    f"NegMag={self.loss_history['neg_mag'][-1]:.2e} | "
                    f"SignViolation={self.loss_history['sign_violation'][-1]:.2e}")
        ax1.text(0.02, 0.98, final_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        
        # 2. 平滑损失（线性尺度）
        ax2 = plt.subplot(5, 2, 2)
        ax2.plot(epochs, self.loss_history['wss'], color=colors['recon'], lw=2, label='Recon')
        ax2.plot(epochs, self.loss_history['mag'], color=colors['mag'], lw=2, label='Magnitude')
        ax2.plot(epochs, self.loss_history['sign'], color=colors['sign'], lw=2, label='Sign')
        ax2.plot(epochs, self.loss_history['phys'], color=colors['phys'], lw=2, label='Physics')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss (smoothed)', fontsize=11)
        ax2.set_title(f'Smoothed Loss Curves (epoch MA)', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. 符号分类准确率
        ax3 = plt.subplot(5, 2, 3)
        ax3.plot(epochs, self.loss_history['sign_acc'], color=colors['sign_acc'], lw=1.5, alpha=0.8, label='Overall Sign Acc')
        ax3.axhline(y=0.5, color='gray', linestyle='--', lw=1, label='Random')
        ax3.axhline(y=0.9, color='green', linestyle=':', lw=1, alpha=0.5, label='Target (90%)')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Accuracy', fontsize=11)
        ax3.set_ylim([0.4, 1.0])
        ax3.set_title('Sign Classification Accuracy', fontsize=13, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        acc_text = (f"Final: Sign-Acc={self.loss_history['sign_acc'][-1]:.3f}")
        ax3.text(0.02, 0.98, acc_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
        
        # 4. 负样本性能（召回率/精确率/F1）
        ax4 = plt.subplot(5, 2, 4)
        ax4.plot(epochs, self.loss_history['neg_recall'], color=colors['neg_recall'], lw=2, label='Neg Recall')
        ax4.plot(epochs, self.loss_history['neg_precision'], color=colors['neg_precision'], lw=2, label='Neg Precision')
        ax4.plot(epochs, self.loss_history['neg_f1'], color=colors['neg_f1'], lw=2.5, label='Neg F1')
        ax4.axhline(y=0.5, color='gray', linestyle='--', lw=1, label='Random')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Score', fontsize=11)
        ax4.set_ylim([0, 1.0])
        ax4.set_title('Negative WSS Classification Performance', fontsize=13, fontweight='bold')
        ax4.legend(loc='lower right', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        neg_text = (f"Final: Recall={self.loss_history['neg_recall'][-1]:.3f} | "
                    f"Precision={self.loss_history['neg_precision'][-1]:.3f} | "
                    f"F1={self.loss_history['neg_f1'][-1]:.3f}")
        ax4.text(0.02, 0.98, neg_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))
        
        # 5. 学习率曲线
        ax5 = plt.subplot(5, 2, 5)
        # 从metrics_logger或optimizer提取历史lr
        lr_history = [self.optimizer.param_groups[0]['lr']] * n_epochs  # 简化，实际应记录历史
        ax5.plot(epochs, lr_history, color='#e377c2', lw=2)
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel('Learning Rate', fontsize=11)
        ax5.set_yscale('log')
        ax5.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, which='both')
        
        # 6. R² 曲线（如果有记录）
        ax6 = plt.subplot(5, 2, 6)
        if 'r2_train' in self.loss_history and len(self.loss_history['r2_train']) > 0:
            r2_train = self.loss_history['r2_train']
            ax6.plot(epochs[:len(r2_train)], r2_train, color=colors['total'], lw=2, label='Train R²')
            ax6.axhline(y=0.8, color='green', linestyle=':', lw=1, alpha=0.5, label='Target (0.8)')
            ax6.set_ylim([0, 1.0])
            ax6.legend(loc='lower right', fontsize=9)
        else:
            ax6.text(0.5, 0.5, 'R² not recorded\\n(add to loss_history in train())',
                    transform=ax6.transAxes, ha='center', va='center', fontsize=12)
        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('R²', fontsize=11)
        ax6.set_title('R² Evolution', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. 损失占比饼图
        ax7 = plt.subplot(5, 2, 7)
        loss_components = [self.loss_history['wss'][-1], self.loss_history['mag'][-1],
                        self.loss_history['sign'][-1], self.loss_history['phys'][-1],
                        self.loss_history['velocity'][-1]]
        loss_labels = ['Recon', 'Magnitude', 'Sign', 'Physics', 'Velocity']
        loss_colors = [colors['recon'], colors['mag'], colors['sign'], colors['phys'], colors['velocity']]
        wedges, texts, autotexts = ax7.pie(loss_components, labels=loss_labels, colors=loss_colors,
                                            autopct='%1.1f%%', startangle=90,
                                            explode=(0.05, 0.05, 0.1, 0, 0))
        ax7.set_title(f'Loss Composition (Final Epoch)', fontsize=13, fontweight='bold')
        
        # 8. 混淆矩阵元素演化（如果有记录）
        ax8 = plt.subplot(5, 2, 8)
        if 'tp' in self.loss_history and len(self.loss_history['tp']) > 0:
            ax8.stackplot(epochs[:len(self.loss_history['tp'])],
                        self.loss_history['tp'], self.loss_history['tn'],
                        self.loss_history['fp'], self.loss_history['fn'],
                        labels=['TP', 'TN', 'FP', 'FN'],
                        colors=['#2ca02c', '#17becf', '#ff7f0e', '#d62728'], alpha=0.7)
            ax8.legend(loc='upper left', fontsize=9)
        else:
            ax8.text(0.5, 0.5, 'Confusion data not recorded\\n(add tp/tn/fp/fn to loss_history)',
                    transform=ax8.transAxes, ha='center', va='center', fontsize=12)
        ax8.set_xlabel('Epoch', fontsize=11)
        ax8.set_ylabel('Count', fontsize=11)
        ax8.set_title('Confusion Matrix Evolution', fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 9. 梯度重要性演化（如果有记录）
        ax9 = plt.subplot(5, 2, 9)
        if len(self.gradient_history.get('v1', [])) > 0:
            grad_epochs = np.arange(len(self.gradient_history['v1'])) * self.gradient_record_interval
            ax9.plot(grad_epochs, self.gradient_history['v1'], color=colors['recon'], lw=2, label='v1 gradient')
            ax9.plot(grad_epochs, self.gradient_history['d1'], color=colors['mag'], lw=2, label='d1 gradient')
            if 'sep_inflection' in self.gradient_history:
                ax9.plot(grad_epochs, self.gradient_history['sep_inflection'], color=colors['sign'], lw=2, label='sep features')
            ax9.legend(loc='upper right', fontsize=9)
        else:
            ax9.text(0.5, 0.5, 'Gradient history empty\\n(run _record_gradients first)',
                    transform=ax9.transAxes, ha='center', va='center', fontsize=12)
        ax9.set_xlabel('Epoch', fontsize=11)
        ax9.set_ylabel('Mean |Gradient|', fontsize=11)
        ax9.set_title('Feature Gradient Importance Evolution', fontsize=13, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        # 10. 综合监控面板
        ax10 = plt.subplot(5, 2, 10)
        ax10.axis('off')
        
        # 构建信息文本
        info_lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║           WSS Prediction Training Summary (Final)            ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║  Loss Metrics                                                ║",
            f"║    Total Loss:        {self.loss_history['total'][-1]:.4e}                              ║",
            f"║    Reconstruction:    {self.loss_history['wss'][-1]:.4e}                              ║",
            f"║    Magnitude:         {self.loss_history['mag'][-1]:.4e}                              ║",
            f"║    Sign:              {self.loss_history['sign'][-1]:.4e}                              ║",
            f"║    Negative Magnitude: {self.loss_history['neg_mag'][-1]:.4e}                              ║",
            f"║    Sign Violation:   {self.loss_history['sign_violation'][-1]:.4e}                              ║",
            f"║    Sign Violation (Positive): {self.loss_history['sign_violation_pos'][-1]:.4e}                              ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║  Sign Classification                                         ║",
            f"║    Overall Accuracy:  {self.loss_history['sign_acc'][-1]:.3f}                                   ║",
            f"║    Negative Recall:   {self.loss_history['neg_recall'][-1]:.3f}  ← Key Metric                    ║",
            f"║    Negative Prec:    {self.loss_history['neg_precision'][-1]:.3f}                                   ║",
            f"║    Negative F1:      {self.loss_history['neg_f1'][-1]:.3f}                                   ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║  Training Status                                             ║",
            f"║    Best R²:           {self.best_r2:.4f}  @ Epoch {self.best_epoch}                            ║",
            f"║    Physics Penalty:   {self.loss_history['phys'][-1]:.4e}                              ║",
            f"║    Velocity Constraint: {self.loss_history['velocity'][-1]:.4e}                            ║",
            "╚══════════════════════════════════════════════════════════════╝"
        ]
        info_text = '\n'.join(info_lines)
        
        ax10.text(0.5, 0.5, info_text, transform=ax10.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('WSS Prediction Training: Loss & Sign Classification Monitoring', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 损失曲线图已保存: {save_path}")
        plt.close(fig)