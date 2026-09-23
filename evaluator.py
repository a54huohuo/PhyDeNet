"""评估和可视化"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score

from config import DEVICE


class DirectWSSEvaluator:
    """WSS预测评估器"""
    
    def __init__(self, model, dataset):
        self.model = model.to(DEVICE)
        self.dataset = dataset
        self.features_characteristics = dataset.features_characteristics
        self.model.eval()

    def _compute_spatial_gradient_r2(self, wss_pred: np.ndarray,
                                       wss_true: np.ndarray,
                                       k_neighbors: int = 6) -> float:
        """
        计算空间梯度 R²：衡量预测 WSS 场的空间光滑性是否匹配真实场。

        对每个壁面点找 k 近邻，构建边集，计算每条边上的 ∂WSS/∂s：
          - 真实梯度: (wss_true[j] - wss_true[i]) / ||x_j - x_i||
          - 预测梯度: (wss_pred[j] - wss_pred[i]) / ||x_j - x_i||
        然后对 N_edges 个梯度值计算 R²。

        Returns:
            r2_spatial: 空间梯度 R²（与 GradientConstraint 训练损失一致）
        """
        coords = self.dataset.wall_coords
        n_points = len(coords)

        tree = cKDTree(coords)
        _, indices = tree.query(coords, k=k_neighbors + 1)  # +1: 排除自身

        true_grads = []
        pred_grads = []

        for i in range(n_points):
            for idx in range(1, k_neighbors + 1):
                j = indices[i, idx]
                dist = np.linalg.norm(coords[j] - coords[i])
                if dist < 1e-8:
                    continue
                d_true = (wss_true[j] - wss_true[i]) / dist
                d_pred = (wss_pred[j] - wss_pred[i]) / dist
                true_grads.append(d_true)
                pred_grads.append(d_pred)

        if len(true_grads) < 2:
            return float('nan')

        true_grads = np.array(true_grads)
        pred_grads = np.array(pred_grads)
        return float(r2_score(true_grads, pred_grads))

    def evaluate(self):
        """评估壁面WSS预测，返回回归指标 + 符号分类指标 + 空间梯度 R²"""
        # 组合点云模型（PointNet++/DGCNN/GAT）使用壁面+内部点云，
        # 通过 k-NN 图从内部流场传播信息
        # PointNet (vanilla) 有 set_coords() 但仅用壁面点，走标准路径
        is_gnn = getattr(self.model, '_model_type', None) in (
            'pointnetpp', 'dgcnn', 'gat', 'pointnet_combined')

        if is_gnn:
            tensors = self.dataset.get_combined_tensors()
            self.model.set_coords(tensors['combined_coords'])
            wall_features = tensors['combined_features']
            wall_velocity_real = tensors.get('wall_velocity')
            wall_pressure_real = tensors.get('wall_pressure')
            wall_mask = tensors['wall_mask']
        else:
            tensors = self.dataset.get_tensors()
            # PointNet (vanilla) / 其他有 set_coords 的轻量模型需要壁面坐标
            if hasattr(self.model, 'set_coords'):
                wall_coords = tensors.get('wall_coords')
                if wall_coords is not None:
                    self.model.set_coords(wall_coords)
            wall_features = tensors['wall_features']
            wall_velocity_real = tensors['wall_velocity']
            wall_pressure_real = tensors['wall_pressure']
            wall_mask = None

        with torch.no_grad():
            if is_gnn:
                wss_pred, magnitude, sign_prob = self.model.forward_coupled(
                    wall_features, wall_velocity_real, wall_pressure_real,
                    wall_mask=wall_mask)
            else:
                wss_pred, magnitude, sign_prob = self.model.forward_coupled(
                    wall_features, wall_velocity_real, wall_pressure_real)

        # 反标准化
        wss_pred = wss_pred.cpu().numpy() * self.features_characteristics['wss']
        wss_true = self.dataset.wall_wss * self.features_characteristics['wss']

        # 回归指标
        mae = np.mean(np.abs(wss_pred - wss_true))
        mape = np.mean(np.abs((wss_pred - wss_true) / (wss_true + 1e-8))) * 100
        r2 = 1 - np.sum((wss_true - wss_pred)**2) / \
             np.sum((wss_true - np.mean(wss_true))**2)
        r2_mag = 1 - np.sum((np.abs(wss_true) - np.abs(wss_pred))**2) / \
                 np.sum((np.abs(wss_true) - np.mean(np.abs(wss_true)))**2)

        # 符号分类指标
        sign_prob_np = sign_prob.cpu().numpy().flatten()
        sign_true = (wss_true > 0).flatten()
        sign_pred = (sign_prob_np > 0.5)

        sign_acc = float(np.mean(sign_pred == sign_true))
        pos_mask = sign_true
        pos_acc = float(np.mean(sign_pred[pos_mask] == sign_true[pos_mask])) if pos_mask.sum() > 0 else 0.0
        neg_mask = ~sign_true
        # 2026-09-15 修正：val 无真负点的 case 对召回率零信息量，记 NaN（原 1.0 会
        # 制造满分伪影，见 canonical 重算 within_case_all_metrics.csv）；
        # 无负预测时 precision 同理记 NaN（F1 由下方 guard 自然落到 0）
        neg_recall = float(np.mean(sign_pred[neg_mask] == sign_true[neg_mask])) if neg_mask.sum() > 0 else float('nan')
        neg_pred_mask = ~sign_pred
        neg_precision = float(np.mean(sign_pred[neg_pred_mask] == sign_true[neg_pred_mask])) if neg_pred_mask.sum() > 0 else float('nan')
        if not np.isfinite(neg_recall):
            neg_f1 = float('nan')          # 空真负类：F1 同为缺失
        elif not np.isfinite(neg_precision) or (neg_recall + neg_precision) <= 0:
            neg_f1 = 0.0                   # 模型无负预测：F1=0（与 canonical 口径一致）
        else:
            neg_f1 = 2 * neg_recall * neg_precision / (neg_recall + neg_precision)

        tp = int(np.sum(sign_true & sign_pred))
        tn = int(np.sum((~sign_true) & (~sign_pred)))
        fp = int(np.sum((~sign_true) & sign_pred))
        fn = int(np.sum(sign_true & (~sign_pred)))

        # 空间梯度 R²（衡量预测 WSS 场的空间光滑性）
        r2_spatial = self._compute_spatial_gradient_r2(
            wss_pred.flatten(), wss_true.flatten()
        )

        print("\nWSS Evaluation:")
        print(f"  MAE:  {mae:.4f}  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}  R²_mag: {r2_mag:.4f}")
        print(f"  R²_spatial (∂WSS/∂s): {r2_spatial:.4f}")
        print(f"  Sign-Acc: {sign_acc:.3f} | Pos-Acc: {pos_acc:.3f} | "
              f"Neg-Recall: {neg_recall:.3f} | Neg-Prec: {neg_precision:.3f} | Neg-F1: {neg_f1:.3f}")
        print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

        return (wss_true, wss_pred,
                r2, r2_mag, mae, mape, r2_spatial,
                sign_acc, pos_acc, neg_recall, neg_precision, neg_f1,
                tp, tn, fp, fn)

    def plot_predictions(self, epoch_num: int, epoch_interval: int,
                        dataset_type: str = 'train', save_dir: str = None):
        """绘制预测结果，返回 (r2, r2_mag, mae, mape, sign_acc, pos_acc,
           neg_recall, neg_precision, neg_f1, tp, tn, fp, fn)"""
        (wss_true, wss_pred,
         r2, r2_mag, mae, mape, r2_spatial,
         sign_acc, pos_acc, neg_recall, neg_precision, neg_f1,
         tp, tn, fp, fn) = self.evaluate()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 2].axis('off')
        
        errors = wss_pred - wss_true
        
        # 1. 散点图
        ax = axes[0, 0]
        ax.scatter(wss_true, wss_pred, alpha=0.5, s=10, c='blue')
        ax.plot([wss_true.min(), wss_true.max()], 
                [wss_true.min(), wss_true.max()], 
                'r--', label='Ideal', linewidth=2)
        ax.set_xlabel('True WSS')
        ax.set_ylabel('Predicted WSS')
        ax.set_title(f'WSS Prediction (R²={r2:.4f}, R²_mag={r2_mag:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 误差分布
        ax = axes[0, 1]
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        
        # 3-5. 空间分布
        coords = self.dataset.wall_coords
        planes = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]
        
        for idx, (i, j, xlabel, ylabel) in enumerate(planes):
            ax = axes[1, idx]
            sc = ax.scatter(coords[:, i], coords[:, j], 
                           c=errors.flatten(), cmap='RdBu_r', 
                           s=20, alpha=0.6)
            plt.colorbar(sc, ax=ax, label='Error')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'Error Spatial ({xlabel}{ylabel})')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = (f'{save_dir}/wss_prediction_results_{dataset_type}_'
                        f'{epoch_num-epoch_interval}_{epoch_num}.png')
            plt.savefig(save_path, dpi=150)
            print(f"\nSaved: {save_path}")

        plt.close(fig)
        return (r2, r2_mag, mae, mape, r2_spatial,
                sign_acc, pos_acc, neg_recall, neg_precision, neg_f1,
                tp, tn, fp, fn)