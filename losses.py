"""损失函数和物理约束"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
import torch.nn.functional as F


class GradientConstraint:
    """
    空间WSS梯度一致性约束
    鼓励相邻点WSS差与几何距离成比例
    """
    
    def __init__(self, weight: float = 0.05, k_neighbors: int = 6):
        self.weight = weight
        self.k = k_neighbors
        self.n_edges = 0
        
    def precompute(self, coords: np.ndarray, wss_true: np.ndarray, device: torch.device):
        """
        预计算边集和真实梯度目标
        
        Args:
            coords: 壁面点坐标 (n_wall, 3)
            wss_true: 真实WSS值
            device: 计算设备
        """
        n_points = len(coords)
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=self.k + 1)
        
        edges_i = []
        edges_j = []
        true_gradients = []
        edge_lengths = []
        
        for i in range(n_points):
            for idx in range(1, self.k + 1):
                j = indices[i, idx]
                dist = distances[i, idx]
                
                if dist > 1e-6:
                    edges_i.append(i)
                    edges_j.append(j)
                    
                    # 真实WSS梯度
                    grad_true = (wss_true[j] - wss_true[i]) / dist
                    true_gradients.append(grad_true)
                    
                    # 边长
                    length = np.linalg.norm(coords[j] - coords[i])
                    edge_lengths.append(length)
        
        self.n_edges = len(edges_i)
        self.edges_i = torch.LongTensor(edges_i).to(device)
        self.edges_j = torch.LongTensor(edges_j).to(device)
        self.wss_true_gradients = torch.FloatTensor(true_gradients).to(device).reshape(-1, 1)
        self.edge_lengths = torch.FloatTensor(edge_lengths).to(device).reshape(-1, 1)
    
    def __call__(self, wss_pred: torch.Tensor) -> torch.Tensor:
        """
        计算梯度一致性损失
        
        Args:
            wss_pred: 预测的WSS [N, 1]
        
        Returns:
            loss: 梯度MSE损失
        """
        wss_i = wss_pred[self.edges_i]
        wss_j = wss_pred[self.edges_j]
        
        # 预测梯度
        wss_diff_pred = wss_j - wss_i
        wss_gradient_pred = wss_diff_pred / (self.edge_lengths + 1e-8)
        
        # 与真实梯度的MSE
        loss = torch.mean((wss_gradient_pred - self.wss_true_gradients) ** 2)
        
        return loss * self.weight


class PhysicsInformedLoss(nn.Module):
    """
    物理启发损失：强制符合近壁衰减规律
    """
    
    def __init__(self, lambda_decay=0.1):
        super().__init__()
        self.lambda_decay = lambda_decay
        
    def forward(self, features, model):
        # 物理损失：梯度衰减一致性（v2的贡献 < v1，v3 < v1）
        # 通过检查模型对v2,v3的输入敏感度实现
        if hasattr(model, 'wss_net'):
            features.requires_grad_(True)
            wss_pred_for_grad = model(features,
                                      torch.zeros_like(features[:, 0:1]),
                                      torch.zeros_like(features[:, 0:1]))
            # create_graph=True 必须开启，否则 loss 脱离计算图，无法反向传播到模型参数
            grads = torch.autograd.grad(wss_pred_for_grad.sum(), features,
                                       create_graph=False)[0]

            # v1的梯度应该明显大于v2，v2大于v3
            grad_v1 = torch.abs(grads[:, 4]).mean()
            grad_v2 = torch.abs(grads[:, 6]).mean()
            grad_v3 = torch.abs(grads[:, 8]).mean()

            # 衰减损失：鼓励 grad_v1 > grad_v2 且 grad_v1 > grad_v3
            decay_loss = torch.relu(grad_v2 - grad_v1) + \
                        torch.relu(grad_v3 - grad_v1)
        else:
            decay_loss = torch.tensor(0.0, device=features.device)

        return self.lambda_decay * decay_loss


class CoupledWSSLoss(nn.Module):
    """
    符号-幅值耦合损失（带完整监控指标）
    
    输出指标：
    - total_loss: 用于反向传播
    - loss_dict: 包含所有分项损失和符号分类指标
    """

    def __init__(self, lambda_recon: float = 1.0,
                 lambda_mag: float = 0.5,
                 lambda_sign: float = 0.5,
                 neg_weight: float = 5.0,
                 neg_mag_weight: float = 2.0,
                 sign_violation_weight: float = 10.0):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_mag = lambda_mag
        self.lambda_sign = lambda_sign
        self.neg_weight = neg_weight
        self.neg_mag_weight = neg_mag_weight
        self.sign_violation_weight = sign_violation_weight

    def forward(self, wss_pred, magnitude, sign_prob, wss_true, neg_thresh=0.50, pos_thresh=0.50, ):
        """
        Args:
            wss_pred: [N,1] 最终WSS
            magnitude: [N,1] |WSS|
            sign_prob: [N,1] P(WSS>0)
            wss_true: [N,1] 标签
        
        Returns:
            total_loss: 标量（用于.backward()）
            metrics: dict（用于日志/监控，不参与梯度）
        """
        # ===== 1. 幅值损失 =====
        mag_loss = F.mse_loss(magnitude, wss_true.abs())

        # ===== 2. 重建损失 =====
        recon_loss = F.mse_loss(wss_pred, wss_true)

        # ===== 3. 符号损失（负样本加权BCE） =====
        # BCE 在 AMP autocast 下数值不稳定，用 autocast(enabled=False) 包裹
        sign_true = (wss_true > 0).float()
        weights = torch.where(
            sign_true > 0.5,
            torch.ones_like(sign_true),
            torch.ones_like(sign_true) * self.neg_weight
        )
        with torch.amp.autocast('cuda', enabled=False):
            sign_loss = F.binary_cross_entropy(
                sign_prob.float(), sign_true.float(), weight=weights.float()
            )

        # 3. 负样本幅值额外加权（解决"符号对了但幅值趋近于0"的问题）
        neg_mask = (wss_true < 0).float()
        pos_mask = (wss_true > 0).float()
        wss_abs_true = wss_true.abs()
        low_mag_thresold = 0.4
        low_pos_mask = pos_mask * (wss_abs_true < low_mag_thresold).float()

        sign_violation_neg = F.softplus(sign_prob - neg_thresh) * neg_mask  # 真负样本的sign_prob必须<neg_thresh
        sign_violation_pos_low = F.softplus(pos_thresh - sign_prob) * low_pos_mask  # 真正正样本且幅值较低的sign_prob必须>pos_thresh
        
        n_neg = neg_mask.sum() + 1e-8
        n_pos_low = low_pos_mask.sum() + 1e-8
        sign_violation_loss = torch.sum(sign_violation_neg) / n_neg 
        sign_violation_loss_pos = torch.sum(sign_violation_pos_low) / n_pos_low # 硬约束

        neg_mag_loss = F.l1_loss(magnitude * neg_mask, wss_true.abs() * neg_mask)

        # ===== 总损失（可反向传播） =====
        total_loss = (self.lambda_recon * recon_loss +
                      self.lambda_mag * mag_loss +
                      self.lambda_sign * sign_loss +
                      self.neg_mag_weight * neg_mag_loss +
                      self.sign_violation_weight * sign_violation_loss +
                      self.sign_violation_weight * sign_violation_loss_pos)

        # ===== 符号分类指标（仅监控，detach） =====
        with torch.no_grad():
            sign_pred = (sign_prob > 0.5).float()
            
            # 整体符号准确率
            sign_acc = (sign_pred == sign_true).float().mean().item()
            
            # 正样本准确率（Precision of positive class）
            pos_mask = (sign_true > 0.5)
            pos_acc = ((sign_pred == sign_true)[pos_mask]).float().mean().item() if pos_mask.sum() > 0 else 0.0
            
            # 负样本召回率（Recall of negative class）—— 关键指标
            neg_mask = (sign_true <= 0.5)
            neg_recall = ((sign_pred == sign_true)[neg_mask]).float().mean().item() if neg_mask.sum() > 0 else 1.0
            
            # 负样本精确率（Precision of negative class）
            neg_pred_mask = (sign_pred <= 0.5)
            neg_precision = ((sign_pred == sign_true)[neg_pred_mask]).float().mean().item() if neg_pred_mask.sum() > 0 else 0.0
            
            # F1分数（负类）
            if neg_recall + neg_precision > 0:
                neg_f1 = 2 * neg_recall * neg_precision / (neg_recall + neg_precision)
            else:
                neg_f1 = 0.0
            
            # 混淆矩阵元素
            tp = ((sign_true > 0.5) & (sign_pred > 0.5)).sum().item()
            tn = ((sign_true <= 0.5) & (sign_pred <= 0.5)).sum().item()
            fp = ((sign_true <= 0.5) & (sign_pred > 0.5)).sum().item()
            fn = ((sign_true > 0.5) & (sign_pred <= 0.5)).sum().item()

        # 返回总损失 + 指标字典
        metrics = {
            'recon': self.lambda_recon * recon_loss.item(),
            'mag': self.lambda_mag * mag_loss.item(),
            'sign': self.lambda_sign * sign_loss.item(),
            'sign_acc': sign_acc,
            'neg_mag': self.neg_mag_weight * neg_mag_loss.item(),
            'pos_acc': pos_acc,
            'neg_recall': neg_recall,
            'neg_precision': neg_precision,
            'neg_f1': neg_f1,
            'sign_violation': self.sign_violation_weight * sign_violation_loss.item(),
            'sign_violation_pos': self.sign_violation_weight * sign_violation_loss_pos.item(),
            'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        }

        return total_loss, metrics