"""工具函数：备份、日志、梯度裁剪等"""

import os
import sys
import shutil
import csv
from datetime import datetime
import torch
import numpy as np
from typing import Dict, Any

from config import DEVICE, SAVE_DIR


# ==================== 脚本备份函数 ====================
def backup_model(backup_dir: str, script_files: str = None, script_dir: str = None) -> str:
    """备份项目所有代码文件到指定目录"""
    if script_files is None:
        script_files = [
            'config.py', 'dataset.py', 'models.py', 'trainer.py',
            'evaluator.py', 'losses.py', 'utils.py', 'main.py',
            'data_splitter.py', 'feature_analyzer.py', 'baseline_ml_trainer.py'
        ]
    
    # 创建时间戳子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = os.path.join(backup_dir, f"backup_{timestamp}")
    os.makedirs(backup_subdir, exist_ok=True)
    
    for script_path in script_files:
        path = os.path.join(script_dir, script_path)
        shutil.copy2(path, backup_subdir)
    
    return backup_subdir


# ==================== 梯度裁剪器 ====================
class AdaptiveGradientClipper:
    """
    基于梯度统计量的自适应裁剪器
    根据梯度分布动态调整裁剪阈值
    """
    
    def __init__(self, clip_factor: float = 0.01, history_size: int = 10):
        self.clip_factor = clip_factor
        self.history_size = history_size
        self.grad_norm_history = []
        
    def __call__(self, parameters) -> float:
        """执行梯度裁剪并返回阈值"""
        grads = []
        for p in parameters:
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        
        if not grads:
            return 1.0
        
        all_grads = torch.cat(grads)
        
        # 计算统计量
        grad_mean = all_grads.abs().mean().item()
        grad_std = all_grads.std().item()
        grad_max = all_grads.abs().max().item()
        
        # 自适应阈值
        threshold = max(grad_mean, grad_std * 3) * self.clip_factor
        threshold = max(threshold, 1e-6)
        threshold = min(threshold, grad_max * 0.5)
        
        # 记录历史
        self.grad_norm_history.append(threshold)
        if len(self.grad_norm_history) > self.history_size:
            self.grad_norm_history.pop(0)
        
        # 执行裁剪
        torch.nn.utils.clip_grad_norm_(parameters, threshold)
        
        return threshold


# ==================== 训练指标记录器 ====================
class MetricsLogger:
    """训练指标记录器，保存到CSV（完整的论文实验指标）"""

    COLUMNS = [
        'timestamp', 'epoch',
        # 损失分量
        'loss_total', 'loss_wss', 'loss_mag', 'loss_sign',
        'loss_phys', 'loss_velocity',
        'loss_neg_mag', 'loss_sign_violation', 'loss_sign_violation_pos',
        # 符号分类指标（训练集）
        'sign_acc', 'pos_acc', 'neg_recall', 'neg_precision', 'neg_f1',
        # 混淆矩阵（训练集）
        'tp', 'tn', 'fp', 'fn',
        # 符号分类指标（验证集）
        'sign_acc_val', 'pos_acc_val', 'neg_recall_val', 'neg_precision_val', 'neg_f1_val',
        # 混淆矩阵（验证集）
        'tp_val', 'tn_val', 'fp_val', 'fn_val',
        # R²指标（幅值R²区分train/val，用于论文对比实验和消融实验）
        'r2_train', 'r2_mag_train', 'r2_val', 'r2_mag_val',
        # 其他
        'lr', 'mae_val', 'mape_val', 'r2_spatial_val',
        'best_val_r2', 'best_epoch',
        'early_stopped', 'notes'
    ]

    def __init__(self, save_dir: str, filename: str = 'training_metrics.csv'):
        self.save_path = os.path.join(save_dir, filename)
        self.file_exists = os.path.exists(self.save_path)

        if not self.file_exists:
            with open(self.save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
            print(f"创建指标日志: {self.save_path}")

    def log(self, **kwargs):
        """记录一行数据"""
        row = {col: kwargs.get(col, '') for col in self.COLUMNS}
        row['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.save_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)

    def log_epoch(self, epoch: int, losses: Dict[str, float], lr: float,
                  r2_train: float = None, r2_mag_train: float = None,
                  r2_val: float = None, r2_mag_val: float = None,
                  mae_val: float = None, mape_val: float = None,
                  sign_acc: float = None, pos_acc: float = None,
                  neg_recall: float = None, neg_precision: float = None,
                  neg_f1: float = None,
                  tp: int = None, tn: int = None,
                  fp: int = None, fn: int = None,
                  best_r2: float = None, best_epoch: int = None,
                  early_stopped: bool = False, notes: str = ''):
        """便捷方法：记录epoch信息（完整指标）"""
        self.log(
            epoch=epoch,
            loss_total=losses.get('total', ''),
            loss_wss=losses.get('wss', ''),
            loss_mag=losses.get('mag', ''),
            loss_sign=losses.get('sign', ''),
            loss_phys=losses.get('phys', ''),
            loss_velocity=losses.get('velocity', ''),
            loss_neg_mag=losses.get('neg_mag', ''),
            loss_sign_violation=losses.get('sign_violation', ''),
            loss_sign_violation_pos=losses.get('sign_violation_pos', ''),
            sign_acc=sign_acc if sign_acc is not None else '',
            pos_acc=pos_acc if pos_acc is not None else '',
            neg_recall=neg_recall if neg_recall is not None else '',
            neg_precision=neg_precision if neg_precision is not None else '',
            neg_f1=neg_f1 if neg_f1 is not None else '',
            tp=tp if tp is not None else '',
            tn=tn if tn is not None else '',
            fp=fp if fp is not None else '',
            fn=fn if fn is not None else '',
            lr=lr,
            r2_train=r2_train if r2_train is not None else '',
            r2_mag_train=r2_mag_train if r2_mag_train is not None else '',
            r2_val=r2_val if r2_val is not None else '',
            r2_mag_val=r2_mag_val if r2_mag_val is not None else '',
            mae_val=mae_val if mae_val is not None else '',
            mape_val=mape_val if mape_val is not None else '',
            best_val_r2=best_r2 if best_r2 is not None else '',
            best_epoch=best_epoch if best_epoch is not None else '',
            early_stopped=early_stopped,
            notes=notes
        )

    def update_val_metrics(self, epoch: int, r2_val: float, r2_mag_val: float,
                           mae_val: float = None, mape_val: float = None,
                           r2_spatial_val: float = None,
                           sign_acc_val: float = None, pos_acc_val: float = None,
                           neg_recall_val: float = None, neg_precision_val: float = None,
                           neg_f1_val: float = None,
                           tp_val: int = None, tn_val: int = None,
                           fp_val: int = None, fn_val: int = None):
        """将验证指标合并到指定 epoch 的训练行中，不新增行"""
        import csv as _csv

        rows = []
        with open(self.save_path, 'r', newline='', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)

        updated = False
        for row in rows:
            if str(row.get('epoch', '')) == str(epoch) and row.get('r2_train', '').strip() != '':
                row['r2_val'] = f'{r2_val:.6f}' if r2_val is not None else ''
                row['r2_mag_val'] = f'{r2_mag_val:.6f}' if r2_mag_val is not None else ''
                if mae_val is not None:
                    row['mae_val'] = f'{mae_val:.6f}'
                if mape_val is not None:
                    row['mape_val'] = f'{mape_val:.6f}'
                if r2_spatial_val is not None:
                    row['r2_spatial_val'] = f'{r2_spatial_val:.6f}'
                if sign_acc_val is not None:
                    row['sign_acc_val'] = f'{sign_acc_val:.6f}'
                if pos_acc_val is not None:
                    row['pos_acc_val'] = f'{pos_acc_val:.6f}'
                if neg_recall_val is not None:
                    row['neg_recall_val'] = f'{neg_recall_val:.6f}'
                if neg_precision_val is not None:
                    row['neg_precision_val'] = f'{neg_precision_val:.6f}'
                if neg_f1_val is not None:
                    row['neg_f1_val'] = f'{neg_f1_val:.6f}'
                if tp_val is not None:
                    row['tp_val'] = str(tp_val)
                if tn_val is not None:
                    row['tn_val'] = str(tn_val)
                if fp_val is not None:
                    row['fp_val'] = str(fp_val)
                if fn_val is not None:
                    row['fn_val'] = str(fn_val)
                updated = True
                break

        if updated:
            with open(self.save_path, 'w', newline='', encoding='utf-8') as f:
                writer = _csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
                writer.writerows(rows)


# ==================== 早停器 ====================
class EarlyStoppingTrainer:
    """早停训练管理器"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.best_state_dict = None

    def save_checkpoint(self, epoch: int, model: torch.nn.Module):
        """保存最佳模型（深度拷贝：state_dict 的 tensor 是共享存储的引用，
        不 clone 会被后续训练原地更新污染）"""
        self.best_state_dict = {
            'epoch': epoch,
            'model_state_dict': {k: v.detach().clone()
                                 for k, v in model.state_dict().items()},
        }
    
    def check_early_stop(self, epoch: int, val_r2: float, model: torch.nn.Module) -> bool:
        """检查是否应该早停"""
        score = val_r2
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, model)
            return False
        
        # 判断是否有提升
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            self.save_checkpoint(epoch, model)
            print(f"  ✓ 新的最佳验证R²: {score:.4f} @ Epoch {epoch}")
            return False
        else:
            self.counter += 1
            print(f"  未提升 {self.counter}/{self.patience}, 最佳={self.best_score:.4f}@{self.best_epoch}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n{'='*50}")
                print(f"早停触发！最佳验证R²={self.best_score:.4f} @ Epoch {self.best_epoch}")
                print(f"{'='*50}")
                return True
        
        return False