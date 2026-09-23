"""XGBoost / Random Forest 基线训练器（sklearn）"""

import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# ==================== CSV 指标记录器（独立版，与 utils.MetricsLogger 兼容） ====================

ML_COLUMNS = [
    'timestamp', 'run_id',
    'model_type', 'n_estimators', 'max_depth',
    'r2_train', 'r2_mag_train', 'mae_train',
    'sign_acc_train', 'neg_recall_train', 'neg_precision_train', 'neg_f1_train',
    'r2_val', 'r2_mag_val', 'mae_val',
    'sign_acc_val', 'neg_recall_val', 'neg_precision_val', 'neg_f1_val',
    'notes'
]


class MLMetricsLogger:
    def __init__(self, save_dir: str, filename: str = 'training_metrics.csv'):
        self.save_path = os.path.join(save_dir, filename)
        self.file_exists = os.path.exists(self.save_path)
        if not self.file_exists:
            with open(self.save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=ML_COLUMNS)
                writer.writeheader()

    def log(self, **kwargs):
        row = {col: kwargs.get(col, '') for col in ML_COLUMNS}
        row['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.save_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ML_COLUMNS)
            writer.writerow(row)


# ==================== 指标计算 ====================

def compute_sign_metrics(wss_true, wss_pred):
    """计算符号分类指标"""
    sign_true = (wss_true > 0).flatten()
    sign_pred = (wss_pred > 0).flatten()

    sign_acc = float(np.mean(sign_pred == sign_true))

    pos_mask = sign_true
    pos_acc = (float(np.mean(sign_pred[pos_mask] == sign_true[pos_mask]))
               if pos_mask.sum() > 0 else 0.0)

    neg_mask = ~sign_true
    neg_recall = (float(np.mean(sign_pred[neg_mask] == sign_true[neg_mask]))
                  if neg_mask.sum() > 0 else 1.0)

    neg_pred_mask = ~sign_pred
    neg_precision = (float(np.mean(sign_pred[neg_pred_mask] == sign_true[neg_pred_mask]))
                     if neg_pred_mask.sum() > 0 else 0.0)

    neg_f1 = (2 * neg_recall * neg_precision / (neg_recall + neg_precision)
              if (neg_recall + neg_precision) > 0 else 0.0)

    return sign_acc, pos_acc, neg_recall, neg_precision, neg_f1


# ==================== 训练入口 ====================

def train_ml_baseline(train_dataset, val_dataset, save_dir: str,
                      algorithm: str = 'xgboost',
                      n_estimators: int = 500, max_depth: int = 10):
    """
    使用 sklearn XGBoost / RandomForest 训练 WSS 预测模型

    Args:
        train_dataset: 训练集 CarotidDataset
        val_dataset: 验证集 CarotidDataset
        save_dir: 保存目录
        algorithm: 'xgboost' 或 'random_forest'
        n_estimators: 树的数量
        max_depth: 最大深度
    """
    logger = MLMetricsLogger(save_dir)

    # 提取特征和标签
    X_train = train_dataset.wall_features_scaled
    y_train = train_dataset.wall_wss.flatten()       # 归一化空间

    X_val = val_dataset.wall_features_scaled
    y_val = val_dataset.wall_wss.flatten()           # 归一化空间

    # WSS 特征尺度（用于将归一化 MAE 还原到物理单位 Pa）
    wss_scale = train_dataset.features_characteristics.get('wss', 1.0)
    # 安全性校验：wss_scale 必须是合理的物理值（> 0.01 Pa），不能是 1.0（归一化空间）
    if wss_scale < 1.0:
        print(f"  ⚠️  WARNING: wss_scale={wss_scale:.6f} 异常小！MAE 可能未转换到物理 Pa。")
        print(f"      检查 _compute_characteristic_scales 是否正确计算。")
        print(f"      将使用 wss_scale={wss_scale} 进行转换（可能导致 MAE 量级异常）。")

    print(f"\n{'='*60}")
    print(f"【{algorithm.upper()} 基线训练】")
    print(f"  训练集: {len(X_train)} 点, 特征维度: {X_train.shape[1]}")
    print(f"  验证集: {len(X_val)} 点")
    print(f"  WSS 特征尺度 (τ_95): {wss_scale:.2f} Pa")
    print(f"  ⚡ MAE 转换: MAE_phys(Pa) = MAE_norm × τ_95 = MAE_norm × {wss_scale:.2f}")
    print(f"{'='*60}")

    # 创建模型
    if algorithm == 'xgboost':
        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("⚠️ xgboost 未安装，回退到 RandomForest")
            algorithm = 'random_forest'

    if algorithm == 'xgboost':
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',  # MSE — 与 GNN 黑盒对比统一
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='squared_error',  # MSE — 与 GNN 黑盒对比统一
            random_state=42,
            n_jobs=-1,
        )

    # 训练
    print(f"\n  训练中...")
    model.fit(X_train, y_train)

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # 训练集指标（R² 和符号指标是尺度不变的；MAE 还原到物理 Pa）
    r2_train = r2_score(y_train, y_pred_train)
    r2_mag_train = r2_score(np.abs(y_train), np.abs(y_pred_train))
    mae_train_norm = mean_absolute_error(y_train, y_pred_train)
    mae_train = mae_train_norm * wss_scale  # 还原到物理 Pa
    sign_acc, pos_acc, nr, np_, nf1 = compute_sign_metrics(y_train, y_pred_train)

    print(f"\n  【训练集】 R²={r2_train:.4f}  R²_mag={r2_mag_train:.4f}  MAE={mae_train:.4f} Pa")
    print(f"    Sign-Acc={sign_acc:.3f}  Neg-Recall={nr:.3f}  Neg-F1={nf1:.3f}")

    # 验证集指标
    r2_val = r2_score(y_val, y_pred_val)
    r2_mag_val = r2_score(np.abs(y_val), np.abs(y_pred_val))
    mae_val_norm = mean_absolute_error(y_val, y_pred_val)
    mae_val = mae_val_norm * wss_scale  # 还原到物理 Pa
    sign_acc_v, pos_acc_v, nr_v, np_v, nf1_v = compute_sign_metrics(y_val, y_pred_val)

    print(f"\n  【验证集】 R²={r2_val:.4f}  R²_mag={r2_mag_val:.4f}  MAE={mae_val:.4f} Pa")
    print(f"    Sign-Acc={sign_acc_v:.3f}  Neg-Recall={nr_v:.3f}  Neg-F1={nf1_v:.3f}")

    # 记录到 CSV（MAE 已还原到物理单位 Pa，与 NN evaluator 一致）
    logger.log(
        run_id=os.path.basename(save_dir),
        model_type=algorithm,
        n_estimators=n_estimators,
        max_depth=max_depth,
        r2_train=r2_train, r2_mag_train=r2_mag_train, mae_train=mae_train,
        sign_acc_train=sign_acc, neg_recall_train=nr,
        neg_precision_train=np_, neg_f1_train=nf1,
        r2_val=r2_val, r2_mag_val=r2_mag_val, mae_val=mae_val,
        sign_acc_val=sign_acc_v, neg_recall_val=nr_v,
        neg_precision_val=np_v, neg_f1_val=nf1_v,
        notes=(f"features={X_train.shape[1]}d, "
               f"wss_scale={wss_scale:.2f}Pa, "
               f"mae_norm={mae_val_norm:.6f}→mae_phys={mae_val:.4f}Pa")
    )

    # 保存预测结果
    np.savez(os.path.join(save_dir, f'{algorithm}_predictions.npz'),
             y_true_train=y_train, y_pred_train=y_pred_train,
             y_true_val=y_val, y_pred_val=y_pred_val)

    # 完整比较图：散点图 + 三视图空间误差（与 NN evaluator 格式一致）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 还原到物理单位
        y_train_pa = y_train * wss_scale
        y_pred_train_pa = y_pred_train * wss_scale
        y_val_pa = y_val * wss_scale
        y_pred_val_pa = y_pred_val * wss_scale

        val_errors = y_pred_val_pa - y_val_pa

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 2].axis('off')  # 右上角留空

        # ── Row 0, Col 0: Train scatter ──
        ax = axes[0, 0]
        ax.scatter(y_train_pa, y_pred_train_pa, alpha=0.5, s=10, c='blue')
        ax.plot([y_train_pa.min(), y_train_pa.max()],
                [y_train_pa.min(), y_train_pa.max()], 'r--', lw=2)
        ax.set_xlabel('True WSS (Pa)'); ax.set_ylabel('Predicted WSS (Pa)')
        ax.set_title(f'Train (R²={r2_train:.4f}, MAE={mae_train:.3f} Pa)')
        ax.grid(True, alpha=0.3)

        # ── Row 0, Col 1: Val scatter ──
        ax = axes[0, 1]
        ax.scatter(y_val_pa, y_pred_val_pa, alpha=0.5, s=10, c='green')
        ax.plot([y_val_pa.min(), y_val_pa.max()],
                [y_val_pa.min(), y_val_pa.max()], 'r--', lw=2)
        ax.set_xlabel('True WSS (Pa)'); ax.set_ylabel('Predicted WSS (Pa)')
        ax.set_title(f'Val (R²={r2_val:.4f}, MAE={mae_val:.3f} Pa)')
        ax.grid(True, alpha=0.3)

        # ── Row 1: XY / XZ / YZ spatial error views (validation set) ──
        planes = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]
        val_coords = val_dataset.wall_coords  # (N_val_wall, 3)

        for idx, (i, j, xlabel, ylabel) in enumerate(planes):
            ax = axes[1, idx]
            sc = ax.scatter(val_coords[:, i], val_coords[:, j],
                            c=val_errors.flatten(), cmap='RdBu_r',
                            s=10, alpha=0.5, edgecolors='none')
            plt.colorbar(sc, ax=ax, label='Error (Pa)')
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(f'Error Spatial ({xlabel}{ylabel})')
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{algorithm}_predictions.png'), dpi=150)
        plt.close()
        print(f"  ✓ 图表已保存（含三视图空间误差）")
    except Exception as e:
        print(f"  ⚠️ 无法绘制图表: {e}")

    print(f"\n  ✓ {algorithm.upper()} 训练完成，结果保存在: {save_dir}")
    return r2_val, r2_mag_val
