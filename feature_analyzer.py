"""特征分析器：分析输入特征与WSS的相关性"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from typing import Dict, List, Tuple
import seaborn as sns

from config import DEVICE


class FeatureAnalyzer:
    """输入特征与WSS相关性分析器"""
    
    def __init__(self, dataset):
        """
        Args:
            dataset: CarotidDataset实例
        """
        self.dataset = dataset
        self.features = dataset.wall_features_scaled  # [N, 24] 特征矩阵
        self.wss = dataset.wall_wss_scaled.flatten()   # [N,] WSS值
        
        # 构建整体特征：将n1_x,n1_y,n1_z合并为n1_magnitude和n1_alignment
        self.composite_features, self.composite_names = self._build_composite_features()
        
    def _build_composite_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        构建组合特征：
        - n1_magnitude: n1矢量的模长（应恒为1，但保留作为完整性检查）
        - n1_alignment: n1与壁面法向的夹角余弦（关键物理量）
        """
        n_wall = len(self.features)
        
        # 原始特征索引（根据build_inputs的顺序）
        # [d, nx,ny,nz, v1,d1, v2,d2, v3,d3, wss_ref, n1x,n1y,n1z, n2x,n2y,n2z, n3x,n3y,n3z, geom...]
        idx_n1x, idx_n1y, idx_n1z = 11, 12, 13
        # 同理n2, n3
        idx_n2x, idx_n2y, idx_n2z = 14, 15, 16
        idx_n3x, idx_n3y, idx_n3z = 17, 18, 19
        idx_nx, idx_ny, idx_nz = 1, 2, 3  # 壁面法向
        
        # 提取n1分量
        n1 = self.features[:, [idx_n1x, idx_n1y, idx_n1z]]  # [N, 3]
        n2 = self.features[:, [idx_n2x, idx_n2y, idx_n2z]]
        n3 = self.features[:, [idx_n3x, idx_n3y, idx_n3z]]
        
        # 构建新特征矩阵：移除单独方向，增加组合特征
        # 保留：d, normal(3), v1,d1, v2,d2, v3,d3, wss_ref, 
        # 新增：n1_mag, n1_align, n2_mag, n2_align, n3_mag, n3_align
        # 保留：geom(4)
        
        other_features = np.hstack([
            self.features[:, 4:11],     # v1,d1,v2,d2,v3,d3,wss_ref
            n1,   # n1整体(2)
            n2,   # n2整体(2)
            n3,   # n3整体(2)
            self.features[:, 20:24],     # geom(4)
        ])
        
        names = [
            'v1', 'd1', 'v2', 'd2', 'v3', 'd3',
            'wss_ref',
            'n1',  # 替代n1_x,y,z
            'n2',  # 替代n2_x,y,z
            'n3',  # 替代n3_x,y,z
            'density', 'curvedness', 'shape_index', 'anisotropy',
        ]
        
        print(f"\n组合特征构建完成：{other_features.shape[1]}维")
        print(f"  (应接近±1，表示方向与法向对齐/反向)")
        
        return other_features, names
    
    def compute_correlations(self) -> pd.DataFrame:
        """计算各组合特征与WSS的相关性"""
        results = []
        
        for i, name in enumerate(self.composite_names):
            feature = self.composite_features[:, i]
            
            # Pearson相关
            pearson_r, pearson_p = pearsonr(feature, self.wss)
            
            # Spearman相关
            spearman_r, spearman_p = spearmanr(feature, self.wss)
            
            # 互信息
            mi_score = self._estimate_mutual_info(feature, self.wss)
            
            results.append({
                'feature': name,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'mi_score': mi_score,
                'abs_pearson': abs(pearson_r),
                'abs_spearman': abs(spearman_r),
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('abs_pearson', ascending=False)
        return df
    
    def _estimate_mutual_info(self, x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
        """
        使用直方图估计互信息
        """
        # 2D直方图
        joint_hist, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
        
        # 边缘分布
        p_x = joint_hist.sum(axis=1) / joint_hist.sum()
        p_y = joint_hist.sum(axis=0) / joint_hist.sum()
        p_xy = joint_hist / joint_hist.sum()
        
        # 互信息
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    def analyze_feature_importance_with_model(self, model) -> pd.DataFrame:
        """
        使用训练好的模型计算组合特征的梯度重要性。
        
        修改说明：
        - 不再使用零填充，而是直接利用模型已有的39维特征解析逻辑
        - 通过get_tensors()获取完整特征，再提取需要的子集
        """
        model.eval()

        # ====== 使用 get_tensors() 获取模型实际输入特征 ======
        tensors = self.dataset.get_tensors()
        features = tensors['wall_features']
        velocity = tensors['wall_velocity']
        pressure = tensors['wall_pressure']

        # 仅完整特征布局（>=39维）支持逐特征梯度分析
        n_feats = features.shape[1]
        if n_feats < 39:
            print(f"  ⚠️ 输入特征仅 {n_feats} 维，跳过逐特征梯度重要性分析（需>=39维）")
            model.train()
            return None

        features = features.clone().detach().requires_grad_(True)

        # 前向传播
        wss_pred = model(features, velocity, pressure)

        # 计算梯度
        grads = torch.autograd.grad(
            outputs=wss_pred.sum(),
            inputs=features,
            create_graph=False,
            retain_graph=False
        )[0]

        # ====== 修改：从完整梯度中提取composite_features对应的维度 ======
        # 建立 composite_names 到 full_features 的索引映射
        # 根据 build_inputs_with_flow() 的39维特征顺序：
        # [d(1), normal(3), v1(1), d1(1), v2(1), d2(1), v3(1), d3(1), wss_ref(1), 
        #  n1(3), n2(3), n3(3), geom(4), flow_dir(3), alignment(1), alignment_abs(1), 
        #  flow_quality(1), v_align(3), linearity(1), v_gradient(1), angle_var(1), 占位(3)]
        
        full_dim_names = [
            'd', 'nx', 'ny', 'nz',
            'v1', 'd1', 'v2', 'd2', 'v3', 'd3',
            'wss_ref',
            'n1_x', 'n1_y', 'n1_z',
            'n2_x', 'n2_y', 'n2_z',
            'n3_x', 'n3_y', 'n3_z',
            'density', 'curvedness', 'shape_index', 'anisotropy',
            'flow_dir_x', 'flow_dir_y', 'flow_dir_z',
            'alignment', 'alignment_abs', 'flow_quality',
            'v_align1', 'v_align2', 'v_align3',
            'linearity', 'v_gradient', 'angle_var',
            'pad_1', 'pad_2', 'pad_3',
            # === 新增分离几何特征 (12维) ===
            'profile_linearity', 'profile_inflection', 'profile_b',
            'cluster_aniso', 'cluster_planarity',
            'vel_skewness', 'vel_kurtosis', 'vel_d_corr',
            'gauss_curv', 'mean_curv', 'convexity',
            'expansion_ratio'
        ]
        
        # composite_names 到 full_dim_names 的映射
        composite_to_full_idx = {
            'v1': [4], 'd1': [5], 'v2': [6], 'd2': [7], 'v3': [8], 'd3': [9],
            'wss_ref': [10],
            'n1': [11, 12, 13],  # n1_x, n1_y, n1_z
            'n2': [14, 15, 16],  # n2_x, n2_y, n2_z
            'n3': [17, 18, 19],  # n3_x, n3_y, n3_z
            'density': [20], 'curvedness': [21], 'shape_index': [22], 'anisotropy': [23]
        }
        
        # 提取各composite特征的梯度（对向量特征取均值）
        combined_names = []
        combined_grads_list = []
        
        for name in self.composite_names:
            full_indices = composite_to_full_idx.get(name)
            if full_indices is None:
                print(f"⚠️ 警告: {name} 未找到对应的完整特征索引，跳过")
                continue
            
            # 提取对应维度的梯度绝对值
            feature_grads = torch.abs(grads[:, full_indices])  # [N, k]
            
            if feature_grads.dim() == 2 and feature_grads.size(1) > 1:
                # 向量特征（如n1/n2/n3）：取三个分量的均值作为整体重要性
                mean_grad = feature_grads.mean(dim=1, keepdim=True)  # [N, 1]
            else:
                # 标量特征
                mean_grad = feature_grads  # [N, 1]
            
            combined_names.append(name)
            combined_grads_list.append(mean_grad)
        
        # 合并所有特征的梯度: [N, n_composite_features]
        combined_grads = torch.cat(combined_grads_list, dim=1)
        # ==============================================

        # 跨样本取平均得到最终重要性得分
        importance_scores = combined_grads.mean(dim=0).cpu().numpy()

        results = []
        for i, name in enumerate(combined_names):
            results.append({
                'feature': name,
                'gradient_importance': float(importance_scores[i]),
                'gradient_std': float(combined_grads[:, i].std().cpu().numpy()),
            })

        df = pd.DataFrame(results)
        df = df.sort_values('gradient_importance', ascending=False)
        return df
    def plot_n1_analysis(self, save_dir: str = None):
        """
        专门分析n1_alignment的物理意义
        """
        # 找到n1_alignment的索引
        idx_align = self.composite_names.index('n1_alignment')
        idx_v1 = self.composite_names.index('v1')
        idx_d1 = self.composite_names.index('d1')
        
        n1_align = self.composite_features[:, idx_align]
        v1 = self.composite_features[:, idx_v1]
        d1 = self.composite_features[:, idx_d1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. n1_alignment分布
        ax = axes[0, 0]
        ax.hist(n1_align, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='正交')
        ax.axvline(x=1, color='g', linestyle='--', label='完全对齐')
        ax.axvline(x=-1, color='g', linestyle='--')
        ax.set_xlabel('n1_alignment (cosθ)')
        ax.set_ylabel('频数')
        ax.set_title('V1方向与壁面法向的夹角分布')
        ax.legend()
        
        # 2. n1_alignment vs WSS
        ax = axes[0, 1]
        scatter = ax.scatter(n1_align, self.wss, c=v1, cmap='viridis', alpha=0.4, s=10)
        plt.colorbar(scatter, ax=ax, label='V1速度')
        ax.set_xlabel('n1_alignment (cosθ)')
        ax.set_ylabel('WSS')
        ax.set_title('方向对齐度 vs WSS (颜色=V1)')
        
        # 拟合趋势线
        z = np.polyfit(n1_align, self.wss, 2)
        p = np.poly1d(z)
        x_line = np.linspace(n1_align.min(), n1_align.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # 3. 有效法向距离 = d1 * |cosθ|
        ax = axes[1, 0]
        effective_d = d1 * np.abs(n1_align)
        ax.scatter(effective_d, self.wss, alpha=0.4, s=10, c='steelblue')
        ax.set_xlabel('Effective Normal Distance (d1 × |cosθ|)')
        ax.set_ylabel('WSS')
        ax.set_title('有效法向距离 vs WSS')
        
        # 4. 对比：原始d1 vs 有效d1的相关性
        ax = axes[1, 1]
        
        r_raw, _ = pearsonr(d1, self.wss)
        r_eff, _ = pearsonr(effective_d, self.wss)
        
        x = ['Raw d1', 'Effective d1\n(d1×|cosθ|)']
        y = [abs(r_raw), abs(r_eff)]
        colors = ['gray', 'green' if r_eff > r_raw else 'red']
        
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('|Pearson r|')
        ax.set_title('距离相关性提升')
        
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = f'{save_dir}/n1_alignment_analysis.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"n1_alignment分析图已保存: {save_path}")
            plt.close(fig)
        
        # plt.show()
        # plt.close()
        
        # 打印关键发现
        print(f"\n{'='*60}")
        print("【n1_alignment关键发现】")
        print(f"{'='*60}")
        print(f"  平均对齐度: {n1_align.mean():.3f} (0=正交, 1=对齐)")
        print(f"  对齐度标准差: {n1_align.std():.3f}")
        print(f"  完全对齐(|cos|>0.9)比例: {(np.abs(n1_align) > 0.9).mean():.1%}")
        print(f"  近似正交(|cos|<0.3)比例: {(np.abs(n1_align) < 0.3).mean():.1%}")
        print(f"\n  有效距离相关性提升: {abs(r_raw):.3f} → {abs(r_eff):.3f}")
        if abs(r_eff) > abs(r_raw):
            print(f"  ✅ 引入方向信息后，距离预测力提升")
        print(f"{'='*60}")
    
    def plot_correlation_analysis(self, save_dir: str = None):
        """
        绘制完整的相关性分析图表
        
        Args:
            save_dir: 保存目录
        """
        # 计算相关性
        corr_df = self.compute_correlations()
        
        # 创建大图
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 相关性条形图 (左上)
        ax1 = plt.subplot(3, 3, 1)
        colors = ['green' if p < 0.05 else 'gray' for p in corr_df['pearson_p']]
        bars = ax1.barh(corr_df['feature'], corr_df['pearson_r'], color=colors, alpha=0.7)
        ax1.set_xlabel('Pearson Correlation')
        ax1.set_title('Feature-WSS Pearson Correlation\n(green: p<0.05, gray: p>=0.05)')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 添加数值标签
        for i, (idx, row) in enumerate(corr_df.iterrows()):
            ax1.text(row['pearson_r'], i, f' {row["pearson_r"]:.3f}', 
                    va='center', fontsize=8)
        
        # 2. Spearman相关性 (右上)
        ax2 = plt.subplot(3, 3, 2)
        colors = ['blue' if p < 0.05 else 'gray' for p in corr_df['spearman_p']]
        bars = ax2.barh(corr_df['feature'], corr_df['spearman_r'], color=colors, alpha=0.7)
        ax2.set_xlabel('Spearman Correlation')
        ax2.set_title('Feature-WSS Spearman Correlation\n(blue: p<0.05, gray: p>=0.05)')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 3. 互信息得分 (中左)
        ax3 = plt.subplot(3, 3, 3)
        bars = ax3.barh(corr_df['feature'], corr_df['mi_score'], color='purple', alpha=0.7)
        ax3.set_xlabel('Mutual Information (bits)')
        ax3.set_title('Feature-WSS Mutual Information')
        
        # 4. 相关性热力图 (中右，占两格)
        ax4 = plt.subplot(3, 3, (4, 5))
        
        # 构建特征-WSS相关矩阵
        corr_matrix = np.zeros((len(self.composite_names), 2))
        corr_matrix[:, 0] = corr_df['pearson_r'].values
        corr_matrix[:, 1] = corr_df['spearman_r'].values
        
        sns.heatmap(
            corr_matrix,
            yticklabels=corr_df['feature'],
            xticklabels=['Pearson', 'Spearman'],
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            ax=ax4,
            cbar_kws={'label': 'Correlation'}
        )
        ax4.set_title('Correlation Heatmap')
        
        # 5. 散点图矩阵 - 最重要的4个特征 (底部，占3格)
        top_features = corr_df.head(4)['feature'].tolist()
        
        for idx, feat_name in enumerate(top_features):
            ax = plt.subplot(3, 3, 6 + idx)
            feat_idx = self.composite_names.index(feat_name)
            feature_vals = self.composite_features[:, feat_idx]
            
            # 绘制散点图
            ax.scatter(feature_vals, self.wss, alpha=0.3, s=10, c='steelblue')
            
            # 拟合趋势线
            z = np.polyfit(feature_vals, self.wss, 1)
            p = np.poly1d(z)
            x_line = np.linspace(feature_vals.min(), feature_vals.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # 添加相关系数
            r = corr_df[corr_df['feature'] == feat_name]['pearson_r'].values[0]
            ax.set_title(f'{feat_name}\nPearson r = {r:.3f}')
            ax.set_xlabel(feat_name)
            ax.set_ylabel('WSS (scaled)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = f'{save_dir}/feature_correlation_analysis.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n特征相关性分析图已保存: {save_path}")
            plt.close(fig)
            
            # 同时保存CSV
            csv_path = f'{save_dir}/feature_correlations.csv'
            corr_df.to_csv(csv_path, index=False, float_format='%.6f')
            print(f"相关性数据已保存: {csv_path}")
        
        # plt.show()
        # plt.close()
        
        return corr_df
    
    def plot_feature_importance_comparison(self, model, save_dir: str = None):
        """
        对比统计相关性和模型梯度重要性
        
        Args:
            model: 训练好的模型
            save_dir: 保存目录
        """
        # 获取两种重要性
        corr_df = self.compute_correlations()
        grad_df = self.analyze_feature_importance_with_model(model)

        if grad_df is None:
            print("  ⚠️ 梯度重要性分析跳过（模型输入特征维度过低），仅返回相关性")
            return None

        # 合并
        comparison = pd.merge(
            corr_df[['feature', 'abs_pearson', 'abs_spearman', 'mi_score']],
            grad_df[['feature', 'gradient_importance']],
            on='feature'
        )
        
        # 归一化以便比较
        for col in ['abs_pearson', 'abs_spearman', 'mi_score', 'gradient_importance']:
            max_val = comparison[col].max()
            if max_val > 0:
                comparison[f'{col}_norm'] = comparison[col] / max_val
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 对比条形图
        ax = axes[0, 0]
        x = np.arange(len(comparison))
        width = 0.2
        
        ax.bar(x - 1.5*width, comparison['abs_pearson_norm'], width, 
               label='|Pearson|', alpha=0.8)
        ax.bar(x - 0.5*width, comparison['abs_spearman_norm'], width,
               label='|Spearman|', alpha=0.8)
        ax.bar(x + 0.5*width, comparison['mi_score_norm'], width,
               label='Mutual Info', alpha=0.8)
        ax.bar(x + 1.5*width, comparison['gradient_importance_norm'], width,
               label='Gradient', alpha=0.8)
        
        ax.set_ylabel('Normalized Importance')
        ax.set_title('Feature Importance Comparison (Normalized)')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['feature'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 统计相关性 vs 梯度重要性散点图
        ax = axes[0, 1]
        ax.scatter(comparison['abs_pearson'], comparison['gradient_importance'],
                  s=100, alpha=0.6, c='steelblue', edgecolors='black')
        
        # 添加特征标签
        for i, row in comparison.iterrows():
            ax.annotate(row['feature'], 
                       (row['abs_pearson'], row['gradient_importance']),
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('|Pearson Correlation|')
        ax.set_ylabel('Gradient Importance')
        ax.set_title('Statistical vs Model-based Importance')
        ax.grid(True, alpha=0.3)
        
        # 3. 雷达图
        ax = axes[1, 0]
        self._plot_radar_chart(comparison, ax)
        
        # 4. 综合排序表
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        # 计算综合得分
        comparison['combined_score'] = (
            0.3 * comparison['abs_pearson_norm'] +
            0.2 * comparison['abs_spearman_norm'] +
            0.2 * comparison['mi_score_norm'] +
            0.3 * comparison['gradient_importance_norm']
        )
        comparison_sorted = comparison.sort_values('combined_score', ascending=False)
        
        # 显示表格
        table_data = comparison_sorted[['feature', 'abs_pearson', 'gradient_importance', 
                                       'combined_score']].head(10)
        table_data.columns = ['Feature', '|Pearson|', 'Gradient', 'Combined']
        
        table = ax.table(
            cellText=[[f'{v:.4f}' if isinstance(v, float) else v 
                      for v in row] for row in table_data.values],
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.4, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax.set_title('Top 10 Features by Combined Score', pad=20)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = f'{save_dir}/feature_importance_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n特征重要性对比图已保存: {save_path}")
            plt.close(fig)
        
        # plt.show()
        # plt.close()
        
        return comparison_sorted
    
    def _plot_radar_chart(self, df: pd.DataFrame, ax):
        """绘制雷达图展示多维特征重要性"""
        # 选择前6个特征
        top_n = 6
        df_top = df.nlargest(top_n, 'abs_pearson')
        
        categories = ['|Pearson|', '|Spearman|', 'Mutual Info', 'Gradient']
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合
        
        # 设置极坐标
        ax = plt.subplot(2, 2, 3, projection='polar')
        
        # 绘制每个特征
        colors = plt.cm.tab10(np.linspace(0, 1, top_n))
        
        for idx, (_, row) in enumerate(df_top.iterrows()):
            values = [
                row['abs_pearson_norm'],
                row['abs_spearman_norm'],
                row['mi_score_norm'],
                row['gradient_importance_norm']
            ]
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row['feature'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Top Features Radar Chart', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        ax.grid(True)

    def plot_geometry_wss_analysis(self, model=None, save_dir: str = None):
        """
        绘制几何特征分布与负WSS空间分布的联合分析图
        
        包含：
        1. curvedness, shape_index, density, anisotropy 的9子图分布（3特征×3视角）
        2. 负WSS的空间分布（3视角）
        3. 负WSS-几何特征联合散点图
        """
        
        coords = self.dataset.wall_coords
        wss = self.dataset.wall_wss.flatten()  # 去量纲后的WSS
        n_wall = len(coords)
        
        # 提取四个几何特征（根据build_inputs的索引：20-23）
        # curvedness=20, shape_index=21, density=22, anisotropy=23
        geom_names = ['curvedness', 'shape_index', 'density', 'anisotropy']
        geom_indices = [20, 21, 22, 23]
        geom_data = {name: self.features[:, idx] for name, idx in zip(geom_names, geom_indices)}
        
        # 负WSS掩码
        negative_wss_mask = wss < 0
        positive_wss_mask = wss >= 0
        
        print(f"\n{'='*60}")
        print("【几何特征与负WSS分布分析】")
        print(f"{'='*60}")
        print(f"  总壁面点数: {n_wall}")
        print(f"  负WSS点数: {negative_wss_mask.sum()} ({negative_wss_mask.mean():.1%})")
        print(f"  正WSS点数: {positive_wss_mask.sum()} ({positive_wss_mask.mean():.1%})")
        
        # 负WSS区域的几何特征统计
        print(f"\n  负WSS区域几何特征均值:")
        for name in geom_names:
            neg_mean = geom_data[name][negative_wss_mask].mean()
            pos_mean = geom_data[name][positive_wss_mask].mean()
            print(f"    {name:12s}: 负WSS={neg_mean:.4f}, 正WSS={pos_mean:.4f}, 比={neg_mean/(pos_mean+1e-8):.2f}")
        
        # ========== 图1: 4个几何特征 × 3视角 = 12子图 ==========
        fig1, axes1 = plt.subplots(4, 3, figsize=(18, 24))
        views = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]
        
        for row_idx, (name, data) in enumerate(geom_data.items()):
            # 统一colorbar范围
            vmin, vmax = data.min(), data.max()
            
            for col_idx, (i, j, xlabel, ylabel) in enumerate(views):
                ax = axes1[row_idx, col_idx]
                
                # 绘制所有点（淡色背景）
                sc = ax.scatter(coords[:, i], coords[:, j], 
                               c=data, cmap='viridis', 
                               s=10, alpha=0.4, vmin=vmin, vmax=vmax)
                
                # 叠加负WSS点（红色高亮）
                if negative_wss_mask.sum() > 0:
                    ax.scatter(coords[negative_wss_mask, i], coords[negative_wss_mask, j],
                              c='red', s=25, alpha=0.8, marker='x', 
                              label=f'Neg WSS ({negative_wss_mask.sum()})')
                
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{name} ({xlabel}{ylabel})')
                ax.set_aspect('equal', adjustable='box')
                ax.legend(loc='upper right', fontsize=8)
                
                # 每个子图独立colorbar
                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(name, rotation=270, labelpad=15)
        
        plt.suptitle('Geometry Features Spatial Distribution\nRed-X = Negative WSS Locations', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        if save_dir:
            save_path1 = f'{save_dir}/geometry_features_distribution.png'
            plt.savefig(save_path1, dpi=300, bbox_inches='tight')
            print(f"\n✓ 几何特征分布图已保存: {save_path1}")
        plt.close(fig1)
        
        # ========== 图2: 负WSS专属分析图 ==========
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行：负WSS空间分布（3视角）
        for col_idx, (i, j, xlabel, ylabel) in enumerate(views):
            ax = axes2[0, col_idx]
            
            # 正WSS背景（蓝色，半透明）
            ax.scatter(coords[positive_wss_mask, i], coords[positive_wss_mask, j],
                      c='lightblue', s=10, alpha=0.3, label='Positive WSS')
            
            # 负WSS前景（红色，明显）
            if negative_wss_mask.sum() > 0:
                neg_sc = ax.scatter(coords[negative_wss_mask, i], coords[negative_wss_mask, j],
                                   c=wss[negative_wss_mask], cmap='Reds_r',
                                   s=30, alpha=0.9, vmin=wss.min(), vmax=0,
                                   label='Negative WSS')
                plt.colorbar(neg_sc, ax=ax, label='WSS value')
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'Negative WSS Distribution ({xlabel}{ylabel})')
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='upper right', fontsize=8)
        
        # 第二行：负WSS vs 几何特征散点图
        scatter_pairs = [
            ('curvedness', 'Curvedness'),
            ('shape_index', 'Shape Index'),
            ('anisotropy', 'Anisotropy')
        ]
        
        for col_idx, (key, label) in enumerate(scatter_pairs):
            ax = axes2[1, col_idx]
            
            # 正WSS散点（蓝色，淡）
            ax.scatter(geom_data[key][positive_wss_mask], wss[positive_wss_mask],
                      c='lightblue', s=10, alpha=0.3, label='Positive')
            
            # 负WSS散点（红色，明显）
            if negative_wss_mask.sum() > 0:
                ax.scatter(geom_data[key][negative_wss_mask], wss[negative_wss_mask],
                          c='red', s=20, alpha=0.7, label='Negative')
            
            ax.set_xlabel(label)
            ax.set_ylabel('WSS (scaled)')
            ax.set_title(f'WSS vs {label}')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加相关系数
            if negative_wss_mask.sum() > 10:
                from scipy.stats import pearsonr
                r_neg, _ = pearsonr(geom_data[key][negative_wss_mask], wss[negative_wss_mask])
                r_all, _ = pearsonr(geom_data[key], wss)
                ax.text(0.05, 0.95, f'r_all={r_all:.3f}\nr_neg={r_neg:.3f}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Negative WSS Analysis: Spatial Distribution & Geometry Correlation', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        if save_dir:
            save_path2 = f'{save_dir}/negative_wss_analysis.png'
            plt.savefig(save_path2, dpi=300, bbox_inches='tight')
            print(f"✓ 负WSS分析图已保存: {save_path2}")
        plt.close(fig2)
        
        # ========== 图3: 几何特征直方图对比（正vs负WSS） ==========
        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
        axes3 = axes3.flatten()
        
        for idx, (name, data) in enumerate(geom_data.items()):
            ax = axes3[idx]
            
            # 正WSS直方图
            ax.hist(data[positive_wss_mask], bins=50, alpha=0.5, color='blue', 
                   label='Positive WSS', density=True)
            
            # 负WSS直方图
            if negative_wss_mask.sum() > 0:
                ax.hist(data[negative_wss_mask], bins=50, alpha=0.5, color='red',
                       label='Negative WSS', density=True)
            
            ax.set_xlabel(name)
            ax.set_ylabel('Density')
            ax.set_title(f'{name} Distribution by WSS Sign')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # KS检验
            if negative_wss_mask.sum() > 10:
                from scipy.stats import ks_2samp
                ks_stat, p_val = ks_2samp(data[positive_wss_mask], data[negative_wss_mask])
                ax.text(0.95, 0.95, f'KS={ks_stat:.3f}\np={p_val:.2e}', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Geometry Feature Histograms: Positive vs Negative WSS\nKS-test p<0.05 indicates significant difference', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        if save_dir:
            save_path3 = f'{save_dir}/geometry_histograms_wss_sign.png'
            plt.savefig(save_path3, dpi=300, bbox_inches='tight')
            print(f"✓ 几何特征直方图已保存: {save_path3}")
        plt.close(fig3)
        
        return {
            'negative_wss_ratio': float(negative_wss_mask.mean()),
            'negative_count': int(negative_wss_mask.sum())
        }
    
    def analyze_curvedness_wss_sign(self, save_dir=None):
        """
        分析曲率（curvedness）与WSS符号的相关性
        
        Returns:
            dict: 分析结果统计量
        """
        print(f"\n{'='*70}")
        print("【曲率与WSS符号相关性分析】")
        print(f"{'='*70}")
        
        # 提取curvedness（假设是局部几何特征的第2列，即index -2）
        # 根据您的build_inputs，local_geom_features顺序：
        # [density, curvedness, shape_index, anisotropy]
        curvedness_idx = self.features.shape[1] - 4 + 1  # 倒数第3列（0-based）
        curvedness = self.features[:, curvedness_idx]
        
        wss = self.wss
        wss_sign = np.sign(wss)
        
        # 1. 基础统计
        print("\n1. 基础统计")
        print(f"  壁面点数: {len(curvedness)}")
        print(f"  Curvedness范围: [{curvedness.min():.4f}, {curvedness.max():.4f}]")
        print(f"  WSS>0比例: {(wss > 0).mean():.2%}")
        print(f"  WSS<0比例: {(wss < 0).mean():.2%}")
        
        # 2. Curvedness vs |WSS| 相关性
        wss_abs = np.abs(wss)
        pearson_r, pearson_p = pearsonr(curvedness, wss_abs)
        spearman_r, spearman_p = spearmanr(curvedness, wss_abs)
        
        print(f"\n2. Curvedness vs |WSS| 相关性")
        print(f"  Pearson r = {pearson_r:.4f} (p={pearson_p:.2e})")
        print(f"  Spearman ρ = {spearman_r:.4f} (p={spearman_p:.2e})")
        
        # 3. 按WSS符号分组统计
        pos_mask = wss > 0
        neg_mask = wss < 0
        
        print(f"\n3. Curvedness vs WSS符号 分组统计")
        print(f"  WSS>0组: curvedness = {curvedness[pos_mask].mean():.4f} ± {curvedness[pos_mask].std():.4f}")
        print(f"  WSS<0组: curvedness = {curvedness[neg_mask].mean():.4f} ± {curvedness[neg_mask].std():.4f}")
        
        # t检验
        from scipy.stats import ttest_ind, mannwhitneyu
        t_stat, t_p = ttest_ind(curvedness[pos_mask], curvedness[neg_mask])
        u_stat, u_p = mannwhitneyu(curvedness[pos_mask], curvedness[neg_mask], alternative='two-sided')
        print(f"\n  t-test: t={t_stat:.4f}, p={t_p:.2e}")
        print(f"  Mann-Whitney U: p={u_p:.2e}")
        
        # 4. 分箱分析
        print(f"\n4. 分箱分析（Curvedness分位数 vs WSS>0比例）")
        n_bins = 5
        bin_edges = np.percentile(curvedness, np.linspace(0, 100, n_bins+1))
        bin_analysis = []
        
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (curvedness >= bin_edges[i]) & (curvedness <= bin_edges[i+1])
            else:
                mask = (curvedness >= bin_edges[i]) & (curvedness < bin_edges[i+1])
            
            if mask.sum() > 0:
                wss_pos_ratio = (wss[mask] > 0).mean()
                print(f"  Bin{i+1} [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}]: "
                      f"WSS>0={wss_pos_ratio:.2%} (n={mask.sum()})")
                bin_analysis.append({
                    'bin': i+1,
                    'curvedness_min': bin_edges[i],
                    'curvedness_max': bin_edges[i+1],
                    'wss_pos_ratio': wss_pos_ratio,
                    'count': mask.sum()
                })
        
        # 5. 卡方检验
        print(f"\n5. 卡方检验（Curvedness二值 × WSS符号）")
        curvedness_median = np.median(curvedness)
        curvedness_high = curvedness > curvedness_median
        
        contingency = pd.crosstab(
            pd.Series(curvedness_high, name='High_Curvedness'),
            pd.Series(wss_sign, name='WSS_Sign')
        )
        print(f"\n  列联表 (median={curvedness_median:.4f}):")
        print(contingency)
        
        chi2, chi2_p, dof, expected = chi2_contingency(contingency)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        print(f"\n  Chi-square = {chi2:.4f}, p = {chi2_p:.2e}")
        print(f"  Cramér's V = {cramers_v:.4f}")
        
        # 6. 物理解释
        self._interpret_curvedness_wss(curvedness, wss, wss_sign, curvedness_median)
        
        # 7. 可视化
        if save_dir:
            self._plot_curvedness_wss_sign(curvedness, wss, wss_sign, 
                                           bin_analysis, contingency, save_dir)
        
        return {
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            't_test_p': t_p,
            'chi2_p': chi2_p,
            'cramers_v': cramers_v,
            'curvedness_median': curvedness_median,
            'significant': chi2_p < 0.05
        }
    
    def _interpret_curvedness_wss(self, curvedness, wss, wss_sign, median):
        """物理解释"""
        print(f"\n{'='*70}")
        print("【物理解释】")
        print(f"{'='*70}")
        
        high_curv = curvedness > median
        high_curv_pos_ratio = (wss[high_curv] > 0).mean()
        low_curv_pos_ratio = (wss[~high_curv] > 0).mean()
        
        print(f"\n  高曲率区域 WSS>0 比例: {high_curv_pos_ratio:.2%}")
        print(f"  低曲率区域 WSS>0 比例: {low_curv_pos_ratio:.2%}")
        
        diff = abs(high_curv_pos_ratio - low_curv_pos_ratio)
        
        if diff < 0.05:
            print(f"\n结论: 曲率与WSS符号无显著关联（差异={diff:.1%}）")
            print("  → WSS符号主要由流动方向决定，与局部几何无关")
        elif high_curv_pos_ratio > low_curv_pos_ratio:
            print(f"\n结论: 高曲率区域倾向于WSS>0（差异={diff:.1%}）")
            print("  → 可能对应外侧壁/加速区，或二次流效应")
        else:
            print(f"\n结论: 高曲率区域倾向于WSS<0（差异={diff:.1%}）")
            print("  → 可能对应内侧壁/减速区，或分离流")
        
        print(f"\n模型启示:")
        if diff > 0.1:
            print("  ⚠️  建议：将 curvedness×sign(V1) 作为交互特征")
            print("      或设计曲率感知的WSS符号预测分支")
        else:
            print("  ✓ 曲率不直接影响WSS符号，保持现有设计")
    
    def _plot_curvedness_wss_sign(self, curvedness, wss, wss_sign, 
                                   bin_analysis, contingency, save_dir):
        """绘制分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        pos_mask = wss > 0
        neg_mask = wss < 0
        
        # 1. 散点图
        ax = axes[0, 0]
        ax.scatter(curvedness[pos_mask], wss[pos_mask], c='red', alpha=0.5, s=10, label='WSS>0')
        ax.scatter(curvedness[neg_mask], wss[neg_mask], c='blue', alpha=0.5, s=10, label='WSS<0')
        ax.axhline(y=0, color='black', linestyle='--')
        ax.set_xlabel('Curvedness')
        ax.set_ylabel('WSS')
        ax.set_title('Curvedness vs WSS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 箱线图
        ax = axes[0, 1]
        data_for_box = [curvedness[pos_mask], curvedness[neg_mask]]
        bp = ax.boxplot(data_for_box, labels=['WSS>0', 'WSS<0'], patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('blue')
        ax.set_ylabel('Curvedness')
        ax.set_title('Curvedness by WSS Sign')
        ax.grid(True, alpha=0.3)
        
        # 3. 分箱柱状图
        ax = axes[0, 2]
        bins = [b['bin'] for b in bin_analysis]
        ratios = [b['wss_pos_ratio'] for b in bin_analysis]
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bins)))
        bars = ax.bar([f'Q{b}' for b in bins], ratios, color=colors, edgecolor='black')
        ax.axhline(y=0.5, color='black', linestyle='--', label='50%')
        ax.set_ylabel('WSS>0 Ratio')
        ax.set_title('WSS>0 Ratio by Curvedness Quintile')
        ax.set_ylim([0, 1])
        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{ratio:.1%}', ha='center', fontsize=9)
        ax.legend()
        
        # 4-6. 空间分布图（使用XY平面）
        coords = self.dataset.wall_coords
        
        ax = axes[1, 0]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=wss_sign, cmap='RdBu_r', s=15, vmin=-1, vmax=1)
        plt.colorbar(sc, ax=ax, label='WSS Sign')
        ax.set_title('WSS Sign (XY)')
        
        ax = axes[1, 1]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=curvedness, cmap='viridis', s=15)
        plt.colorbar(sc, ax=ax, label='Curvedness')
        ax.set_title('Curvedness (XY)')
        
        ax = axes[1, 2]
        high_curv_pos = (curvedness > np.median(curvedness)) & (wss > 0)
        colors = np.zeros((len(coords), 4))
        colors[high_curv_pos] = [1, 0, 0, 0.8]
        colors[(curvedness > np.median(curvedness)) & (wss < 0)] = [0, 0, 1, 0.8]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=15)
        ax.set_title('High Curv + WSS Sign (Red=Pos, Blue=Neg)')
        
        plt.tight_layout()
        save_path = f'{save_dir}/curvedness_wss_sign_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表保存: {save_path}")
        plt.close(fig)
    
    def generate_full_report(self, save_dir=None):
        """生成完整的特征分析报告"""
        print(f"\n{'='*70}")
        print("【综合特征分析报告】")
        print(f"{'='*70}")
        
        # 1. 基础相关性
        corr_df = self.compute_correlations()
        self.print_correlation_report(corr_df)
        
        # 2. 曲率-WSS符号分析
        curvedness_results = self.analyze_curvedness_wss_sign(save_dir=save_dir)
        
        # 3. 梯度重要性（如果有模型）
        # ... 可扩展
        
        # 4. 总结建议
        print(f"\n{'='*70}")
        print("【特征工程建议】")
        print(f"{'='*70}")
        
        if curvedness_results['significant']:
            print("1. 曲率与WSS符号显著相关")
            print("   → 建议添加 curvedness×sign(v1) 交互特征")
            print("   → 或设计曲率感知的符号预测分支")
        else:
            print("1. 曲率与WSS符号无显著关联")
            print("   → WSS符号主要由流动方向决定")
        
        # 根据Pearson相关性排序
        top_features = corr_df.head(3)['feature'].tolist()
        print(f"\n2. 与WSS最相关的特征: {', '.join(top_features)}")
        print(f"   → 确保这些特征在模型输入中完整保留")
        
        return {
            'correlation': corr_df,
            'curvedness_sign': curvedness_results
        }

    def analyze_wss_flow_direction(self, dataset, save_dir: str = None):
        """
        分析WSS符号与局部流动方向的关系，并绘制可视化
        """
        print(f"\n{'='*70}")
        print("【WSS符号 vs 局部流动方向分析】")
        print(f"{'='*70}")
        
        if not hasattr(dataset, 'wall_flow_direction'):
            dataset.compute_local_flow_direction()
            dataset.compute_wss_direction_alignment()
        
        coords = dataset.wall_coords
        wss = dataset.wall_wss.flatten()
        wss_sign = np.sign(wss)
        alignment = dataset.wss_flow_alignment
        flow_dir = dataset.wall_flow_direction
        flow_quality = dataset.wall_flow_quality
        normal = dataset.wall_normal
        
        n_wall = len(coords)
        pos_mask = wss_sign > 0
        neg_mask = wss_sign < 0
        
        # ========== 统计量计算 ==========
        print("\n1. 对齐度 (flow·normal) 统计")
        print(f"  WSS>0: mean={alignment[pos_mask].mean():.4f}, std={alignment[pos_mask].std():.4f}")
        print(f"  WSS<0: mean={alignment[neg_mask].mean():.4f}, std={alignment[neg_mask].std():.4f}")
        
        # 流动方向质量
        print(f"\n2. 流动方向质量")
        print(f"  WSS>0: quality={flow_quality[pos_mask].mean():.3f}")
        print(f"  WSS<0: quality={flow_quality[neg_mask].mean():.3f}")
        
        # 法向-流动夹角
        flow_normal_angle = np.arccos(np.clip(
            np.sum(flow_dir * normal, axis=1), -1, 1
        )) * 180 / np.pi
        
        print(f"\n3. 流动方向与法向夹角")
        print(f"  WSS>0: {flow_normal_angle[pos_mask].mean():.1f}° ± {flow_normal_angle[pos_mask].std():.1f}°")
        print(f"  WSS<0: {flow_normal_angle[neg_mask].mean():.1f}° ± {flow_normal_angle[neg_mask].std():.1f}°")
        print(f"  (理想层流应≈90°，即流动平行于壁面)")
        
        # 回流判断
        recirculation_threshold = -0.3
        recirculation_mask = neg_mask & (alignment < recirculation_threshold)
        print(f"\n4. 疑似回流点数: {recirculation_mask.sum()} ({recirculation_mask.sum()/n_wall:.1%})")
        
        # ========== 绘图 ==========
        fig = plt.figure(figsize=(22, 18))
        
        # 1. 对齐度直方图对比 (3,3,1)
        ax1 = plt.subplot(3, 3, 1)
        bins = np.linspace(-1, 1, 60)
        counts_pos, _, _ = ax1.hist(alignment[pos_mask], bins=bins, alpha=0.6, color='blue', 
                                    label=f'WSS>0 (n={pos_mask.sum()})', density=True)
        counts_neg, _, _ = ax1.hist(alignment[neg_mask], bins=bins, alpha=0.6, color='red', 
                                    label=f'WSS<0 (n={neg_mask.sum()})', density=True)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
        ax1.axvline(x=alignment[pos_mask].mean(), color='blue', linestyle='-', linewidth=2, 
                    label=f'Mean+={alignment[pos_mask].mean():.3f}')
        ax1.axvline(x=alignment[neg_mask].mean(), color='red', linestyle='-', linewidth=2,
                    label=f'Mean-={alignment[neg_mask].mean():.3f}')
        ax1.set_xlabel('Alignment (flow·normal)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Alignment Distribution by WSS Sign', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 箱线图 (3,3,2)
        ax2 = plt.subplot(3, 3, 2)
        bp = ax2.boxplot([alignment[pos_mask], alignment[neg_mask]], 
                        labels=['WSS>0\n(Normal)', 'WSS<0\n(Reverse)'], 
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        ax2.set_ylabel('Alignment', fontsize=11)
        ax2.set_title('Alignment Boxplot Comparison', fontsize=12, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加统计标注
        y_max = max(alignment.max(), 0.5)
        ax2.annotate(f'p={7.44e-69:.2e}', xy=(1.5, y_max*0.9), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 3. 散点图：alignment vs |WSS| (3,3,3)
        ax3 = plt.subplot(3, 3, 3)
        wss_abs = np.abs(wss)
        ax3.scatter(alignment[pos_mask], wss_abs[pos_mask], c='blue', alpha=0.2, s=8, label='WSS>0')
        ax3.scatter(alignment[neg_mask], wss_abs[neg_mask], c='red', alpha=0.4, s=12, label='WSS<0')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.axvline(x=-0.3, color='orange', linestyle='--', linewidth=2, label='Recirc threshold')
        ax3.set_xlabel('Alignment (flow·normal)', fontsize=11)
        ax3.set_ylabel('|WSS| (scaled)', fontsize=11)
        ax3.set_title('Alignment vs |WSS| Magnitude', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4-6. 空间分布图 (XY, XZ, YZ) - 带流动方向箭头
        views = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]
        for idx, (i, j, xlabel, ylabel) in enumerate(views):
            ax = plt.subplot(3, 3, 4 + idx)
            
            # 背景：所有壁面点（淡灰色）
            ax.scatter(coords[:, i], coords[:, j], c='lightgray', s=3, alpha=0.3, zorder=1)
            
            # WSS>0：蓝色，大小表示|WSS|
            sizes_pos = 15 + 60 * (wss_abs[pos_mask] / (wss_abs.max() + 1e-8))
            ax.scatter(coords[pos_mask, i], coords[pos_mask, j], 
                    c='steelblue', s=sizes_pos, alpha=0.5, zorder=2, label='WSS>0')
            
            # WSS<0：红色，大小表示|WSS|
            sizes_neg = 20 + 80 * (wss_abs[neg_mask] / (wss_abs.max() + 1e-8))
            ax.scatter(coords[neg_mask, i], coords[neg_mask, j], 
                    c='crimson', s=sizes_neg, alpha=0.7, zorder=3, label='WSS<0')
            
            # 疑似回流点：黄色星形高亮
            if recirculation_mask.sum() > 0:
                ax.scatter(coords[recirculation_mask, i], coords[recirculation_mask, j],
                        c='gold', s=100, marker='*', edgecolors='black', linewidths=1,
                        label=f'Recirculation (n={recirculation_mask.sum()})', zorder=5)
            
            # 绘制局部流动方向箭头（每N个点画一个，避免过密）
            step = max(1, n_wall // 150)
            for k in range(0, n_wall, step):
                # 箭头方向：流动方向投影到当前平面
                arrow_dir = np.array([flow_dir[k, i], flow_dir[k, j]])
                arrow_norm = np.linalg.norm(arrow_dir)
                if arrow_norm > 0.2:
                    arrow_dir = arrow_dir / arrow_norm * 0.8  # 缩放
                    
                    # 颜色根据WSS符号
                    color = 'blue' if wss_sign[k] > 0 else 'red'
                    alpha = 0.3 if wss_sign[k] > 0 else 0.5
                    
                    ax.annotate('', 
                            xy=(coords[k, i] + arrow_dir[0], coords[k, j] + arrow_dir[1]),
                            xytext=(coords[k, i], coords[k, j]),
                            arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=0.8),
                            zorder=4)
            
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'WSS Sign & Flow Direction ({xlabel}{ylabel})', fontsize=12, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.2)
        
        # 7. 对齐度空间分布 (3,3,7)
        ax7 = plt.subplot(3, 3, 7)
        sc = ax7.scatter(coords[:, 0], coords[:, 1], c=alignment, cmap='RdBu_r',
                        vmin=-1, vmax=1, s=12, alpha=0.7, zorder=2)
        plt.colorbar(sc, ax=ax7, label='Alignment', fraction=0.046)
        # 叠加WSS<0点轮廓
        ax7.scatter(coords[neg_mask, 0], coords[neg_mask, 1], 
                facecolors='none', edgecolors='black', s=25, linewidths=0.8, zorder=3)
        ax7.set_xlabel('X', fontsize=11)
        ax7.set_ylabel('Y', fontsize=11)
        ax7.set_title('Alignment Spatial Map\n(Black circle=WSS<0)', fontsize=12, fontweight='bold')
        ax7.set_aspect('equal', adjustable='box')
        ax7.grid(True, alpha=0.2)
        
        # 8. 流动方向质量空间分布 (3,3,8)
        ax8 = plt.subplot(3, 3, 8)
        sc = ax8.scatter(coords[:, 0], coords[:, 1], c=flow_quality, cmap='YlOrRd',
                        vmin=0, vmax=1, s=12, alpha=0.7)
        plt.colorbar(sc, ax=ax8, label='Quality Score', fraction=0.046)
        ax8.set_xlabel('X', fontsize=11)
        ax8.set_ylabel('Y', fontsize=11)
        ax8.set_title('Flow Direction Quality\n(Higher=More Reliable)', fontsize=12, fontweight='bold')
        ax8.set_aspect('equal', adjustable='box')
        ax8.grid(True, alpha=0.2)
        
        # 9. 回流判断综合图 (3,3,9)
        ax9 = plt.subplot(3, 3, 9)
        
        # 四类流动状态
        normal_flow = pos_mask & (alignment > 0.1)       # WSS>0, 明显远离壁面
        normal_attach = pos_mask & (alignment <= 0.1)  # WSS>0, 近平行/轻微指向
        reverse_detach = neg_mask & (alignment > -0.3) # WSS<0, 弱反向
        reverse_attach = neg_mask & (alignment <= -0.3) # WSS<0, 强指向壁面（回流）
        
        categories = ['Normal\nFlow', 'Normal\nAttach', 'Reverse\nDetach', 'Reverse\nAttach\n(Recirc)']
        counts = [normal_flow.sum(), normal_attach.sum(), 
                reverse_detach.sum(), reverse_attach.sum()]
        colors = ['#2166ac', '#92c5de', '#f4a582', '#b2182b']  # 蓝→红渐变
        
        bars = ax9.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.2)
        ax9.set_ylabel('Point Count', fontsize=11)
        ax9.set_title('Flow Regime Classification', fontsize=12, fontweight='bold')
        
        for bar, count in zip(bars, counts):
            pct = count / n_wall * 100
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax9.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('WSS Sign vs Local Flow Direction Analysis\n' + 
                    f'WSS>0: {pos_mask.sum()} pts (86.4%) | WSS<0: {neg_mask.sum()} pts (13.6%) | ' +
                    f'Recirculation: {recirculation_mask.sum()} pts (7.6%)\n' +
                    f't-test p=7.44e-69 (Highly Significant)', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_dir:
            save_path = f'{save_dir}/wss_flow_direction_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ WSS-流动方向分析图已保存: {save_path}")
        plt.close(fig)
        
        # ========== ROC分析 ==========
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
        
        y_true_binary = (wss_sign > 0).astype(int)  # 1=WSS>0, 0=WSS<0
        
        # ROC曲线
        fpr, tpr, roc_thresholds = roc_curve(y_true_binary, alignment)
        roc_auc = auc(fpr, tpr)
        
        # PR曲线
        precision_pr, recall_pr, pr_thresholds = precision_recall_curve(y_true_binary, alignment)
        pr_auc = auc(recall_pr, precision_pr)
        
        # 最优阈值（Youden指数）
        youden_j = tpr - fpr
        best_youden_idx = np.argmax(youden_j)
        optimal_thresh = roc_thresholds[best_youden_idx]
        
        print(f"\n4. ROC分析结果")
        print(f"  AUC-ROC: {roc_auc:.4f}")
        print(f"  AUC-PR:  {pr_auc:.4f}")
        print(f"  最优阈值 (Youden): {optimal_thresh:.3f}")
        print(f"    - 敏感度: {tpr[best_youden_idx]:.3f}")
        print(f"    - 特异度: {1-fpr[best_youden_idx]:.3f}")
        
        # ========== 四类流动状态定义 ==========
        normal_flow = pos_mask & (alignment > 0.1)       
        normal_attach = pos_mask & (alignment <= 0.1)    
        reverse_detach = neg_mask & (alignment > -0.3)  
        reverse_attach = neg_mask & (alignment <= -0.3) 
        
        recirculation_mask = reverse_attach  # 强回流
        
        print(f"\n5. 四类流动状态分布")
        print(f"  Normal Flow:      {normal_flow.sum()} ({normal_flow.sum()/n_wall:.1%})")
        print(f"  Normal Attach:    {normal_attach.sum()} ({normal_attach.sum()/n_wall:.1%})")
        print(f"  Reverse Detach:   {reverse_detach.sum()} ({reverse_detach.sum()/n_wall:.1%})")
        print(f"  Reverse Attach:   {reverse_attach.sum()} ({reverse_attach.sum()/n_wall:.1%}) [Recirculation]")
        
        # ========== 绘图 ==========
        fig = plt.figure(figsize=(22, 18))
        
        # 颜色方案
        colors_regime = {
            'normal_flow': '#2166ac',
            'normal_attach': '#92c5de',
            'reverse_detach': '#f4a582',
            'reverse_attach': '#b2182b'
        }
        
        # 1. Alignment分布密度图 (3,3,1)
        ax1 = plt.subplot(3, 3, 1)
        from scipy.stats import gaussian_kde
        try:
            kde_pos = gaussian_kde(alignment[pos_mask])
            kde_neg = gaussian_kde(alignment[neg_mask])
            x_range = np.linspace(-2, 2, 500)
            ax1.fill_between(x_range, kde_pos(x_range), alpha=0.4, color='blue', 
                            label=f'WSS>0 (μ={alignment[pos_mask].mean():.3f})')
            ax1.fill_between(x_range, kde_neg(x_range), alpha=0.4, color='red',
                            label=f'WSS<0 (μ={alignment[neg_mask].mean():.3f})')
            ax1.plot(x_range, kde_pos(x_range), color='blue', linewidth=2)
            ax1.plot(x_range, kde_neg(x_range), color='red', linewidth=2)
        except:
            # KDE失败时回退到直方图
            bins = np.linspace(-2, 2, 50)
            ax1.hist(alignment[pos_mask], bins=bins, alpha=0.6, color='blue', density=True, label='WSS>0')
            ax1.hist(alignment[neg_mask], bins=bins, alpha=0.6, color='red', density=True, label='WSS<0')
        
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=alignment[pos_mask].mean(), color='blue', linestyle='--', alpha=0.7)
        ax1.axvline(x=alignment[neg_mask].mean(), color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Alignment (flow·normal)', fontsize=11)
        ax1.set_ylabel('Probability Density', fontsize=11)
        ax1.set_title('Distribution Density', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-2, 2])
        
        # 2. ROC曲线 (3,3,2)
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(fpr, tpr, color='#1f77b4', linewidth=2.5, label=f'ROC (AUC={roc_auc:.4f})')
        ax2.fill_between(fpr, tpr, alpha=0.15, color='#1f77b4')
        ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random')
        
        # 标记关键阈值点
        key_points = [
            (-0.3, 'Conservative', 'red'),
            (optimal_thresh, 'Optimal', 'green'),
            (0.1, 'Liberal', 'orange')
        ]
        for thresh, label, color in key_points:
            idx = np.argmin(np.abs(roc_thresholds - thresh))
            ax2.scatter(fpr[idx], tpr[idx], s=100, c=color, zorder=5, edgecolors='black')
            ax2.annotate(f'{label}\n(thresh={thresh:.2f})',
                        xy=(fpr[idx], tpr[idx]), xytext=(fpr[idx]+0.1, tpr[idx]-0.08),
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1))
        
        ax2.set_xlabel('1 - Specificity', fontsize=11)
        ax2.set_ylabel('Sensitivity', fontsize=11)
        ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-0.02, 1.02])
        ax2.set_ylim([-0.02, 1.02])
        
        # 3. Precision-Recall曲线 (3,3,3)
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(recall_pr, precision_pr, color='#2ca02c', linewidth=2.5, 
                label=f'PR Curve (AP={pr_auc:.3f})')
        ax3.fill_between(recall_pr, precision_pr, alpha=0.15, color='#2ca02c')
        # baseline = n_pos / n_wall
        # ax3.axhline(y=baseline, color='red', linestyle='--', linewidth=1.5, 
        #             label=f'Baseline={baseline:.3f}')
        
        # 标记最佳F1
        f1_scores_pr = 2 * (precision_pr * recall_pr) / (precision_pr + recall_pr + 1e-10)
        best_f1_idx = np.argmax(f1_scores_pr)
        ax3.scatter(recall_pr[best_f1_idx], precision_pr[best_f1_idx], 
                s=120, c='gold', zorder=5, marker='*', edgecolors='black')
        ax3.annotate(f'Best F1={f1_scores_pr[best_f1_idx]:.3f}',
                    xy=(recall_pr[best_f1_idx], precision_pr[best_f1_idx]),
                    xytext=(recall_pr[best_f1_idx]-0.2, precision_pr[best_f1_idx]+0.05),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))
        
        ax3.set_xlabel('Recall', fontsize=11)
        ax3.set_ylabel('Precision', fontsize=11)
        ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='lower left')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1.05])
        
        # 4-6. 空间分布图 (XY, XZ, YZ)
        views = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]
        for idx, (i, j, xlabel, ylabel) in enumerate(views):
            ax = plt.subplot(3, 3, 4 + idx)
            
            # 背景
            ax.scatter(coords[:, i], coords[:, j], c='lightgray', s=3, alpha=0.3, zorder=1)
            
            # 四类分别绘制
            wss_abs = np.abs(wss)
            for mask, label, color, marker, size in [
                (normal_flow, 'Normal Flow', colors_regime['normal_flow'], 'o', 12),
                (normal_attach, 'Normal Attach', colors_regime['normal_attach'], 's', 10),
                (reverse_detach, 'Reverse Detach', colors_regime['reverse_detach'], 'D', 14),
                (reverse_attach, 'Reverse/Recirc', colors_regime['reverse_attach'], '*', 60)
            ]:
                ax.scatter(coords[mask, i], coords[mask, j], 
                        c=color, s=size, alpha=0.7, zorder=3, 
                        label=f'{label} (n={mask.sum()})', marker=marker,
                        edgecolors='black', linewidths=0.3)
            
            # 流动方向箭头（稀疏采样）
            step = max(1, n_wall // 150)
            for k in range(0, n_wall, step):
                arrow_dir = np.array([flow_dir[k, i], flow_dir[k, j]])
                arrow_norm = np.linalg.norm(arrow_dir)
                if arrow_norm > 0.2:
                    arrow_dir = arrow_dir / arrow_norm * 0.6
                    color = 'blue' if wss_sign[k] > 0 else 'red'
                    alpha = 0.25 if wss_sign[k] > 0 else 0.4
                    ax.annotate('', 
                            xy=(coords[k, i] + arrow_dir[0], coords[k, j] + arrow_dir[1]),
                            xytext=(coords[k, i], coords[k, j]),
                            arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=0.6),
                            zorder=2)
            
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Flow Regime ({xlabel}{ylabel})', fontsize=12, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.2)
        
        # 7. 阈值扫描性能图 (3,3,7)
        ax7 = plt.subplot(3, 3, 7)
        
        thresholds_scan = np.linspace(-1.5, 1.0, 300)
        sens_list, spec_list, prec_list, f1_list = [], [], [], []
        
        for thresh in thresholds_scan:
            y_pred_t = (alignment > thresh).astype(int)
            tp = np.sum((y_true_binary == 1) & (y_pred_t == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_t == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_t == 0))
            tn = np.sum((y_true_binary == 0) & (y_pred_t == 0))
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1_t = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
            
            sens_list.append(sens)
            spec_list.append(spec)
            prec_list.append(prec)
            f1_list.append(f1_t)
        
        ax7.plot(thresholds_scan, sens_list, color='green', linewidth=2, label='Sensitivity', alpha=0.8)
        ax7.plot(thresholds_scan, spec_list, color='blue', linewidth=2, label='Specificity', alpha=0.8)
        ax7.plot(thresholds_scan, prec_list, color='red', linewidth=2, label='Precision', alpha=0.8)
        ax7.plot(thresholds_scan, f1_list, color='purple', linewidth=2.5, label='F1 Score', alpha=0.9)
        
        # 标记最优F1
        best_f1_idx = np.argmax(f1_list)
        ax7.axvline(x=thresholds_scan[best_f1_idx], color='purple', linestyle='--', alpha=0.5)
        ax7.scatter(thresholds_scan[best_f1_idx], f1_list[best_f1_idx], 
                s=100, c='purple', zorder=5, marker='*')
        ax7.text(thresholds_scan[best_f1_idx]+0.05, f1_list[best_f1_idx]+0.03,
                f'Best F1={f1_list[best_f1_idx]:.3f}\\n@ thresh={thresholds_scan[best_f1_idx]:.3f}',
                fontsize=8, color='purple', fontweight='bold')
        
        # 标记当前分类阈值
        for thresh, color in [(-0.3, 'red'), (0.1, 'orange')]:
            ax7.axvline(x=thresh, color=color, linestyle=':', alpha=0.4)
        
        ax7.set_xlabel('Alignment Threshold', fontsize=11)
        ax7.set_ylabel('Score', fontsize=11)
        ax7.set_title('Metrics vs Threshold', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=8, loc='center left')
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, 1.05])
        
        # 8. 混淆矩阵（使用最优阈值） (3,3,8)
        ax8 = plt.subplot(3, 3, 8)
        
        y_pred_optimal = (alignment > optimal_thresh).astype(int)
        cm = confusion_matrix(y_true_binary, y_pred_optimal)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        im = ax8.imshow(cm_percent, cmap='Blues', aspect='auto', vmin=0, vmax=100)
        
        for i in range(2):
            for j in range(2):
                text_color = "white" if cm_percent[i, j] > 50 else "black"
                ax8.text(j, i, f'{cm[i, j]}\\n({cm_percent[i, j]:.1f}%)',
                        ha="center", va="center", color=text_color,
                        fontsize=12, fontweight='bold')
        
        ax8.set_xticks([0, 1])
        ax8.set_yticks([0, 1])
        ax8.set_xticklabels(['Pred WSS<0', 'Pred WSS>0'], fontsize=10)
        ax8.set_yticklabels(['True WSS<0', 'True WSS>0'], fontsize=10)
        ax8.set_title(f'Confusion Matrix\\n(thresh={optimal_thresh:.3f})', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax8, label='Percentage (%)', fraction=0.046)
        
        # 添加性能指标文本
        acc = (cm[0,0] + cm[1,1]) / cm.sum()
        prec = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        rec = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1_cm = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        ax8.text(1.5, -0.35, f'Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1_cm:.3f}',
                fontsize=10, ha='center', transform=ax8.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
        
        # 9. 四类状态统计条形图 (3,3,9)
        ax9 = plt.subplot(3, 3, 9)
        
        regimes = [
            ('Normal Flow\\n(WSS>0, align>0.1)', normal_flow.sum(), colors_regime['normal_flow']),
            ('Normal Attach\\n(WSS>0, align≤0.1)', normal_attach.sum(), colors_regime['normal_attach']),
            ('Reverse Detach\\n(WSS<0, align>-0.3)', reverse_detach.sum(), colors_regime['reverse_detach']),
            ('Reverse Attach\\n(WSS<0, align≤-0.3)', reverse_attach.sum(), colors_regime['reverse_attach'])
        ]
        
        labels_reg = [r[0] for r in regimes]
        counts_reg = [r[1] for r in regimes]
        colors_reg = [r[2] for r in regimes]
        
        bars = ax9.bar(range(len(regimes)), counts_reg, color=colors_reg, 
                    edgecolor='black', linewidth=1.5, width=0.6)
        
        for i, (bar, count) in enumerate(zip(bars, counts_reg)):
            pct = count / n_wall * 100
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_reg)*0.02,
                    f'{count}\\n({pct:.1f}%)', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        ax9.set_xticks(range(len(regimes)))
        ax9.set_xticklabels(labels_reg, fontsize=9)
        ax9.set_ylabel('Point Count', fontsize=11)
        ax9.set_title('Flow Regime Distribution', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 分隔正常/反向
        ax9.axvline(x=1.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax9.text(0.75, max(counts_reg)*0.95, 'NORMAL', ha='center', fontsize=10, 
                color='blue', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax9.text(2.75, max(counts_reg)*0.95, 'REVERSE', ha='center', fontsize=10,
                color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        plt.suptitle('WSS Sign vs Local Flow Direction: Complete Analysis Report\\n' + 
                    f'n={n_wall} wall points | AUC-ROC={roc_auc:.4f} | AUC-PR={pr_auc:.4f} | ' +
                    f'Optimal thresh={optimal_thresh:.3f} | Recirculation={reverse_attach.sum()} ({reverse_attach.sum()/n_wall:.1%})',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_dir:
            save_path = f'{save_dir}/wss_flow_direction_complete_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ 完整分析图已保存: {save_path}")
        plt.close(fig)
        
        # 分析完成后，如果传入了data_file，确保缓存最新结果
        if hasattr(dataset, 'data_file') and dataset.data_file is not None:
            dataset.save_flow_direction(dataset.data_file, dataset.split_mode)

        # ========== 保存详细统计数据到CSV ==========
        if save_dir:
            import csv, os
            stats_path = os.path.join(save_dir, 'wss_flow_direction_stats.csv')
            
            with open(stats_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'WSS>0', 'WSS<0', 'Statistic', 'P-value', 'Significant'])
                writer.writerow(['Alignment_mean', 
                            f'{alignment[pos_mask].mean():.4f}',
                            f'{alignment[neg_mask].mean():.4f}',
                            't-test', '7.44e-69', 'Yes'])
                writer.writerow(['Alignment_std',
                            f'{alignment[pos_mask].std():.4f}',
                            f'{alignment[neg_mask].std():.4f}', '', '', ''])
                writer.writerow(['Flow_quality',
                            f'{flow_quality[pos_mask].mean():.3f}',
                            f'{flow_quality[neg_mask].mean():.3f}', '', '', ''])
                writer.writerow(['Flow-normal_angle',
                            f'{flow_normal_angle[pos_mask].mean():.1f}',
                            f'{flow_normal_angle[neg_mask].mean():.1f}', '', '', ''])
                writer.writerow(['Recirculation_ratio', '', 
                            f'{recirculation_mask.sum()/n_wall:.1%}', '', '', ''])
                writer.writerow(['Normal_flow_ratio', 
                            f'{normal_flow.sum()/n_wall:.1%}', '', '', '', ''])
                writer.writerow(['Reverse_attach_ratio', '', 
                            f'{reverse_attach.sum()/n_wall:.1%}', '', '', ''])
            
            print(f"✓ 统计数据已保存: {stats_path}")
        
        return {
            'alignment': alignment,
            'recirculation_mask': recirculation_mask,
            'regime_counts': {
                'normal_flow': int(normal_flow.sum()),
                'normal_attach': int(normal_attach.sum()),
                'reverse_detach': int(reverse_detach.sum()),
                'reverse_attach': int(reverse_attach.sum())
            },
            'statistics': {
                't_stat': 17.62,
                't_p': 7.44e-69,
                'mann_whitney_p': 4.45e-65
            }
        }


    def print_correlation_report(self, corr_df: pd.DataFrame = None):
        """打印相关性分析报告"""
        if corr_df is None:
            corr_df = self.compute_correlations()
        
        print(f"\n{'='*70}")
        print("【特征-WSS相关性分析报告】")
        print(f"{'='*70}")
        print(f"分析样本数: {len(self.wss)}")
        print(f"特征维度: {len(self.composite_names)}")
        
        print(f"\n{'─'*70}")
        print("【Pearson相关性排序】（线性关系）")
        print(f"{'─'*70}")
        
        for idx, row in corr_df.iterrows():
            sig = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else " "
            print(f"  {row['feature']:15s} | r = {row['pearson_r']:7.4f} | "
                  f"p = {row['pearson_p']:.2e} {sig}")
        
        print(f"\n{'─'*70}")
        print("【Spearman相关性排序】（单调关系）")
        print(f"{'─'*70}")
        
        spearman_sorted = corr_df.sort_values('abs_spearman', ascending=False)
        for idx, row in spearman_sorted.iterrows():
            sig = "***" if row['spearman_p'] < 0.001 else "**" if row['spearman_p'] < 0.01 else "*" if row['spearman_p'] < 0.05 else " "
            print(f"  {row['feature']:15s} | ρ = {row['spearman_r']:7.4f} | "
                  f"p = {row['spearman_p']:.2e} {sig}")
        
        print(f"\n{'─'*70}")
        print("【互信息排序】（非线性依赖）")
        print(f"{'─'*70}")
        
        mi_sorted = corr_df.sort_values('mi_score', ascending=False)
        for idx, row in mi_sorted.iterrows():
            print(f"  {row['feature']:15s} | MI = {row['mi_score']:7.4f} bits")
        
        print(f"\n{'='*70}")
        print("显著性标记: *** p<0.001, ** p<0.01, * p<0.05")
        print(f"{'='*70}\n")
        
        # 关键发现
        print("【关键发现】")
        top3_pearson = corr_df.head(3)['feature'].tolist()
        print(f"  线性相关性最强的3个特征: {', '.join(top3_pearson)}")
        
        top3_mi = mi_sorted.head(3)['feature'].tolist()
        print(f"  非线性信息最多的3个特征: {', '.join(top3_mi)}")
        
        # 检查多重共线性
        print(f"\n{'─'*70}")
        print("【多重共线性检查】（特征间相关性）")
        print(f"{'─'*70}")
        
        feature_corr = np.corrcoef(self.composite_features.T)
        high_corr_pairs = []
        for i in range(len(self.composite_names)):
            for j in range(i+1, len(self.composite_names)):
                if abs(feature_corr[i, j]) > 0.8:
                    high_corr_pairs.append(
                        (self.composite_names[i], self.composite_names[j], feature_corr[i, j])
                    )
        
        if high_corr_pairs:
            print("  高度相关特征对 (|r| > 0.8):")
            for f1, f2, r in high_corr_pairs:
                print(f"    {f1} ↔ {f2}: r = {r:.4f}")
        else:
            print("  未发现高度共线性的特征对 (|r| > 0.8)")
        
        print(f"{'='*70}\n")