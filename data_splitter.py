"""自适应WSS分布数据切分器"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class AdaptiveWSSSplitter:
    """
    自适应WSS分布切分器 - 双向搜索版本
    同时评估两种切分方向，选择分布相似性最高的切分方式
    """
    
    DIRECTION_BOTTOM_UP = 'bottom_up'   # 下部分训练，上部分验证
    DIRECTION_TOP_DOWN = 'top_down'     # 上部分训练，下部分验证
    
    def __init__(self, coords: np.ndarray, wss_values: np.ndarray, z_axis: int = 2,
                 min_train_ratio: float = 0.6, max_val_ratio: float = 0.3,
                 n_candidates: int = 100):
        """
        Args:
            coords: 坐标数组 (N, 3)
            wss_values: WSS值 (N, 1) 或 (N,)
            z_axis: Z轴维度索引
            min_train_ratio: 训练集最小比例
            max_val_ratio: 验证集最大比例
            n_candidates: 候选切分点数量
        """
        self.coords = coords
        self.wss_raw = wss_values.flatten().copy()
        self.z = coords[:, z_axis]
        self.z_axis = z_axis
        self.min_train_ratio = min_train_ratio
        self.max_val_ratio = max_val_ratio
        self.n_candidates = n_candidates
        
        # 按Z排序
        self.sort_idx = np.argsort(self.z)
        self.z_sorted = self.z[self.sort_idx]
        self.wss_sorted = self.wss_raw[self.sort_idx]
        self.coords_sorted = self.coords[self.sort_idx]
        
        self.best_result = None

    def compute_distribution_similarity(self, wss_train: np.ndarray, 
                                        wss_val: np.ndarray) -> Dict[str, float]:
        """计算两个WSS分布的相似性指标"""
        
        if len(wss_train) < 10 or len(wss_val) < 10:
            return {'composite': -1e6, 'ks_stat': 1.0, 'overlap': 0.0}
        
        # KS统计量
        ks_stat, p_value = stats.ks_2samp(wss_train, wss_val)
        ks_similarity = max(0, 1 - ks_stat)
        
        # 直方图重叠度
        hist_range = (min(wss_train.min(), wss_val.min()), 
                      max(wss_train.max(), wss_val.max()))
        hist_train, bin_edges = np.histogram(wss_train, bins=30, 
                                              range=hist_range, density=True)
        hist_val, _ = np.histogram(wss_val, bins=bin_edges, 
                                   range=hist_range, density=True)
        overlap = np.sum(np.minimum(hist_train, hist_val)) * np.diff(bin_edges).mean()
        
        # 均值相对差异
        mean_rel_diff = abs(wss_train.mean() - wss_val.mean()) / (wss_train.std() + 1e-8)
        mean_similarity = max(0, 1 - mean_rel_diff)
        
        # 标准差相对差异
        std_rel_diff = abs(wss_train.std() - wss_val.std()) / (wss_train.std() + 1e-8)
        std_similarity = max(0, 1 - std_rel_diff)
        
        # 分位数相似度
        quants = [25, 50, 75]
        train_quants = np.percentile(wss_train, quants)
        val_quants = np.percentile(wss_val, quants)
        quantile_mae = np.mean(np.abs(train_quants - val_quants))
        quantile_similarity = max(0, 1 - quantile_mae / (wss_train.std() + 1e-8))
        
        # 分布范围覆盖率
        train_min, train_max = wss_train.min(), wss_train.max()
        val_min, val_max = wss_val.min(), wss_val.max()
        covered_min = max(train_min, val_min)
        covered_max = min(train_max, val_max)
        coverage = (covered_max - covered_min) / (val_max - val_min + 1e-8) if covered_max > covered_min else 0.0
        
        # 综合评分
        composite = (
            0.25 * ks_similarity +
            0.25 * overlap +
            0.15 * mean_similarity +
            0.10 * std_similarity +
            0.15 * quantile_similarity +
            0.10 * coverage
        )
        
        return {
            'composite': composite,
            'ks_similarity': ks_similarity,
            'histogram_overlap': overlap,
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'quantile_similarity': quantile_similarity,
            'coverage': coverage,
            'ks_stat': ks_stat,
            'p_value': p_value
        }

    def check_coverage_constraint(self, wss_train: np.ndarray, wss_val: np.ndarray,
                                  margin_ratio: float = 0.05) -> Tuple[bool, float, float, Tuple]:
        """检查验证集WSS是否被训练集完全覆盖"""
        train_min, train_max = wss_train.min(), wss_train.max()
        val_min, val_max = wss_val.min(), wss_val.max()
        
        train_range = train_max - train_min
        margin = margin_ratio * train_range
        
        lower_gap = max(0, (train_min - margin) - val_min)
        upper_gap = max(0, val_max - (train_max + margin))
        
        is_satisfied = (lower_gap == 0) and (upper_gap == 0)
        violation = lower_gap + upper_gap
        
        val_range = val_max - val_min
        covered_range = max(0, min(train_max, val_max) - max(train_min, val_min))
        coverage_score = covered_range / val_range if val_range > 0 else 1.0
        
        return is_satisfied, coverage_score, violation, (train_min, train_max, val_min, val_max)

    def _evaluate_split_direction(self, direction: str, 
                                  candidate_indices: np.ndarray) -> List[Dict]:
        """评估单一方向的候选切分点"""
        n_total = len(self.z_sorted)
        results = []
        
        for idx in candidate_indices:
            z_thresh = self.z_sorted[idx]
            
            if direction == self.DIRECTION_BOTTOM_UP:
                train_mask = self.z <= z_thresh
                val_mask = self.z > z_thresh
                train_desc = "Z <= threshold (下部)"
                val_desc = "Z > threshold (上部)"
            else:
                train_mask = self.z >= z_thresh
                val_mask = self.z < z_thresh
                train_desc = "Z >= threshold (上部)"
                val_desc = "Z < threshold (下部)"
            
            wss_train = self.wss_raw[train_mask]
            wss_val = self.wss_raw[val_mask]
            
            train_ratio = len(wss_train) / n_total
            val_ratio = len(wss_val) / n_total
            
            if train_ratio < self.min_train_ratio or val_ratio > self.max_val_ratio:
                continue
            
            similarity = self.compute_distribution_similarity(wss_train, wss_val)
            is_covered, coverage_score, violation, ranges = self.check_coverage_constraint(
                wss_train, wss_val
            )
            
            coverage_penalty = 0 if is_covered else (1.0 - coverage_score) * 2.0
            final_score = similarity['composite'] - coverage_penalty
            
            results.append({
                'direction': direction,
                'z_threshold': z_thresh,
                'train_mask': train_mask,
                'val_mask': val_mask,
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'train_desc': train_desc,
                'val_desc': val_desc,
                'similarity': similarity,
                'is_covered': is_covered,
                'coverage_score': coverage_score,
                'wss_ranges': ranges,
                'final_score': final_score
            })
        
        return results

    def find_optimal_split(self, prefer: str = 'middle', verbose: bool = True,
                          return_all_candidates: bool = False):
        """双向搜索最佳切分点"""
        n_total = len(self.z_sorted)
        
        min_idx = int(n_total * 0.15)
        max_idx = int(n_total * 0.85)
        candidate_indices = np.linspace(min_idx, max_idx, self.n_candidates, dtype=int)
        
        print(f"\n{'='*70}")
        print("【双向WSS分布自适应切分】")
        print(f"评估 {self.n_candidates} 个候选点 × 2 个方向")
        
        bottom_up_results = self._evaluate_split_direction(
            self.DIRECTION_BOTTOM_UP, candidate_indices
        )
        top_down_results = self._evaluate_split_direction(
            self.DIRECTION_TOP_DOWN, candidate_indices
        )
        
        all_results = bottom_up_results + top_down_results
        
        if len(all_results) == 0:
            raise ValueError("未找到满足约束的切分方案")
        
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 处理平局
        best_score = all_results[0]['final_score']
        top_candidates = [r for r in all_results if abs(r['final_score'] - best_score) < 0.001]
        
        if len(top_candidates) > 1:
            if prefer == 'middle':
                top_candidates.sort(key=lambda x: abs(x['train_ratio'] - 0.75))
        
        best_result = top_candidates[0]
        self.best_result = best_result
        
        if verbose:
            self._print_results(best_result, all_results[:10])
        
        if return_all_candidates:
            return best_result, all_results
        return best_result

    def _print_results(self, best: Dict, top_candidates: List[Dict]):
        """打印详细结果"""
        print(f"\n{'='*70}")
        print("【最佳切分方案】")
        
        dir_name = "从下往上" if best['direction'] == self.DIRECTION_BOTTOM_UP else "从上往下"
        print(f"\n切分方向: {dir_name}")
        print(f"  Z切分点: {best['z_threshold']:.4f}")
        
        tmin, tmax, vmin, vmax = best['wss_ranges']
        print(f"\nWSS分布范围:")
        print(f"  Train: [{tmin:.4f}, {tmax:.4f}]")
        print(f"  Val:   [{vmin:.4f}, {vmax:.4f}]")
        
        status = "✅ 完全覆盖" if best['is_covered'] else "⚠️ 覆盖不完全"
        print(f"  {status} (覆盖率: {best['coverage_score']:.1%})")
        
        sim = best['similarity']
        print(f"\n分布相似性指标:")
        print(f"  综合评分: {sim['composite']:.4f}")
        print(f"  KS相似度: {sim['ks_similarity']:.4f}")
        print(f"  直方图重叠: {sim['histogram_overlap']:.4f}")

    def get_split_masks(self, result: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """获取切分掩码"""
        if result is None:
            result = self.best_result
        if result is None:
            raise ValueError("请先运行 find_optimal_split()")
        return result['train_mask'], result['val_mask']

    def get_split_info(self) -> Dict:
        """生成切分信息字典"""
        if self.best_result is None:
            raise ValueError("请先运行 find_optimal_split()")
        
        return {
            'direction': self.best_result['direction'],
            'z_threshold': self.best_result['z_threshold'],
            'is_covered': self.best_result['is_covered'],
            'coverage_score': self.best_result['coverage_score'],
            'train_wss_range': self.best_result['wss_ranges'][:2],
            'val_wss_range': self.best_result['wss_ranges'][2:],
            'similarity_composite': self.best_result['similarity']['composite']
        }


def visualize_split(coords: np.ndarray, wss: np.ndarray, z_threshold: float,
                    z_axis: int = 2, save_path: str = None):
    """可视化切分结果"""
    z = coords[:, z_axis]
    train_mask = z <= z_threshold
    val_mask = z > z_threshold
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Z-WSS散点图
    ax = axes[0, 0]
    ax.scatter(z[train_mask], wss[train_mask], c='blue', alpha=0.5, s=10, label='Train')
    ax.scatter(z[val_mask], wss[val_mask], c='red', alpha=0.5, s=10, label='Val')
    ax.axvline(x=z_threshold, color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('Z Coordinate')
    ax.set_ylabel('WSS')
    ax.set_title('WSS Distribution along Z-axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # WSS分布直方图
    ax = axes[0, 1]
    ax.hist(wss[train_mask], bins=30, alpha=0.6, color='blue', label='Train', density=True)
    ax.hist(wss[val_mask], bins=30, alpha=0.6, color='red', label='Val', density=True)
    ax.set_xlabel('WSS')
    ax.set_ylabel('Density')
    ax.set_title('WSS Distribution Comparison')
    ax.legend()
    
    # 空间分布
    ax = axes[1, 0]
    sc = ax.scatter(coords[train_mask, 0], coords[train_mask, 1], 
                   c=wss[train_mask], cmap='Blues', alpha=0.6, s=15)
    ax.scatter(coords[val_mask, 0], coords[val_mask, 1], 
              c=wss[val_mask], cmap='Reds', alpha=0.6, s=15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Spatial Distribution (XY plane)')
    plt.colorbar(sc, ax=ax, label='WSS')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {save_path}")
    
    plt.show()
    return fig