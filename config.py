"""配置和全局设置"""

import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，加速绘图

# ==================== GPU 加速全局设置 ====================
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# ==================== 随机种子 ====================
torch.manual_seed(42)
np.random.seed(42)

# ==================== 实验选择（修改此项切换实验） ====================
# 消融实验（no_geom = baseline, 18-dim 纯物理特征）:
#   with_geom | no_sign | no_sep_geom | no_physics | no_sign_no_geom | flow_only
# 对比实验:  pure_mlp_xyz | no_geometry_mlp | coupled_head | coupled_head_no_geom | xgboost | rf
# 点云对比:  pointnet_xyz | pointnet | pointnetpp_xyz | pointnetpp | dgcnn_xyz | dgcnn | gat_xyz | gat
EXPERIMENT = 'no_geom'

# ==================== 实验预设配置 ====================
_EXP = {
    # ---- 消融实验（no_geom = baseline, 18-dim 纯物理特征） ----
    'with_geom': {
        'name': 'ablation_with_geometry',
        'description': '消融：在 no_geom baseline (18-dim) 的幅值分支中加入 4 维 Wall Topography 特征，22-dim 输入',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': True,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,  # None = PyTorch NN
    },
    'no_sign': {
        'name': 'ablation_no_sign',
        'description': '消融：移除符号分支，仅幅值输出',
        'use_sign_branch': False,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': True,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'no_sep_geom': {
        'name': 'ablation_no_sep_geom',
        'description': '消融：移除12维分离几何特征',
        'use_sign_branch': True,
        'use_sep_geom': False,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': True,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'no_physics': {
        'name': 'ablation_no_physics',
        'description': '消融：移除物理梯度约束损失',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': True,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'no_geom': {
        'name': 'ablation_no_geometry',
        'description': 'Baseline — 纯物理特征 wss_net (18-dim) + 解耦符号分支 + 物理约束',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,  # ← 幅值网络不含 geom
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    # ---- COMSOL 理想几何验证（重训练对照，flags 与 no_geom 完全一致）----
    # 用法: python main.py -e no_geom_comsol --data-dir data_comsol
    'no_geom_comsol': {
        'name': 'validation_comsol_no_geom',
        'description': 'COMSOL 验证对照 — no_geom 在理想几何(直管/弯管/分叉)上重训练',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'no_geom_logo1': {
        'name': 'validation_logo1',
        'description': 'LOGO fold 1/6: 5-geometry training, 1 held out',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'no_geom_logo2': {
        'name': 'validation_logo2',
        'description': 'LOGO fold 2/6: 5-geometry training, 1 held out',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'no_geom_logo3': {
        'name': 'validation_logo3',
        'description': 'LOGO fold 3/6: 5-geometry training, 1 held out',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'no_geom_logo4': {
        'name': 'validation_logo4',
        'description': 'LOGO fold 4/6: 5-geometry training, 1 held out',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'no_geom_logo5': {
        'name': 'validation_logo5',
        'description': 'LOGO fold 5/6: 5-geometry training, 1 held out',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'no_geom_logo6': {
        'name': 'validation_logo6',
        'description': 'LOGO fold 6/6: 5-geometry training, 1 held out',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    # ---- COMSOL 验证 v2：新课程 3 case（taper→bifurcation2→stenosis，符号case最后）----
    # 用法: python main.py -e no_geom_v2 --data-dir data_comsol
    'no_geom_v2': {
        'name': 'validation_v2curv_no_geom',
        'description': 'COMSOL 验证v2 — 新课程(渐缩管→分叉mk2→狭窄管)训练, 符号富集case最后防冲刷',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    # ── 退火调度对照实验 (基于 no_geom) ──
    'no_geom_anneal': {
        'name': 'ablation_no_geometry_anneal',
        'description': '退火对照 A 组: no_geom + Warmup→Cosine 退火',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
        'scheduler_type': 'warmup_cosine',
    },
    'no_geom_const_lr': {
        'name': 'ablation_no_geometry_const_lr_full',
        'description': '退火对照 B 组全量: no_geom + Adam + 恒定 lr (无 warmup 无 cosine), 113 case, 早停对齐',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
        'scheduler_type': 'constant',
        'optimizer_type': 'adam',
    },
    # ── 退火对照实验 扩充 6 case (临时调用, 已注释) ──
    # 'no_geom_anneal_ext': {
    #     'name': 'ablation_no_geometry_anneal_ext',
    #     'description': '退火对照 A 组扩充: no_geom + Warmup→Cosine, 新 6 case',
    #     'use_sign_branch': True,
    #     'use_sep_geom': True,
    #     'use_physics_loss': True,
    #     'use_coords_only': False,
    #     'use_geometry_features': False,
    #     'use_velocity_pressure': True,
    #     'decouple_head': True,
    #     'ml_algorithm': None,
    #     'scheduler_type': 'warmup_cosine',
    # },
    # 'no_geom_const_lr_ext': {
    #     'name': 'ablation_no_geometry_const_lr_ext',
    #     'description': '退火对照 B 组扩充: no_geom + Adam + 恒定 lr, 新 6 case',
    #     'use_sign_branch': True,
    #     'use_sep_geom': True,
    #     'use_physics_loss': True,
    #     'use_coords_only': False,
    #     'use_geometry_features': False,
    #     'use_velocity_pressure': True,
    #     'decouple_head': True,
    #     'ml_algorithm': None,
    #     'scheduler_type': 'constant',
    #     'optimizer_type': 'adam',
    # },
    'no_sign_no_geom': {
        'name': 'ablation_no_sign_no_geometry',
        'description': '消融：同时移除符号分支+幅值几何特征，wss_net仅18维输入',
        'use_sign_branch': False,
        'use_sep_geom': True,        # 无符号分支，此标志无影响
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,  # ← 幅值网络不含4维geom，18维
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    # ---- 对比实验 ----
    'pure_mlp_xyz': {
        'name': 'baseline_pure_mlp_xyz',
        'description': '对比：纯坐标MLP（验证坐标记忆假说）',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': True,
        'use_geometry_features': False,
        'use_velocity_pressure': False,
        'decouple_head': False,
        'ml_algorithm': None,
    },
    'no_geometry_mlp': {
        'name': 'baseline_no_geometry_mlp',
        'description': '对比：无几何特征MLP（验证几何特征必要性）— 符号分支仅用magnitude.detach()',
        'use_sign_branch': True,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'ml_algorithm': None,
    },
    'coupled_head': {
        'name': 'baseline_coupled_head',
        'description': '对比：耦合头（验证解耦必要性）— 符号分支用magnitude而非magnitude.detach()，梯度耦合',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': True,
        'use_velocity_pressure': True,
        'decouple_head': True,
        'detach_magnitude': False,
        'ml_algorithm': None,
    },
    'coupled_head_no_geom': {
        'name': 'ablation_coupled_head_no_geometry',
        'description': '对比：耦合头无几何（18-dim）— 隔离梯度耦合效应，与 no_geom baseline 对照',
        'use_sign_branch': True,
        'use_sep_geom': True,
        'use_physics_loss': True,
        'use_coords_only': False,
        'use_geometry_features': False,  # 18-dim，无几何噪声放大
        'use_velocity_pressure': True,
        'decouple_head': True,
        'detach_magnitude': False,       # 梯度耦合
        'ml_algorithm': None,
    },
    'xgboost': {
        'name': 'baseline_xgboost',
        'description': '对比：XGBoost（MSE损失，验证深度网络非线性优势）',
        'use_sign_branch': False,
        'use_sep_geom': True,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': True,
        'use_velocity_pressure': True,
        'decouple_head': False,
        'ml_algorithm': 'xgboost',
    },
    'rf': {
        'name': 'baseline_rf',
        'description': '对比：随机森林（MSE损失，验证深度网络非线性优势）',
        'use_sign_branch': False,
        'use_sep_geom': True,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': True,
        'use_velocity_pressure': True,
        'decouple_head': False,
        'ml_algorithm': 'random_forest',
    },
    'raw_neighbor_mlp': {
        'name': 'comparison_raw_neighbor_mlp',
        'description': '对比：原始K近邻MLP — 壁面+近邻内部点(xyz+|v|+p)直接输入MLP，不手工提取v/d/n特征，验证手工特征是否等价于NN自主学习',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': False,
        'ml_algorithm': None,
        'use_raw_neighbors': True,   # NEW: 使用原始近邻特征替代手工特征
        'raw_neighbor_k': 10,         # 每个壁面点取10个最近内部点
    },
    'flow_only': {
        'name': 'ablation_flow_only',
        'description': '消融：仅速度+压力预测WSS（无几何/坐标/法向量/WSS参考）',
        'use_sign_branch': True,
        'use_sep_geom': False,
        'use_physics_loss': False,  # 物理损失依赖几何特征索引(cols 4-10)，flow_only仅有2维velocity+pressure
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'use_velocity_pressure_only': True,  # NEW: 幅值网络仅用 velocity_real + pressure_real
        'decouple_head': True,
        'ml_algorithm': None,
    },
    # ---- 点云/GNN 对比实验 ----
    # 轻量级 PointNet (vanilla) — 仅壁面点，O(N)，比 PointNet++ 快 50-100×
    'pointnet_xyz': {
        'name': 'comparison_pointnet_xyz',
        'description': '对比：PointNet (vanilla) 壁面纯坐标(x,y,z) — O(N)轻量黑盒，比PointNet++快50×',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': False,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'pointnet',
        'pointnet_input': 'coords_only',
    },
    'pointnet': {
        'name': 'comparison_pointnet',
        'description': '对比：PointNet (vanilla) 壁面坐标+速度+压力 — O(N)轻量端到端学习',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'pointnet',
        'pointnet_input': 'coords_features',
    },
    # PointNet 组合点云（壁面+内部）— 与 PointNet++/DGCNN/GAT 同条件对比
    'pointnet_combined_xyz': {
        'name': 'comparison_pointnet_combined_xyz',
        'description': '对比：PointNet (vanilla) 组合点云纯坐标(x,y,z,is_wall) — 与PointNet++同输入条件',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': False,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'pointnet_combined',
        'pointnet_input': 'coords_only',    # xyz + is_wall flag
    },
    'pointnet_combined': {
        'name': 'comparison_pointnet_combined',
        'description': '对比：PointNet (vanilla) 组合点云坐标+速度+压力+is_wall — 与GNN同输入条件',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'pointnet_combined',
        'pointnet_input': 'coords_features',  # xyz + |v| + p + is_wall
    },
    # 重量级点云方法 — 壁面+内部组合点云，内部点参与消息传递
    'pointnetpp_xyz': {
        'name': 'comparison_pointnetpp_xyz',
        'description': '对比：PointNet++ 壁面+内部组合点云(xyz+is_wall) — 纯几何推断WSS',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': False,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'pointnetpp',
        'pointnetpp_input': 'coords_only',
    },
    'pointnetpp': {
        'name': 'comparison_pointnetpp',
        'description': '对比：PointNet++ 壁面+内部组合点云(xyz+|v|+p+is_wall) — 含内部流场',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'pointnetpp',
        'pointnetpp_input': 'coords_features',
    },
    # ---- DGCNN 对比实验 ----
    'dgcnn_xyz': {
        'name': 'comparison_dgcnn_xyz',
        'description': '对比：DGCNN 壁面+内部组合点云(xyz+is_wall) — 纯几何EdgeConv',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': False,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'dgcnn',
        'gnn_input': 'coords_only',
    },
    'dgcnn': {
        'name': 'comparison_dgcnn',
        'description': '对比：DGCNN 壁面+内部组合点云(xyz+|v|+p+is_wall) — 含内部流场EdgeConv',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'dgcnn',
        'gnn_input': 'coords_features',
    },
    # ---- GAT 对比实验 ----
    'gat_xyz': {
        'name': 'comparison_gat_xyz',
        'description': '对比：GAT 壁面+内部组合点云(xyz+is_wall) — 纯几何图注意力',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': False,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'gat',
        'gnn_input': 'coords_only',
    },
    'gat': {
        'name': 'comparison_gat',
        'description': '对比：GAT 壁面+内部组合点云(xyz+|v|+p+is_wall) — 含内部流场图注意力',
        'use_sign_branch': False,
        'use_sep_geom': False,
        'use_physics_loss': False,
        'use_coords_only': False,
        'use_geometry_features': False,
        'use_velocity_pressure': True,
        'decouple_head': False,
        'ml_algorithm': None,
        'model_type': 'gat',
        'gnn_input': 'coords_features',
    },
}

# ---- LOPO 折实验条目（cross_patient_lopo_plan.md §4.2；按折隔离 SAVE_DIR，
# ---- 避免多折并行时 latest/session_final 互相覆盖）----
for _i in range(1, 74):
    _EXP['no_geom_lopo%02d' % _i] = dict(
        _EXP['no_geom'], name='lopo_fold%02d' % _i,
        description='LOPO fold %d/73: leave-one-patient-out' % _i)

# 阶段1 pilot 专用：fold_42 的 cap-500 收敛对照（独立输出目录）
_EXP['no_geom_lopo42_cap500'] = dict(
    _EXP['no_geom'], name='lopo_fold42_cap500',
    description='LOPO fold42 cap-500 pilot (convergence comparison)')

# lean 模式数值/速度对照（同折同 cap，LOPO_LEAN=1）
_EXP['no_geom_lopo42_lean'] = dict(
    _EXP['no_geom'], name='lopo_fold42_lean',
    description='LOPO fold42 cap-500 LEAN validation run')

# ---- 物理约束 best-epoch 对照重训（cap5000, lean）----
_EXP['no_geom_bestic'] = dict(
    _EXP['no_geom'], name='bestic_no_geom',
    description='no_geom best-epoch retrain (cap5000 lean)')
_EXP['no_physics_bestic'] = dict(
    _EXP['no_physics'], name='bestic_no_physics',
    description='no_physics best-epoch retrain (cap5000 lean)')

# 当前实验配置
EXP_CONFIG = _EXP[EXPERIMENT]

# ==================== 路径配置 ====================
# 所有实验输出写入 <BASE_DIR>/<experiment_name>/；默认 ./runs（可用环境变量
# PHYDENET_RUNS 覆盖），保证仓库克隆到任何机器均可直接运行
BASE_DIR = os.environ.get('PHYDENET_RUNS', './runs')
SAVE_DIR = os.path.join(BASE_DIR, EXP_CONFIG['name'])
SAVE_DIR_ORIGINAL = SAVE_DIR
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 设备配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_device_info():
    """打印设备信息"""
    print(f"使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ==================== 训练配置 ====================
class TrainConfig:
    """训练配置类"""
    TOTAL_EPOCHS = int(os.environ.get('LOPO_TOTAL_EPOCHS', '3000'))
    # 每-case epoch 上限；LOPO 试点/战役用环境变量按进程覆盖，避免反复改文件
    SAVE_INTERVAL = 100
    START_EPOCH = 100
    PRINT_EVERY = 50

    # 学习率配置
    LR_INITIAL = 1e-4
    LR_WARMUP_STEPS = 1000
    LR_MIN = 1e-6

    # 调度器类型: 'warmup_cosine' (默认) | 'constant'
    # 可被 EXP_CONFIG['scheduler_type'] 覆盖
    SCHEDULER_TYPE = 'warmup_cosine'

    # 早停配置 (恢复原始, 与 ablation_no_geometry 对齐)
    EARLY_STOP_PATIENCE = 20
    EARLY_STOP_MIN_DELTA = 0.001

    # 损失权重
    LAMBDA_WSS = 1.0
    LAMBDA_PHYS = 0.1
    LAMBDA_SMOOTH = 0.05

    # 梯度裁剪
    GRAD_CLIP_FACTOR = 0.01

# ==================== 数据配置 ====================
class DataConfig:
    """数据配置类"""
    Z_SPLIT_RATIO = 0.8
    BOUNDARY_LAYER_THRESHOLD = 3.0  # mm
    N_NEIGHBORS = 3
    K_NEIGHBORS_WALL = 10  # 壁面点邻居数

    # 切分策略
    SPLIT_STRATEGY = 'adaptive_wss_bidirectional'
    MIN_TRAIN_RATIO = 0.6
    MAX_VAL_RATIO = 0.3
    N_CANDIDATES = 100

# ==================== 流体参数 ====================
FLUID_PARAMS = {
    'mu': 0.00345,    # 动力粘度 Pa·s
    'rho': 1000.0,    # 密度 kg/m^3
}
