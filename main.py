"""主程序入口（添加特征分析）

示例命令:
  python main.py                                    # 使用 config.py 中的 EXPERIMENT 默认值
  python main.py -e no_geom                         # 运行 Baseline 实验
  python main.py -e with_geom                       # 运行 ablation（baseline + geometry）

可用实验:
  Baseline:      no_geom (18-dim, 纯物理特征)
  消融实验:      with_geom | no_sign | no_sep_geom | no_physics | no_sign_no_geom | flow_only
  对比实验:      pure_mlp_xyz | no_geometry_mlp | coupled_head | coupled_head_no_geom | xgboost | rf
  点云对比:      pointnet_xyz | pointnet | pointnetpp_xyz | pointnetpp | dgcnn_xyz | dgcnn | gat_xyz | gat
  Note: with_geom 即旧 baseline——在 no_geom 上加 Wall Topography 的消融变体
"""

import os
import sys
import glob
import argparse
import torch

# 导入各模块
import config as cfg
from config import DEVICE, SAVE_DIR, SAVE_DIR_ORIGINAL, DataConfig, print_device_info, SCRIPT_DIR, EXP_CONFIG

# LOPO lean 模式：跳过训练途中的绘图与特征分析（训练数学零改动，只省 I/O）
LEAN = os.environ.get('LOPO_LEAN', '') == '1'
from utils import backup_model, EarlyStoppingTrainer
from dataset import CarotidDataset
from models import maybe_compile, create_model
from trainer import DirectWSSTrainer
from evaluator import DirectWSSEvaluator
from feature_analyzer import FeatureAnalyzer  # 新增
from baseline_ml_trainer import train_ml_baseline


def analyze_features(dataset, model=None, save_dir=None, stage="初始"):
    """
    执行特征分析
    
    Args:
        dataset: 数据集
        model: 训练好的模型（可选）
        save_dir: 保存目录
        stage: 分析阶段描述
    """
    print(f"\n{'='*70}")
    print(f"【{stage}特征分析】")
    print(f"{'='*70}")
    
    analyzer = FeatureAnalyzer(dataset)

    # 生成完整报告（包含曲率-WSS符号分析）
    report = analyzer.generate_full_report(save_dir=save_dir)

    # 计算并打印相关性
    corr_df = analyzer.compute_correlations()
    analyzer.print_correlation_report(corr_df)
    
    # 绘制相关性图表
    corr_df = analyzer.plot_correlation_analysis(save_dir=save_dir)

    # ====== 新增：几何特征与负WSS分布分析 ======
    print(f"\n{'='*70}")
    print("【几何特征与负WSS空间分布分析】")
    print(f"{'='*70}")
    geom_stats = analyzer.plot_geometry_wss_analysis(model=model, save_dir=save_dir)
    # ============================================

    # 新增：WSS与局部流动方向分析
    print(f"\n{'='*70}")
    print("【WSS符号 vs 局部流动方向分析】")
    print(f"{'='*70}")
    
    # 计算流动方向（如果不存在）
    if not hasattr(dataset, 'wall_flow_direction'):
        dataset.compute_local_flow_direction(k_neighbors=15)
        dataset.compute_wss_direction_alignment()
    
    # 执行分析并绘图
    flow_stats = analyzer.analyze_wss_flow_direction(dataset, save_dir=save_dir)
    
    # 如果有模型，对比梯度重要性
    if model is not None:
        print(f"\n{'='*70}")
        print("【模型梯度重要性分析】")
        print(f"{'='*70}")
        comparison = analyzer.plot_feature_importance_comparison(model, save_dir=save_dir)
        return corr_df, comparison
    
    return corr_df, None


def main():
    """主函数"""
    # ===== 命令行参数解析 =====
    # 必须在访问 EXP_CONFIG 之前解析，因为其他模块已通过 from-import 持有引用
    # 使用 dict 原位更新（clear+update），确保所有模块引用的同一对象被修改
    parser = argparse.ArgumentParser(
        description='WSS 预测模型 — 消融实验 / 对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例:\n'
               '  python main.py -e no_geom\n'
               '  python main.py -e with_geom\n'
               '  python main.py                    # 使用 config.py 默认值 (no_geom)',
    )
    parser.add_argument(
        '-e', '--experiment', type=str, default=None,
        choices=list(cfg._EXP.keys()),
        help=f'实验名称（默认: config.py 中的 EXPERIMENT = {cfg.EXPERIMENT!r}）',
    )
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='数据目录（含 *_combined.csv，由 preprocess/vtu_to_combined.py 生成）',
    )
    args = parser.parse_args()

    # 如果命令行指定了实验，原位更新 cfg.EXP_CONFIG（其他模块持有同一引用）
    # 注意: EXP_CONFIG = _EXP[EXPERIMENT] 是同一对象，必须先 copy 再 clear+update
    if args.experiment is not None:
        cfg.EXPERIMENT = args.experiment
        new_config = cfg._EXP[args.experiment].copy()
        cfg.EXP_CONFIG.clear()
        cfg.EXP_CONFIG.update(new_config)
        cfg.SAVE_DIR = os.path.join(cfg.BASE_DIR, cfg.EXP_CONFIG['name'])
        cfg.SAVE_DIR_ORIGINAL = cfg.SAVE_DIR
        print(f"📌 命令行覆盖实验: {args.experiment} → {cfg.EXP_CONFIG['description']}")

    print("WSS预测模型：几何特征 + 真实速度/压力 -> WSS")
    print("="*60)
    
    print_device_info()
    
    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 备份脚本
    backup_model(backup_dir=cfg.SAVE_DIR, script_dir=SCRIPT_DIR)
    
    # 模型将在加载第一个数据集后创建（NN路径），或使用ML基线
    model = None
    ml_algorithm = EXP_CONFIG['ml_algorithm']

    # 获取数据文件
    folder_path = args.data_dir
    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
    
    # 处理统计
    success_files = []
    failed_files = []
    
    number_id_begin = 0
    csv_files = csv_files[number_id_begin:]
    # 遍历处理每个文件
    for number_id, file_path in enumerate(csv_files):
        print(f"\n{'='*70}")
        print(f"处理文件 [{number_id + number_id_begin}]: {file_path}")
        print(f"{'='*70}")
        
        # 加载训练和验证数据
        train_dataset = CarotidDataset(
            file_path,
            split_mode='train',
            z_split_ratio=DataConfig.Z_SPLIT_RATIO
        )
        
        val_dataset = CarotidDataset(
            file_path,
            split_mode='val',
            z_split_ratio=DataConfig.Z_SPLIT_RATIO
        )
        
        # 创建保存目录
        filename = os.path.basename(file_path)
        field_name = os.path.splitext(filename)[0]
        save_dir = os.path.join(cfg.SAVE_DIR_ORIGINAL, f"{number_id+number_id_begin}_{field_name}")
        os.makedirs(save_dir, exist_ok=True)

        # ===== ML 基线路径 =====
        if ml_algorithm is not None:
            print(f"\n{'='*60}")
            print(f"【{ml_algorithm.upper()} 基线训练】文件: {filename}")
            print(f"{'='*60}")
            train_ml_baseline(train_dataset, val_dataset, save_dir, algorithm=ml_algorithm)
            success_files.append({
                'id': number_id+number_id_begin,
                'filename': filename,
                'path': save_dir
            })
            continue

        # ===== NN 路径：首次创建模型 =====
        if model is None:
            input_dim = train_dataset.get_tensors()['wall_features'].shape[1]
            model = create_model(input_dim).to(DEVICE)
            print(f"\n模型参数量: {sum(p.numel() for p in model.parameters())}")
            print(f"输入维度: {input_dim}")

            model = maybe_compile(model, enabled=False)  # 当前模型含分支，graph break 反致慢

            # 点云模型（PointNet++/DGCNN/GAT）：无分支，compile 收益显著
            # 必须在 .to(DEVICE) 之后编译，否则 BatchNorm buffers 设备不匹配
            if hasattr(model, '_model_type') and model._model_type in ('pointnetpp', 'dgcnn', 'gat'):
                model = maybe_compile(model, enabled=True)
                print(f"✓ torch.compile 已启用（{model._model_type}）")

        # # ==================== 训练前特征分析 ====================
        # print("\n" + "="*60)
        # print("【训练前：原始特征相关性分析】")
        # print("="*60)
        
        # pretrain_corr, _ = analyze_features(
        #     train_dataset, 
        #     model=None, 
        #     save_dir=save_dir,
        #     stage="训练前"
        # )
        
        # 预训练评估
        print("\n" + "="*60)
        print("预训练评估")
        pretrain_evaluator = DirectWSSEvaluator(model, train_dataset)
        if LEAN:
            _, _, pretrain_r2, pretrain_r2_mag, *_ = pretrain_evaluator.evaluate()
        else:
            pretrain_r2, pretrain_r2_mag, *_ = pretrain_evaluator.plot_predictions(
                0, 100, 'train', save_dir
            )
        print(f"预训练R²: {pretrain_r2:.4f}")
        print(f"预训练R² (magnitude): {pretrain_r2_mag:.4f}")
        
        # 创建训练器
        trainer = DirectWSSTrainer(model, train_dataset, save_dir)
        
        # 训练循环
        from config import TrainConfig
        early_stopping = EarlyStoppingTrainer(
            patience=TrainConfig.EARLY_STOP_PATIENCE,
            min_delta=TrainConfig.EARLY_STOP_MIN_DELTA,
            mode='max'
        )
        
        current_epochs = 0
        final_corr_df = None
        final_comparison = None
        
        for epoch in range(TrainConfig.START_EPOCH, 
                            TrainConfig.TOTAL_EPOCHS, 
                            TrainConfig.SAVE_INTERVAL):
            
            interval = min(TrainConfig.SAVE_INTERVAL,
                            TrainConfig.TOTAL_EPOCHS - epoch)
            
            # 训练
            trainer.train(
                epochs=interval,
                print_every=TrainConfig.PRINT_EVERY,
                current_epochs=current_epochs
            )
            current_epochs += interval
            
            # 评估
            train_eval = DirectWSSEvaluator(model, train_dataset)
            if LEAN:
                _, _, train_r2, train_r2_mag, *_ = train_eval.evaluate()
            else:
                train_r2, train_r2_mag, *_ = train_eval.plot_predictions(
                    epoch, TrainConfig.SAVE_INTERVAL, 'train', save_dir)

            val_eval = DirectWSSEvaluator(model, val_dataset)
            if LEAN:
                (_, _, val_r2, val_r2_mag, val_mae, val_mape, val_r2_spatial,
                 val_sign_acc, val_pos_acc, val_neg_recall, val_neg_precision, val_neg_f1,
                 val_tp, val_tn, val_fp, val_fn) = val_eval.evaluate()
            else:
                (val_r2, val_r2_mag, val_mae, val_mape, val_r2_spatial,
                 val_sign_acc, val_pos_acc, val_neg_recall, val_neg_precision, val_neg_f1,
                 val_tp, val_tn, val_fp, val_fn) = val_eval.plot_predictions(
                    epoch, TrainConfig.SAVE_INTERVAL, 'val', save_dir)
            # 将验证指标合并到同epoch训练行，不新增行
            trainer.metrics_logger.update_val_metrics(
                epoch, val_r2, val_r2_mag, val_mae, val_mape,
                r2_spatial_val=val_r2_spatial,
                sign_acc_val=val_sign_acc, pos_acc_val=val_pos_acc,
                neg_recall_val=val_neg_recall, neg_precision_val=val_neg_precision,
                neg_f1_val=val_neg_f1,
                tp_val=val_tp, tn_val=val_tn, fp_val=val_fp, fn_val=val_fn)
            # ====== 新增：在验证集上绘制比例图 ======
            if epoch == 100 and not LEAN:                
                # 为验证集创建临时trainer来绘制比例图
                val_trainer = DirectWSSTrainer(model, val_dataset, save_dir)
                val_trainer._plot_velocity_constraint_violation(
                    model, val_dataset, save_dir,
                    epoch + TrainConfig.SAVE_INTERVAL, force_plot=True, name="val"
                )
                # 清理临时trainer
                del val_trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # ==================== 阶段性特征分析 ====================
            # 每1000轮或最后一轮进行分析
            if not LEAN and ((epoch + TrainConfig.SAVE_INTERVAL) % 1000 == 0 or
                             (epoch + TrainConfig.SAVE_INTERVAL) >= TrainConfig.TOTAL_EPOCHS):
                
                trainer.plot_loss_curves()  # 保存到默认路径

                print(f"\n{'='*70}")
                print(f"【Epoch {epoch + TrainConfig.SAVE_INTERVAL} 特征重要性分析】")
                print(f"{'='*70}")
                
                final_corr_df, final_comparison = analyze_features(
                    train_dataset,
                    model=model,
                    save_dir=save_dir,
                    stage=f"Epoch_{epoch + TrainConfig.SAVE_INTERVAL}"
                )

                # 区域化分析（训练完成后）
                print(f"\n{'='*70}")
                print("【区域化特征重要性分析】")
                print(f"{'='*70}")
            
            # 保存最新模型
            latest_path = f'{save_dir}/direct_wss_model_latest.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch
            }, latest_path)

            # 保存 best-epoch 模型（与 latest 并行；早停回载前的最优状态快照）
            if hasattr(early_stopping, 'best_state_dict') and \
               early_stopping.best_state_dict and \
               'model_state_dict' in early_stopping.best_state_dict:
                best_path = f'{save_dir}/direct_wss_model_best.pth'
                torch.save({
                    'model_state_dict':
                        early_stopping.best_state_dict['model_state_dict'],
                    'epoch': early_stopping.best_epoch
                }, best_path)
            
            # 清空缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 早停检查
            if early_stopping.check_early_stop(epoch, val_r2, model):
                model.load_state_dict(early_stopping.best_state_dict['model_state_dict'])

                # 记录早停信息到 CSV
                trainer.metrics_logger.log(
                    epoch=epoch,
                    best_val_r2=early_stopping.best_score,
                    best_epoch=early_stopping.best_epoch,
                    early_stopped=True,
                    notes="early_stopped"
                )

                # 早停后最终分析
                if not LEAN:
                    print(f"\n{'='*70}")
                    print("【早停后最终特征分析】")
                    print(f"{'='*70}")

                    final_corr_df, final_comparison = analyze_features(
                        train_dataset,
                        model=model,
                        save_dir=save_dir,
                        stage="训练完成"
                    )
                break
        
        # 保存最终分析结果
        if final_corr_df is not None:
            final_corr_df.to_csv(f'{save_dir}/final_feature_correlations.csv', 
                                index=False)
        if final_comparison is not None:
            final_comparison.to_csv(f'{save_dir}/final_feature_importance_comparison.csv',
                                    index=False)
        
        success_files.append({
            'id': number_id+number_id_begin,
            'filename': filename,
            'path': save_dir
        })
            
        
    
    # 最终报告
    print(f"\n{'='*70}")
    print("【处理完成统计】")
    print(f"总计: {len(csv_files)} | 成功: {len(success_files)} | 失败: {len(failed_files)}")

    # 顺序训练全部 case 结束后的最终权重（zero-shot 验证的主 checkpoint；
    # 早停时 per-case latest.pth 是最后 interval 权重，与协议最终权重有差异）
    if model is not None:
        session_final = os.path.join(cfg.SAVE_DIR_ORIGINAL,
                                     'direct_wss_model_session_final.pth')
        torch.save({'model_state_dict': model.state_dict(),
                    'epoch': 'session_final'}, session_final)
        print(f"✓ session-final checkpoint 已保存: {session_final}")
    
    if failed_files:
        print("\n失败文件:")
        for f in failed_files:
            print(f"  [{f['id']}] {f['filename']}: {f['error']}")


if __name__ == "__main__":
    main()