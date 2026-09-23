"""predict_and_eval.py — run a trained checkpoint on one case and score it.

Loads a checkpoint produced by main.py, runs a full-wall forward pass on one
{case}_combined.csv, and reports the paper's canonical metrics:

  * regression: R^2, MAE, MAPE on wall points (physical Pa units)
  * sign classification (canonical convention): a point is negative iff its
    TRUE WSS is strictly below zero (zero-WSS points excluded); predicted
    negative iff the sign-head probability of the negative class > 0.5;
    NegRec = TN/(TN+FN), reported as null when the case has fewer than 50
    true-negative points (too few to score meaningfully)
  * spatial-gradient R^2: fidelity of the predicted WSS gradient field over
    the k=6 nearest-neighbour edge set (identical to the training-time
    GradientConstraint target)

Outputs (in --out-dir):
  {case}_predictions.csv   x, y, z, wss_true, wss_pred   (Pa)
  {case}_metrics.json      all metrics

Usage (from the repository root):
  python tools/predict_and_eval.py \
      --ckpt runs/ablation_no_geometry/.../direct_wss_model_best.pth \
      --experiment no_geom \
      --csv data/case_k_003_left_combined.csv \
      --out-dir results
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config as cfg                     # noqa: E402
from config import DEVICE                # noqa: E402
from dataset import CarotidDataset       # noqa: E402
from models import create_model          # noqa: E402

NA_NEG_THRESHOLD = 50     # canonical: NegRec is n.a. below this many true negatives


def r2_spatial(coords, wss_pred, wss_true, k=6):
    """R^2 between true and predicted edge gradients (k-NN graph)."""
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    tg, pg = [], []
    for i in range(len(coords)):
        for j in idx[i, 1:]:
            d = np.linalg.norm(coords[j] - coords[i])
            if d < 1e-8:
                continue
            tg.append((wss_true[j] - wss_true[i]) / d)
            pg.append((wss_pred[j] - wss_pred[i]) / d)
    if len(tg) < 2:
        return float('nan')
    return float(r2_score(np.array(tg), np.array(pg)))


def sign_metrics(wss_true, s_neg_prob):
    """Canonical sign metrics (negative = truth strictly < 0; zeros excluded)."""
    mask = wss_true != 0
    y_true_neg = wss_true[mask] < 0
    y_pred_neg = s_neg_prob[mask] > 0.5
    tn = int((y_true_neg & y_pred_neg).sum())
    fn = int((y_true_neg & ~y_pred_neg).sum())
    fp = int((~y_true_neg & y_pred_neg).sum())
    tp = int((~y_true_neg & ~y_pred_neg).sum())
    pos_acc = tp / (tp + fp) if (tp + fp) else float('nan')
    n_neg = tn + fn
    neg_recall = tn / n_neg if n_neg >= NA_NEG_THRESHOLD else None
    neg_precision = tn / (tn + fp) if (tn + fp) >= NA_NEG_THRESHOLD else None
    sign_acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(sign_acc=sign_acc, pos_acc=pos_acc,
                neg_recall=neg_recall, neg_precision=neg_precision,
                n_true_neg=n_neg, tn=tn, fp=fp, fn=fn, tp=tp)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', required=True, help='checkpoint .pth from main.py')
    parser.add_argument('--experiment', required=True, choices=list(cfg._EXP.keys()),
                        help='experiment name the checkpoint was trained with')
    parser.add_argument('--csv', required=True, help='one {case}_combined.csv')
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()

    cfg.EXPERIMENT = args.experiment
    nc = dict(cfg._EXP[args.experiment])
    cfg.EXP_CONFIG.clear()
    cfg.EXP_CONFIG.update(nc)

    ds = CarotidDataset(args.csv, split_mode='train', z_split_ratio=1.0,
                        split_strategy='z_ratio')
    t = ds.get_tensors()
    feat = t['wall_features'].to(DEVICE)
    model = create_model(feat.shape[1]).to(DEVICE)
    sd = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    sd = sd['model_state_dict'] if 'model_state_dict' in sd else sd
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in sd.items()})
    model.eval()

    with torch.no_grad():
        wss_pred, _, sign_prob = model.forward_coupled(
            feat, t['wall_velocity'].to(DEVICE), t['wall_pressure'].to(DEVICE))

    fc = ds.features_characteristics
    y = np.asarray(ds.wall_wss).ravel() * fc['wss']
    p = np.asarray(wss_pred.cpu().numpy()).ravel() * fc['wss']
    # sign head outputs P(positive); P(negative) is its complement
    s_neg = 1.0 - np.asarray(sign_prob.cpu().numpy()).ravel()
    coords = np.asarray(ds.wall_coords)

    m = dict(
        case=os.path.basename(args.csv).replace('_combined.csv', ''),
        experiment=args.experiment,
        n_wall=int(len(y)),
        r2=float(r2_score(y, p)),
        mae=float(np.abs(y - p).mean()),
        r2_spatial=r2_spatial(coords, p, y),
    )
    m.update(sign_metrics(y, s_neg))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = m['case'] + ('' if m['experiment'] == 'no_geom'
                        else '_' + m['experiment'])
    import pandas as pd
    pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2],
                  'wss_true': y, 'wss_pred': p}).to_csv(
        os.path.join(args.out_dir, stem + '_predictions.csv'),
        index=False, float_format='%.4g')
    with open(os.path.join(args.out_dir, stem + '_metrics.json'), 'w') as f:
        json.dump(m, f, indent=2, default=lambda v: None if v is None else v)

    nr = 'n.a.' if m['neg_recall'] is None else f"{m['neg_recall']:.3f}"
    print(f"{m['case']}  n_wall={m['n_wall']}")
    print(f"  R2 = {m['r2']:.3f}   MAE = {m['mae']:.3f} Pa   "
          f"R2_spatial = {m['r2_spatial']:.3f}")
    print(f"  sign_acc = {m['sign_acc']:.3f}   pos_acc = {m['pos_acc']:.3f}   "
          f"NegRec = {nr}  (true negatives: {m['n_true_neg']})")
    print(f"  -> {os.path.join(args.out_dir, stem + '_metrics.json')}")


if __name__ == '__main__':
    main()
