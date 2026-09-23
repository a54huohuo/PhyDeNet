# PhyDeNet — physics-guided decoupled magnitude–sign WSS prediction

Official code for the Nature Machine Intelligence submission
**"Physics-guided decoupled magnitude–sign neural network for wall shear
stress prediction on unstructured point clouds"**.

PhyDeNet predicts signed wall shear stress (WSS) directly from an
unstructured point cloud of **scalar** velocity and pressure measurements —
no mesh, no centreline extraction, no vector velocity field. A magnitude
branch aggregates scalar velocity and pressure from the three nearest
interior points of each wall location; a sign branch classifies flow
direction from twelve physics-derived geometric features behind a stop
gradient; a spatial-gradient consistency loss regularises the predicted WSS
field.

Headline results (paper): signed R² = 0.784, magnitude R² = 0.838,
MAE = 2.96 Pa, 62% recirculation recall on 131 patient-specific carotid
bifurcation cases; zero-shot recirculation recall 0.71–0.99 on unseen
ideal-geometry benchmarks; leave-one-patient-out positive R² on 90% of
held-out cases.

## Installation

```bash
git clone <repository-url>
cd PhyDeNet
pip install -r requirements.txt
```

PyTorch with CUDA is recommended (training uses GPU when available; a
~600k-parameter model trains in minutes per case on one consumer GPU).
Outputs are written to `./runs/` (override with the `PHYDENET_RUNS`
environment variable).

## Data

Training data comes from the public
**Carotid Bifurcation Haemodynamics Dataset** (Eulzer et al.), Zenodo
[DOI 10.5281/zenodo.10695923](https://zenodo.org/records/10695923),
CC BY 4.0. Download and unpack `carotid_flow_database.zip` (the VTU flow
fields), then convert every `{case}_wss.vtu` + `{case}_fluid.vtu` pair into
the per-case point-cloud CSVs used for training:

```bash
python preprocess/vtu_to_combined.py --src-dir /path/to/carotid_flow_database \
                                     --out-dir data/
```

Each output `data/{case}_combined.csv` has columns
`x, y, z, velocity, pressure, wss` (mm, m/s, Pa, Pa); wall points carry the
WSS label, interior points carry `wss = 0`.

## Training

Everything runs through one entry point:

```bash
python main.py --experiment <name> --data-dir data/
```

The model trains **sequentially across cases** (curriculum): each case starts
from the previous case's weights, matching the paper's protocol. Key
experiments and where they appear in the paper:

| `--experiment` | description | paper role |
|---|---|---|
| `no_geom` | **PhyDeNet-S** (18-dim input, sign branch, physics loss) | main model |
| `no_sign_no_geom` | **PhyDeNet-M** (magnitude-only operating point) | main model |
| `with_geom` | + 4 wall-topography features | ablation |
| `no_sep_geom` | − 12 separation-geometry features | ablation |
| `no_physics` | − spatial smoothness loss | ablation |
| `flow_only` | wall-point scalars only (2-dim) | ablation |
| `coupled_head` | gradient-coupled sign head | architecture baseline |
| `no_geometry_mlp` | bridge-sign MLP | architecture baseline |
| `rf` / `xgboost` | tree baselines on the same features | comparison |
| `pointnet`, `pointnet_combined`, `pointnetpp_*`, `dgcnn_*`, `gat_*` | point-cloud / GNN baselines | comparison |
| `no_geom_lopo01` … `no_geom_lopo73` | leave-one-patient-out folds | cross-patient evaluation |

Useful environment variables:

* `LOPO_TOTAL_EPOCHS` — override per-case epochs (default 3000; note training
  starts at epoch 100, so values ≤ 100 run no epochs)
* `LOPO_LEAN=1` — skip plots/analysis I/O for faster bulk runs

Per-case outputs land in `./runs/<experiment_name>/<case>/`
(`direct_wss_model_best.pth`, `training_metrics.csv`, prediction figures).
Sign metrics inside `training_metrics.csv` use the training-time ramped
thresholds (0.50 → 0.35/0.60); the paper's tables use the fixed-threshold
canonical convention — reproduce those with `tools/predict_and_eval.py`.

**Leave-one-patient-out protocol.** Cases are grouped by patient from the
file name (`case_<patient>_NNN_<side>_combined.csv`, e.g. `case_k_003_left`
and `case_k_003_right` belong to patient `k_003`). Each fold trains on all
cases of the other 72 patients (point `--data-dir` at a folder containing
exactly those CSVs) and evaluates zero-shot on the held-out patient's
case(s).

**Ideal-geometry zero-shot benchmark.** The six COMSOL benchmark cases
(straight/bend/bifurcation/taper/enlarged-bulb bifurcation/stenosis) and the
LOGO protocol are provided as a separate reproduction package (model sources,
mesh settings, analytic verification) in the paper's supplementary material;
run `python main.py -e no_geom_logo1 ... no_geom_logo6` on the converted
benchmark CSVs to reproduce the zero-shot numbers.

## Predict & evaluate

Score a trained checkpoint on one case with the paper's canonical metrics
(R², MAE, sign metrics with the n.a.-below-50-negatives convention, and the
k=6 spatial-gradient R²):

```bash
python tools/predict_and_eval.py \
    --ckpt runs/ablation_no_geometry/<case>/direct_wss_model_best.pth \
    --experiment no_geom \
    --csv data/case_k_003_left_combined.csv \
    --out-dir results
```

Writes `{case}_predictions.csv` (x, y, z, wss_true, wss_pred in Pa) and
`{case}_metrics.json`.

## Repository structure

```
PhyDeNet/
├── main.py                  training entry point
├── config.py                experiment registry, device, hyperparameters
├── dataset.py               point-cloud dataset + physics feature engineering
├── data_splitter.py         adaptive z-axis train/val split
├── models.py                model factory (PhyDeNet + baselines)
├── losses.py                WSS loss, GradientConstraint, coupled loss
├── trainer.py               training loop, schedulers, logging
├── evaluator.py             metrics (R², sign metrics, R²_spatial)
├── utils.py                 checkpointing, early stopping, logging
├── feature_analyzer.py      feature correlation / importance analysis
├── baseline_ml_trainer.py   Random Forest / XGBoost baselines
├── pointnet_model.py, pointnetpp_model.py,
│   dgcnn_model.py, gat_model.py     comparison architectures
├── preprocess/vtu_to_combined.py    Zenodo VTU -> training CSVs
└── tools/predict_and_eval.py        checkpoint inference + canonical metrics
```

## Citation

If you use this code, please cite the paper and the underlying dataset:

```bibtex
@article{phydenet2026,
  title   = {Physics-guided decoupled magnitude--sign neural network for wall
             shear stress prediction on unstructured point clouds},
  author  = {...},
  journal = {Nature Machine Intelligence},
  year    = {2026}
}
@dataset{eulzer2024carotid,
  title   = {A Dataset of Reconstructed Carotid Bifurcation Lumen and Plaque
             Models with Centerline Tree and Simulated Hemodynamics},
  author  = {Eulzer, Pepe and Richter, Kevin and Probst, Tristan and
             Hundertmark, Anna and Lawonn, Kai},
  year    = {2024},
  doi     = {10.5281/zenodo.10695923}
}
```

## License

Code: MIT (see [LICENSE](LICENSE)). Training data: third-party, CC BY 4.0 —
see the dataset record above for attribution requirements.
