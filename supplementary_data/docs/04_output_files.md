# Key outputs

## Data

- `data/processed/data_summary.json`
- `data/processed/supervised.csv`
- `data/processed/unknown_pool.csv`
- `data/processed/iid_split.csv`
- `data/processed/data_analysis.xlsx`
- `data/processed/duplicate_resolution.csv`

## Individual models

The original `results/<category>/<experiment>/seed_<seed>/` layout includes:

- `best_checkpoint.pt`
- `history.csv`
- `val_predictions.csv`
- `test_predictions.csv`
- `metrics.json`
- `thresholds.json`
- `resolved_experiment.yaml`
- `environment.json`

In this compact package, only the final five checkpoints are retained, under `models/experiments/main/M06_full_mtl/`. Existing result tables remain under `results/`.

## Full model

- `results/main/M06_full_mtl/ensemble/`
- `results/main/M06_full_mtl/mc_dropout/`
- `results/main/M06_full_mtl/uncertainty_evaluation/`
- `results/main/M06_full_mtl/joint_fusion/`
- `results/main/M06_full_mtl/ranking/`
- `results/main/M06_full_mtl/task_gradient_analysis/`

## Unknown screening

`results/main/M10_unknown_screening/` contains:

- `unknown_all_predictions.csv`
- `probability_top100.csv/.xlsx`
- `risk_top100.csv/.xlsx`
- `unknown_screening_top100.xlsx`
- `probability_top100_cif/`
- `risk_top100_cif/`

## Manuscript exports

`results/paper/` contains:

- `paper_tables.xlsx`
- Main-experiment and ablation summary CSV files
- PNG/PDF exports of PR curves, calibration plots, risk-coverage curves, and ranking plots
