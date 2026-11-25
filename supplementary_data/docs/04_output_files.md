# 关键输出

## 数据

- `data/processed/data_summary.json`
- `data/processed/supervised.csv`
- `data/processed/unknown_pool.csv`
- `data/processed/iid_split.csv`
- `data/processed/data_analysis.xlsx`
- `data/processed/duplicate_resolution.csv`

## 单个模型

每个`results/<category>/<experiment>/seed_<seed>/`包含：

- `best_checkpoint.pt`
- `history.csv`
- `val_predictions.csv`
- `test_predictions.csv`
- `metrics.json`
- `thresholds.json`
- `resolved_experiment.yaml`
- `environment.json`

## 完整模型

- `results/main/M06_full_mtl/ensemble/`
- `results/main/M06_full_mtl/mc_dropout/`
- `results/main/M06_full_mtl/uncertainty_evaluation/`
- `results/main/M06_full_mtl/joint_fusion/`
- `results/main/M06_full_mtl/ranking/`
- `results/main/M06_full_mtl/task_gradient_analysis/`

## Unknown筛选

`results/main/M10_unknown_screening/`：

- `unknown_all_predictions.csv`
- `probability_top100.csv/.xlsx`
- `risk_top100.csv/.xlsx`
- `unknown_screening_top100.xlsx`
- `probability_top100_cif/`
- `risk_top100_cif/`

## 论文导出

`results/paper/`：

- `paper_tables.xlsx`
- 主实验和消融汇总CSV
- PR曲线、校准图、risk-coverage曲线、排序图的PNG/PDF
