# Main and ablation experiments

## Main experiments

| ID | Experiment | Question addressed |
| --- | --- | --- |
| M01 | Random Forest | Performance of handcrafted composition/structure feature baselines |
| M02 | XGBoost | Whether a strong tree baseline can replace crystal graph models |
| M03 | CGCNN-style | Differences between conventional crystal graph convolution and ALIGNN |
| M04 | Single-task ALIGNN | Whether multitask learning improves on single-task learning |
| M05 | Equal-weight MTL ALIGNN | Whether standard multitask modeling is effective |
| M06 | Full MTL ALIGNN, 5 seeds | Classification performance of the full model and Deep Ensemble members |
| M07 | Single/MC Dropout/Deep Ensemble | Uncertainty, calibration, error detection, and selective prediction |
| M08 | Direct/Product/Minimum/Learned fusion | Whether a separate Joint head is necessary |
| M09 | Probability and risk-aware ranking | Whether risk-aware ranking improves screening yield at a fixed budget |
| M10 | Unknown Top-100 | Final candidate outputs and Materials Studio-compatible CIF exports |

## Ablation experiments

All ablations share the same split, training hyperparameters, and seeds. In principle, only one factor changes.

| ID | Single change relative to the full model |
| --- | --- |
| A01 | Remove the Dynamic auxiliary task; retain Joint+Thermo |
| A02 | Remove the Thermo auxiliary task; retain Joint+Dynamic |
| A03 | Remove the logical-consistency loss |
| A04 | Replace adaptive auxiliary weights with fixed weights |
| A05 | In a control without logical constraints, shuffle only the training Thermo and Dynamic auxiliary labels |
| A06 | Include Joint in task-level adaptive weighting |
| A07 | Train no new model; calculate task-gradient cosine similarities in the shared encoder |

## Outside the default workflow

`RUN_OOD_OPTIONAL.ps1` generates chemical-system, structure-family, layer-group, and element holdout definitions. The default `RUN_ALL.ps1` does not start OOD training.
