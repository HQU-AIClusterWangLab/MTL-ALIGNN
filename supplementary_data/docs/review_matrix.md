# Review experiment matrix with strict splits

This matrix separates four data-splitting protocols and two distinct review questions. All review configurations use the five seeds `[2025, 2026, 2027, 2028, 2029]`. Models within a split share identical train/validation/test manifests, allowing paired comparisons by seed.

## Data splits

| Name | Manifest | Constraint |
| --- | --- | --- |
| random | `review_splits/random_iid_stratified.csv` | Stratified random IID control |
| composition | `review_splits/composition_disjoint.csv` | No canonical-composition overlap across subsets |
| chemical | `review_splits/chemical_system_disjoint.csv` | No element-set overlap across subsets |
| structure | `review_splits/structure_family_disjoint.csv` | No structure-family-key overlap across subsets |

The runner first executes `data/make_review_splits.py` to regenerate and audit these manifests. Do not treat results from the old `iid_split.csv` and the new strict manifests as paired experiments on the same split.

## Deployment baselines: R01--R08

Each split includes two models predicting `dynamic` only and using validation dynamic PR-AUC for early stopping:

- D-only: structure graph inputs.
- D+E_hull: structure graph inputs plus continuous `ehull`, standardized using training-only statistics.

These 8 configurations comprise 40 GNN seed runs. They test whether known/available E_hull improves dynamical-stability prediction at deployment. Do not directly mix their metrics with those of the joint-label models below.

## Full joint-stability matrix: R21--R36

| Split | Single (joint-only) | Equal MTL | Full MTL | Full-no-logic |
| --- | ---: | ---: | ---: | ---: |
| random | R21 | R22 | R23 | R24 |
| composition | R25 | R26 | R27 | R28 |
| chemical | R29 | R30 | R31 | R32 |
| structure | R33 | R34 | R35 | R36 |

Reuse the existing templates strictly: M04 for Single, M05 for Equal MTL, M06 for Full MTL, and A03 for Full-no-logic. Apart from experiment names, result categories, split manifests, and the common five seeds, do not change model size, optimizers, or training hyperparameters. Full and Full-no-logic differ only in logical regularization and support direct paired ablation. Single, Equal, and Full distinguish the contributions of multitask learning and adaptive auxiliary weighting.

The complete matrix contains 16 configurations and 80 GNN seed runs.

## Execution options

Historical commands from the original project root:

```powershell
# Default: XGBoost deployment and all GNNs (R01--R08 and R21--R36).
.\RUN_REVIEW_BASELINES.ps1

# Deployment GNNs only (R01--R08), skipping XGBoost.
.\RUN_REVIEW_BASELINES.ps1 -SkipTabular -GnnSet Deployment

# Full joint-stability GNN matrix only (R21--R36).
.\RUN_REVIEW_BASELINES.ps1 -SkipTabular -GnnSet FullMatrix

# XGBoost deployment baselines only.
.\RUN_REVIEW_BASELINES.ps1 -SkipGnn
```

The default workload contains 8 XGBoost configurations (40 seed runs) and 24 GNN configurations (120 seed runs), totaling 160 seed runs. `-Force` is forwarded to every training entry point. Without it, completed experiments follow the existing skip rules. Wall-clock time depends on graph caches, GPU hardware, and early stopping. Run `-GnnSet Deployment` or `-GnnSet FullMatrix` in separate batches when appropriate, and preserve independent outputs for every seed.

In the compact package, the historical PowerShell scripts are under `code/`. Check their absolute paths and regenerate omitted graph caches before running them; consult the root README for supported entry points.
