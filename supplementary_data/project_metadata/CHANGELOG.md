# Rebuild notes

- Fixed interpreter to the validated `two` environment.
- Fixed C2DB fields to `uid`, `ehull`, and `dyn_stab`.
- Added conservative duplicate resolution and label-conflict exclusion.
- Added formula-group leakage control and optional OOD split generation.
- Adapted the ALIGNN 2026.4.2 backbone to expose pooled crystal embeddings.
- Added main experiments, core ablations, five-member Deep Ensemble, MC Dropout, calibration, selective prediction, Joint fusion, ranking and Unknown screening.
- Added common-temperature member-logit calibration for Deep Ensemble.
- Added validation-only uncertainty and lambda selection.
- Added paired bootstrap comparison of risk-aware and probability ranking.
- Added checkpoint resume, GPU OOM batch fallback, VSCode tasks, PowerShell pipelines, unit tests, paper tables, figures, Excel files and CIF export.
