# Project validation report

Validation performed before packaging:

- Python syntax compilation: passed for every `.py` file.
- YAML parsing: passed for all experiment and project configurations.
- JSON parsing: passed for all VSCode configuration files.
- Unit tests: 6/6 passed.
  - multi-task logical consistency loss;
  - classification and selective-prediction metrics;
  - ensemble uncertainty aggregation.
- Synthetic grouped IID split: passed, with no reduced-formula overlap.
- Synthetic chemical-system OOD split: passed, with zero test/development group overlap.
- VSCode task references: all referenced project files exist.

Not executable in the packaging environment:

- Windows PowerShell execution;
- CUDA/DGL/ALIGNN graph construction on the user's RTX 4090 D;
- full `c2db.db` graph generation and model training.

Those machine-specific checks are performed by `RUN_SMOKE.ps1`, which must pass before long training runs.
