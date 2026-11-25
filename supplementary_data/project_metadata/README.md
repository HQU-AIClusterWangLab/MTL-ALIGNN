# two: joint stability prediction and reliable screening of two-dimensional materials

This is the translated historical README from the original `D:\ck\two` project. It describes that project's original layout and task configuration. For the organized compact package, use the current root README; historical source paths below identify the original files.

The original project uses `D:\ck\two\data\c2db.db` to perform:

- Thermo, Dynamic, and Joint stability prediction.
- Random Forest, XGBoost, CGCNN-style, and ALIGNN baselines.
- Multitask learning with Joint as the main task.
- Deep Ensemble and MC Dropout uncertainty evaluation.
- Calibration, error detection, risk-coverage analysis, and OOD stress tests.
- Probability and risk-aware ranking.
- Top-100 screening and CIF export for materials with `dyn_stab=Unknown`.

## Fixed original environment

- Python executable: `D:\ck\anaconda_env\envs\two\python.exe`
- Python 3.10.20
- PyTorch 2.2.0+cu121
- DGL 2.2.1+cu121
- ALIGNN 2026.4.2
- ASE 3.28.0
- GPU: NVIDIA GeForce RTX 4090 D, 23.99 GB
- Database: 16,905 records
- Fields: `uid`, `ehull`, `dyn_stab`
- Dynamical labels: Yes 4,791; No 3,311; Unknown 8,803

The project does not install or upgrade packages automatically, preserving the validated PyTorch/DGL/CUDA combination.

## Original installation layout

1. Back up existing code and results in `D:\ck\two`.
2. Retain the existing database at `D:\ck\two\data\c2db.db`.
3. Extract the original archive into `D:\ck`, producing `D:\ck\two`.

The original distribution archive did not contain `c2db.db` and did not replace the database. The current compact package does retain a database copy; see its root README.

## Label definitions

- Thermo: `ehull <= 0.10 eV/atom`.
- Dynamic: `dyn_stab == "Yes"`.
- Joint: both Thermo and Dynamic are 1.
- Supervised set: `dyn_stab` is Yes or No.
- Unknown pool: `dyn_stab == "Unknown"`.

By default, `ehull` is used only for labels and statistics and is excluded from main-model inputs. Final Unknown screening directly predicts all 8,803 materials and ranks them by calibrated `pJ`, frozen using validation data. The 420-material `ehull <= 0.10 eV/atom` subset is retained only for post-hoc diagnostics. The review D+E_hull deployment baseline is a separately identified Dynamic-only control.

## Original VS Code workflow

Open `D:\ck\two`, then select `Ctrl+Shift+P -> Tasks: Run Task`.

Run the initial checks in order:

```text
00 - Environment, data and model smoke test
01 - Run unit tests
```

Then run stages as needed:

```text
10 - Prepare all data and supervised graphs
20 - Run all main experiments
30 - Run all ablation experiments
40 - Screen Unknown pool and export Top-100
50 - Build review supplementary statistics
51 - Run review XGBoost deployment baselines
52 - Run complete review GNN matrix (long)
```

Alternatively, use `90 - Run all default experiments sequentially`. The default sequential workflow does not run OOD experiments. Start those separately with `95 - Run optional brute-force OOD experiments`.

## Original PowerShell workflow

```powershell
Set-ExecutionPolicy -Scope Process Bypass
cd D:\ck\two
.\RUN_SMOKE.ps1
.\RUN_DATA.ps1
.\RUN_MAIN.ps1
.\RUN_ABLATIONS.ps1
.\RUN_UNKNOWN.ps1
.\RUN_REVIEW_SUPPLEMENTS.ps1
.\RUN_REVIEW_BASELINES.ps1 -SkipGnn
```

For sequential execution, run `.\RUN_ALL.ps1`. After interruption, rerun the workflow. Neural-network training resumes from `last_checkpoint.pt`, while `COMPLETED.json` marks completed tasks for skipping.

## Main and ablation experiments

See the historical `docs\01_experiment_map.md`, preserved under `supplementary_data/docs/` in the compact package.

Main experiments:

- M01 Random Forest
- M02 XGBoost
- M03 CGCNN-style
- M04 Single-task ALIGNN
- M05 Equal-weight MTL ALIGNN
- M06 Full MTL ALIGNN, five-member Deep Ensemble
- M07 Uncertainty and calibration
- M08 Joint-fusion comparison
- M09 Retrospective ranking evaluation
- M10 Unknown screening

Ablations:

- A01 Joint+Thermo
- A02 Joint+Dynamic
- A03 No logical consistency
- A04 Fixed task weights
- A05 Shuffled auxiliary labels
- A06 Adaptive weighting for all tasks
- A07 Gradient-conflict analysis

## Original outputs

Main result directories were `D:\ck\two\results\main`, `results\ablation`, `results\paper`, `results\review`, and `results\review_supplements`.

Unknown Top-100 lists and Materials Studio-compatible CIF files were written to `D:\ck\two\results\main\M10_unknown_screening`, including:

- `probability_top100.xlsx`
- `risk_top100.xlsx`
- `unknown_screening_top100.xlsx`
- `probability_top100_cif\`
- `risk_top100_cif\`

Review summaries are in `results\review\summary.xlsx` (D-only and D+E_hull) and `results\review_supplements\summary.xlsx`. The latter's `figures\` directory provides 300-dpi PNG and vector PDF outputs. The fail-closed DFT/phonon input preparation directory for Ir3BrI2Te3 was `dft\ir3bri2te3_phonon\`.

Final DFT candidates are selected manually in Materials Studio using Top-100 ranks, chemical diversity, and structural diversity.

## Methodological rules

1. Test data do not select thresholds, temperatures, uncertainty scores, or lambda.
2. The Unknown pool is excluded from training, validation, calibration, and model selection.
3. Review splits use canonical compositions independent of element order and supercell multiplicity, with frozen random, composition-disjoint, chemical-system-disjoint, and structure-family-disjoint manifests. The old `iid_split.csv` is not a strictly composition-disjoint split.
4. Exclude all exact structure duplicates with conflicting known labels.
5. Deep Ensemble temperature scaling is applied to each member's logits before probability averaging.
6. Refer to `p_T*p_D` only as a product-fusion score.
7. Uncertainty standard deviation represents model-prediction disagreement only.
8. Refer to `p_J-lambda*u` only as a risk-aware ranking score.
9. Evaluate ranking gains using Precision@K, Yield@K, and paired bootstrap analysis.

See `docs\02_methodological_guards.md` in the original layout for details.

## Optional pretrained baseline

No pretrained weights were supplied, so the default workflow trains from scratch. Optional settings are provided in `train\experiments\optional\O03_pretrained_alignn.yaml` and `pretrained\README.md`. Add compatible weights and run that experiment separately; it is outside the default workflow.

## Review experiments and practical limits

See `docs\review_matrix.md` for strict splits and the complete GNN matrix. The default matrix contains 120 ALIGNN seed runs and requires substantial GPU time. The runner supports deployment/full-matrix batches, checkpoint resumption, and skipping completed runs. Ir3BrI2Te3 phonon/soft-mode experiments also require an external DFT engine, pseudopotentials, and phonon tools. Missing inputs do not produce fabricated calculation results.
