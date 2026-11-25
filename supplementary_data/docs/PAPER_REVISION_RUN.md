# Running the manuscript revision experiments

This document describes the task configuration of the original project. In the compact package, code is under `code/` and graph caches are omitted. Check historical paths and the root README before executing the original tasks.

## Task to run

Open the complete `D:\ck\two` folder in VS Code, press **Ctrl+Shift+P**, select **Tasks: Run Task**, then select:

**`60 - Paper revision - pilot 18 runs (resume)`**

This task checks the environment, fixed splits, graph caches, and existing baselines; performs 18 training runs sequentially; and then calibrates, evaluates, and summarizes automatically. No separate data-preparation task is needed in the original project. If the task list has not refreshed, run `Developer: Reload Window`.

The task terminal shows the current experiment, seed, and epoch. After an interruption, rerun task 60: completed members are skipped, and incomplete members resume from the last complete epoch checkpoint. Work from an interrupted epoch is recomputed.

## The 18 additional training runs

Use the existing composition-disjoint and structure-family-disjoint splits, with seeds 2025, 2026, and 2027 for each.

| Control | Training objective | Runs per split | Purpose |
| --- | --- | ---: | --- |
| Equal+LC | T, D, and J BCE terms with all weights exactly 1 and logical coefficient 0.10 | 3 | Form weighting-strategy by logical-constraint controls with Equal, Full, and Full-noLC |
| Four-state | One softmax head for states 00, 01, 10, and 11, using four-class cross-entropy only | 3 | Test an explicit joint distribution against soft logical constraints |
| T+D-product | Shared ALIGNN encoder with T and D BCE terms only; each member has J=pT*pD | 3 | Test whether an independent joint-prediction head is necessary |

Total: **3 methods x 2 splits x 3 seeds = 18 runs**. Match the existing strict-split baselines in encoder, learning rate, batch size, maximum 200 epochs, and early-stopping patience=30. The structure-family split uses the existing project identifiers. Its strictness depends on that grouping definition; it is not a comprehensive crystal-structure-similarity deduplication.

All three models select checkpoints by validation Joint AP. Therefore, describe T+D-product as having no joint-task training loss, not as avoiding joint labels throughout development. Fix the product rule in advance; do not select among product/min/other fusion rules using the test set.

The four-state model fits one shared temperature to its native four-class logits before deriving T, D, and J. T+D-product calibrates T and D separately, forms each member's product, and then averages. Both retain joint-probability bounds after calibration. Equal+LC and historical three-head baselines calibrate each head using the original method and report resulting logical violations without post-hoc clipping.

## Reusing existing results

The runner reads existing R25--R28 (composition) and R33--R36 (structure) results, selecting only the same seeds as this comparison. Historical result directories remain unchanged; new comparison tables go to `results\paper_revision`.

Preflight checks validation/test material IDs, labels, and training configurations. New configurations also freeze SHA256 hashes of the supervised-data and split CSVs. Regenerated or changed inputs stop execution with an error. New training records code, runtime environment, and graph-cache file state to prevent silent mixing of versions. Cache checks cover existence, size, and modification time, not binary hashes of every graph.

If GPU memory is insufficient, use the original batch-size fallback and record the actual batch size. If fallback occurs, assess its difference from historical baselines in the formal comparison.

## Where to inspect results

Start with:

`D:\ck\two\results\paper_revision\summary_3seeds\seed_mean_std.csv`

Then inspect these files in the same directory:

- `ensemble_comparison.csv`: validation/test and raw/calibrated AP, MCC, NLL, Brier, ECE, plus risk metrics for several uncertainty scores.
- `member_comparison.csv`: per-seed results, including actual batch size and best epoch.
- `paired_seed_differences.csv`: Joint-metric differences from Full for matching splits and seeds.
- Each experiment's `calibration_and_logic.json`: temperatures, validation-selected thresholds, and member/ensemble logical-violation rates.
- Each experiment's `test_reliability.csv` and `test_risk_coverage.csv`: data for manuscript calibration and risk-coverage plots. Risk-curve files retain at most 201 plotting points; AURC still uses coverage levels from all samples.

AP uses `average_precision_score`; between-seed standard deviation uses `ddof=1`. Match member counts and seeds across methods. Three-seed differences and means/standard deviations do not substitute for adequately powered significance evidence. Do not select the best test seed.

Run status is in `D:\ck\two\results\paper_revision\_runner\status.json`; logs and `preflight.json` are in the same directory in the original project.

## Other tasks

| Task | When to use |
| --- | --- |
| `59 - Paper revision - preflight only` | Check without training; task 60 already includes this step |
| `61 - Paper revision - extend to 30 runs (5 seeds)` | Add 2028 and 2029; if the 18 runs are complete, perform only 12 additional runs and produce `summary_5seeds` |
| `62 - Paper revision - summarize 3 seeds` | Independently regenerate the three-seed summary after all 18 runs finish |
| `63 - Paper revision - summarize 5 seeds` | Independently regenerate the five-seed summary after all 30 runs finish |

Use five seeds consistently for formal submission when the prespecified computational budget permits. Decide whether to extend to five seeds independently of whether the three-seed test results appear favorable.

The original task can also be run from a VS Code PowerShell terminal:

```powershell
Set-Location -LiteralPath 'D:\ck\two'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_paper_revision.ps1 -Mode pilot
```

## Relationship to external validation using published DFT

These 18/30 runs supplement model evidence in the first two manuscript sections through internal retrospective experiments. **They do not establish completion of the external DFT validation in the third section.**

Published-DFT material lists, structures, and evidence require separate curation and structure/composition deduplication against training and validation data. Dynamical evidence alone does not supply complete external T, D, and J labels. Once the external set is frozen, excluding previously trained-on materials may require retraining matched models and baselines; such training is not included in the 18/30 runs above.

No new DFT calculation was started for this revision setup, and unverified literature materials were not automatically labeled as external ground truth.
