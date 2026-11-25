# Core supplementary experiments on the fixed IID split

Status: Partially complete; missing training runs are listed below. Only the existing iid_split.csv is used; the data were not repartitioned.

## Questions addressed by the tables

- main_5seed_table.csv: mean±SD (ddof=1) of raw-probability individual-run metrics for four methods on matching seeds. Missing seeds are marked explicitly; incomplete results are not presented as complete five-seed conclusions. Ensemble metrics are reported separately in 01_main_5seed/ensemble_metrics.csv.
- joint_construction_table.csv: compare direct Joint prediction with Hard AND, Product, and Minimum fusion of independent T-only/D-only ensembles. AP/ROC-AUC are not calculated for Hard AND.
- paired_bootstrap_table.csv: AP of Full MTL minus each alternative, strictly paired by material ID, with 2000 bootstrap samples and 95% percentile CIs. The two-sided sign-tail probability is a descriptive bootstrap quantity without multiple-testing correction; it is not a stand-alone confirmatory significance test.
- beta_sensitivity_table.csv: relationships between beta and AP/MCC/logical-violation rates. Core results use raw probabilities; calibrated results are separate. Missing points are not interpolated, and beta is not selected using test data.
- screening_efficiency_table.csv: fraction of candidates requiring validation to recover 50%/80%/90% of true positives. Full curves, 10/20/30% enrichment, and material-bootstrap CIs are in 04_screening_efficiency. This measures candidate-count budgets, not actual DFT runtime savings.
- risk_increment_table.csv: whether task-decomposed risk adds information for error detection and selective risk. All methods are retained, without a predetermined ranking conclusion.

## Reused and new results

Reused sources are recorded for every seed in configuration_audit.csv and the manifest. Scientific configurations are checked before merging M04 and Q1 Joint results; omitted monitor_task=joint values are normalized to the code default. A03 has beta=0, so the logical branch is not executed and lower_bound flag differences do not change its loss. Beta configurations retain lower_bound=true and vary beta only. Existing bootstrap, temperature, ensemble, and risk-coverage implementations are reused.

Equal+LC and unfinished beta runs require new training; other completed, eligible models are skipped. Joint construction, screening, risk, and bootstrap analyses post-process existing frozen predictions. Actual new-training sources are under core_revision/training. Currently missing jobs:

- Equal + LC, seed=2025
- Equal + LC, seed=2026
- Equal + LC, seed=2027
- Equal + LC, seed=2028
- Equal + LC, seed=2029
- beta=0.01, seed=2025
- beta=0.01, seed=2026
- beta=0.01, seed=2027
- beta=0.05, seed=2025
- beta=0.05, seed=2026
- beta=0.05, seed=2027
- beta=0.20, seed=2025
- beta=0.20, seed=2026
- beta=0.20, seed=2027
- beta=0.50, seed=2025
- beta=0.50, seed=2026
- beta=0.50, seed=2027

## Leakage prevention and provenance

Temperatures, thresholds, risk normalization, and LogisticRegression are fitted on validation data only. Test data are used only for frozen evaluation and bootstrap analysis. The risk model fixes C=1, lbfgs, and random_state=20260917, without test-based tuning. Main ensemble and risk analyses consistently use calibrated probabilities; member summaries and core beta results use raw probabilities in separate tables. No successful-seed selection is applied.

Historical runs did not save training-time code/data content hashes, so byte-for-byte historical code identity cannot be established retrospectively. The complete resolved configuration, environment, actual batch size, split, materials, and labels have been checked in the current audit. New results preserve run_contract and current code/data fingerprints.

## Execution and result versions

VS Code task 04 performs preflight checks, 05 runs the complete CUDA queue, and 06 performs analysis only. Repeated calls skip completed analyses with identical inputs. When seeds are added, old scientific files are retained and new versions are written to revisions/<signature>/ under each analysis directory. Consult the analysis entry in _runner/status.json and this result_manifest.json for the latest paths. Training failures stop execution immediately and record failed_job while preserving the original failure log. An empty literature template exits normally without generating literature evidence.
