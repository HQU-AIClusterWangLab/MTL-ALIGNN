# Methodological safeguards

1. The main Joint/MTL model does not use `ehull` as an input. Only the explicitly named review Dynamic+E_hull deployment baseline uses continuous `ehull`; describe it separately from the main model.
2. Data with `dyn_stab=Unknown` are completely isolated during model development, threshold selection, calibration, and ranking-parameter selection.
3. A canonical-composition audit found 399 composition groups crossing subsets in the old reduced-formula-based `iid_split.csv`. Treat it only as the legacy IID result. Review comparisons use frozen random, composition-disjoint, chemical-system-disjoint, and structure-family-disjoint manifests in `data/processed/review_splits`. Refer to `audit.json` and `manifest_sha256.json` for the audit results.
4. Audit exact structure duplicates first. Exclude all duplicates with conflicting known labels. When a known-label structure duplicates an Unknown structure, retain the known-label entry preferentially.
5. Early stopping monitors only the configured validation-task PR-AUC: Joint for Joint models and Dynamic for Dynamic-only deployment baselines.
6. Choose classification thresholds by MCC on validation data only.
7. Fit temperatures and classification thresholds only on the non-test validation set corresponding to the checkpoint or ensemble being evaluated. Freeze them before test evaluation. Do not transfer probability scales between different models or ensemble sizes.
8. Deep Ensemble calibration applies a shared validation-fitted temperature to each member's logits before averaging probabilities within that ensemble. Five-member Unknown inference must use the five-member temperatures and thresholds.
9. Choose uncertainty scores by validation AURC only.
10. Choose the risk coefficient lambda only on the validation Thermo-positive subset.
11. Use test data only for final comparisons, not rule selection.
12. `p_T*p_D` is a product-fusion score, not a universally valid joint probability.
13. Ensemble/MC standard deviation is a proxy for predictive disagreement, not a physical DFT error bar.
14. `p_J-lambda*u` is an empirical risk-ranking rule, not a confidence lower bound with guaranteed coverage.
15. As specified for this project, all 8,803 Unknown candidates receive calibrated `pJ` predictions and ranks directly. The `ehull<=0.10` subset is diagnostic only. The Top-K list prioritizes DFT calculations; it is not a set of verified stable materials.
