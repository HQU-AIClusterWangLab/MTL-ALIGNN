# Literature phonon-validation inclusion protocol

This interface processes genuine, manually verified evidence only. It does not generate DOIs, paper titles, or phonon conclusions automatically.

1. Candidates must come from the existing 8803 materials in `data/processed/unknown_pool.csv` and be linked by unique material_id. Define the candidate scope or model-based ranking before searching the literature; do not select successful literature examples first and then choose materials retrospectively. Before searching, record the selection rule, search date, terms, and databases in notes. Record unsuccessful searches separately as well.
2. Identical formulas do not establish identical structures. Prioritize matching space group, lattice parameters, atomic arrangement, layer count, phase/polytype, and calculation dimensionality. In structure_match_basis, record specific structural evidence, literature pages/figures/tables, and discrepancies.
3. Allowed structure_match_status values are matched, partial, uncertain, and excluded. Retain audit records for both matched and excluded cases; do not remove failures. Use uncertain for insufficient evidence, partial for incomplete correspondence, and excluded for clear incompatibility.
4. Allowed literature_dynamic_conclusion values are stable, unstable, and ambiguous. Check phonon calculation settings, sampling paths, imaginary-mode locations, and the authors' interpretation; do not infer stability from the title alone. Use ambiguous for uncertain conclusions and never label a case stable without phonon evidence.
5. Literature phonons directly validate Dynamic only. They do not automatically establish Joint-positive status in this study. Joint also requires Ehull <= 0.10 eV/atom for the same structure and applicable calculation conditions.
6. Enter DOI, title, and year manually from genuine sources. Provide at least one verifiable source in source_title/doi. Preserve structural evidence, dynamical evidence, calculation conditions, and inclusion/exclusion reasons in notes. Multiple sources for one material_id may occupy separate rows; do not preferentially discard conflicting conclusions.
7. Template formula/rank/probability fields may be blank. After joining genuine model outputs, the script preserves manual snapshots as manual_*; model columns come from existing Unknown screening outputs. model_rank sorts all 8803 candidates by descending calibrated pJ, breaking ties by material_id. u_dynamic/u_joint use existing raw-member sample standard deviations (ddof=1).
8. Dynamic agreement uses only matched records with stable/unstable conclusions. Exclude partial, uncertain, excluded, and ambiguous records from the agreement denominator while retaining all records. Report record-level and candidate-level results separately. Flag conflicting sources for the same candidate explicitly.
9. The default Dynamic decision threshold comes from the M06 five-member ensemble's validation calibration corresponding to the Unknown predictions. If predictions are replaced, also provide their matching frozen thresholds.json. Do not reselect thresholds using literature or test labels.

Historical command from the original project root:

```powershell
& 'D:\ck\anaconda_env\envs\two\python.exe' -B src/train/analyze_literature_validation.py
```

The compact package preserves that source under `code/src/train/`; check its environment and paths before running. If the template is empty, the script prints a message and exits normally without creating scientific results. After manual completion, results go to `results/core_revision/06_literature_validation/`. Changed inputs preserve previous versions and add `revisions/<signature>/`.
