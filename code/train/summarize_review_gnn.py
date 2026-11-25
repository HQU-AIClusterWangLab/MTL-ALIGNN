"""Fail-closed five-seed summary for the strict review ALIGNN experiments.

The analysis has two families:

* deployment (R01--R08): Dynamic-only versus Dynamic+E_hull;
* architecture (R21--R36): Full versus Single, Equal, and Full-no-logic.

Every comparison uses exactly paired validation/test material IDs.  A single
threshold is selected on the mean validation probability of each five-member
ensemble and is then frozen for test evaluation.  Confidence intervals use a
paired material-level bootstrap of the mean test probabilities.

The command is intentionally fail-closed.  It always writes ``status.json``
and ``progress.csv`` first.  Statistical outputs are valid only when
``status.json`` says ``state=completed`` and ``summary_artifacts_valid=true``.
With incomplete seeds the default exit code is 2; ``--allow-pending`` is only
for non-blocking progress inspection and never creates statistical summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from train.metrics import best_mcc_threshold
from train.summarize_review_baselines import (
    METRICS,
    _bootstrap_metric_differences,
    classification_metrics,
    sha256,
    subset_mask,
)


DEFAULT_SEEDS = (2025, 2026, 2027, 2028, 2029)
SCOPES = ("deployment", "architecture", "all")
SUBSETS = ("all_test", "thermo_positive")
REQUIRED_ARTIFACTS = (
    "val_predictions.csv",
    "test_predictions.csv",
    "thresholds.json",
    "metrics.json",
    "resolved_experiment.yaml",
)


@dataclass(frozen=True)
class ExperimentSpec:
    family: str
    split: str
    variant: str
    experiment: str
    task: str


@dataclass(frozen=True)
class PairSpec:
    family: str
    pair: str
    reference_variant: str
    candidate_variant: str
    task: str


DEPLOYMENT_EXPERIMENTS = {
    "random": ("R01_dynamic_only_alignn_random", "R02_dynamic_ehull_alignn_random"),
    "composition": (
        "R03_dynamic_only_alignn_composition",
        "R04_dynamic_ehull_alignn_composition",
    ),
    "chemical_system": (
        "R05_dynamic_only_alignn_chemical",
        "R06_dynamic_ehull_alignn_chemical",
    ),
    "structure_family": (
        "R07_dynamic_only_alignn_structure",
        "R08_dynamic_ehull_alignn_structure",
    ),
}

ARCHITECTURE_EXPERIMENTS = {
    "random": (
        "R21_single_joint_alignn_random",
        "R22_equal_mtl_alignn_random",
        "R23_full_mtl_alignn_random",
        "R24_full_no_logic_alignn_random",
    ),
    "composition": (
        "R25_single_joint_alignn_composition",
        "R26_equal_mtl_alignn_composition",
        "R27_full_mtl_alignn_composition",
        "R28_full_no_logic_alignn_composition",
    ),
    "chemical_system": (
        "R29_single_joint_alignn_chemical",
        "R30_equal_mtl_alignn_chemical",
        "R31_full_mtl_alignn_chemical",
        "R32_full_no_logic_alignn_chemical",
    ),
    "structure_family": (
        "R33_single_joint_alignn_structure",
        "R34_equal_mtl_alignn_structure",
        "R35_full_mtl_alignn_structure",
        "R36_full_no_logic_alignn_structure",
    ),
}


def experiment_specs(scope: str) -> list[ExperimentSpec]:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    specs: list[ExperimentSpec] = []
    if scope in {"deployment", "all"}:
        for split, (dynamic_only, dynamic_ehull) in DEPLOYMENT_EXPERIMENTS.items():
            specs.extend(
                [
                    ExperimentSpec(
                        "deployment", split, "dynamic_only", dynamic_only, "dynamic"
                    ),
                    ExperimentSpec(
                        "deployment", split, "dynamic_ehull", dynamic_ehull, "dynamic"
                    ),
                ]
            )
    if scope in {"architecture", "all"}:
        variants = ("single", "equal", "full", "no_logic")
        for split, experiments in ARCHITECTURE_EXPERIMENTS.items():
            specs.extend(
                ExperimentSpec("architecture", split, variant, experiment, "joint")
                for variant, experiment in zip(variants, experiments)
            )
    return specs


def pair_specs(scope: str) -> list[PairSpec]:
    pairs: list[PairSpec] = []
    if scope in {"deployment", "all"}:
        pairs.append(
            PairSpec(
                "deployment",
                "dynamic_ehull_minus_dynamic_only",
                "dynamic_only",
                "dynamic_ehull",
                "dynamic",
            )
        )
    if scope in {"architecture", "all"}:
        pairs.extend(
            [
                PairSpec("architecture", "full_minus_single", "single", "full", "joint"),
                PairSpec("architecture", "full_minus_equal", "equal", "full", "joint"),
                PairSpec(
                    "architecture",
                    "full_minus_no_logic",
                    "no_logic",
                    "full",
                    "joint",
                ),
            ]
        )
    return pairs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def inspect_progress(
    review_dir: Path,
    scope: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in experiment_specs(scope):
        for seed in seeds:
            seed_dir = review_dir / spec.experiment / f"seed_{seed}"
            marker = seed_dir / "COMPLETED.json"
            row: dict[str, object] = {
                "family": spec.family,
                "split": spec.split,
                "variant": spec.variant,
                "experiment": spec.experiment,
                "task": spec.task,
                "seed": int(seed),
                "seed_dir": str(seed_dir.resolve()),
                "completion_marker": marker.exists(),
            }
            missing = [name for name in REQUIRED_ARTIFACTS if not (seed_dir / name).is_file()]
            row["missing_artifacts"] = ";".join(missing)
            marker_error = ""
            marker_completed = False
            if marker.exists():
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                    marker_completed = payload.get("status") == "completed"
                    if not marker_completed:
                        marker_error = "COMPLETED.json status is not completed"
                    if "seed" in payload and int(payload["seed"]) != int(seed):
                        marker_error = (
                            f"COMPLETED.json seed={payload['seed']} does not match {seed}"
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    marker_error = f"invalid COMPLETED.json: {error}"
            row["completion_marker_valid"] = marker_completed and not marker_error
            row["marker_error"] = marker_error
            if not marker.exists():
                row["state"] = "pending"
            elif marker_error or missing:
                row["state"] = "invalid"
            else:
                row["state"] = "complete"
            rows.append(row)

    progress = pd.DataFrame(rows)
    counts = progress["state"].value_counts().to_dict()
    state = (
        "invalid"
        if counts.get("invalid", 0)
        else "pending"
        if counts.get("pending", 0)
        else "completed"
    )
    status: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "state": state,
        "scope": scope,
        "summary_artifacts_valid": False,
        "required_seeds": [int(seed) for seed in seeds],
        "required_seed_count": len(seeds),
        "expected_experiments": len(experiment_specs(scope)),
        "expected_seed_runs": len(progress),
        "complete_seed_runs": int(counts.get("complete", 0)),
        "pending_seed_runs": int(counts.get("pending", 0)),
        "invalid_seed_runs": int(counts.get("invalid", 0)),
        "completion_rule": (
            "Every expected seed must have a valid COMPLETED.json plus all "
            f"required artifacts: {', '.join(REQUIRED_ARTIFACTS)}"
        ),
    }
    return progress, status


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")


def read_prediction(path: Path, task: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = (
        "material_id",
        "ehull",
        "y_thermo",
        "y_dynamic",
        "y_joint",
        f"p_{task}",
    )
    _require_columns(frame, required, path)
    frame = frame.copy()
    frame["material_id"] = frame["material_id"].astype(str)
    if frame["material_id"].duplicated().any():
        duplicates = frame.loc[
            frame["material_id"].duplicated(keep=False), "material_id"
        ].head(5)
        raise ValueError(f"Duplicate material_id values in {path}: {duplicates.tolist()}")
    numeric = ("ehull", "y_thermo", "y_dynamic", "y_joint", f"p_{task}")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(f"Non-finite {column} values in {path}")
    for column in ("y_thermo", "y_dynamic", "y_joint"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"Non-binary {column} values in {path}")
    if not np.array_equal(
        frame["y_joint"].to_numpy(int),
        frame["y_thermo"].to_numpy(int) * frame["y_dynamic"].to_numpy(int),
    ):
        raise ValueError(f"y_joint != y_thermo * y_dynamic in {path}")
    if not frame[f"p_{task}"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"p_{task} outside [0, 1] in {path}")
    return frame.sort_values("material_id", kind="mergesort").reset_index(drop=True)


def read_task_threshold(path: Path, task: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        threshold = float(payload[task])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {task} threshold in {path}") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{task} threshold outside [0, 1] in {path}: {threshold}")
    return threshold


def align_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    reference_name: str,
    candidate_name: str,
) -> None:
    if reference["material_id"].tolist() != candidate["material_id"].tolist():
        left_ids = set(reference["material_id"])
        right_ids = set(candidate["material_id"])
        raise ValueError(
            f"material_id mismatch for {reference_name} vs {candidate_name}; "
            f"left_only={sorted(left_ids - right_ids)[:5]}, "
            f"right_only={sorted(right_ids - left_ids)[:5]}"
        )
    for column in ("y_thermo", "y_dynamic", "y_joint"):
        if not np.array_equal(
            reference[column].to_numpy(int), candidate[column].to_numpy(int)
        ):
            raise ValueError(f"{column} mismatch for {reference_name} vs {candidate_name}")
    if not np.allclose(
        reference["ehull"].to_numpy(float),
        candidate["ehull"].to_numpy(float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"ehull mismatch for {reference_name} vs {candidate_name}")


def _load_inputs(
    review_dir: Path,
    scope: str,
    seeds: Sequence[int],
) -> tuple[
    dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame, float]],
    list[dict[str, object]],
]:
    loaded: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame, float]] = {}
    references: dict[tuple[str, str], tuple[pd.DataFrame, str]] = {}
    manifest: list[dict[str, object]] = []
    for spec in experiment_specs(scope):
        for seed in seeds:
            seed_dir = review_dir / spec.experiment / f"seed_{seed}"
            validation_path = seed_dir / "val_predictions.csv"
            test_path = seed_dir / "test_predictions.csv"
            threshold_path = seed_dir / "thresholds.json"
            validation = read_prediction(validation_path, spec.task)
            test = read_prediction(test_path, spec.task)
            threshold = read_task_threshold(threshold_path, spec.task)
            for partition, frame, path in (
                ("validation", validation, validation_path),
                ("test", test, test_path),
            ):
                key = (spec.split, partition)
                if key in references:
                    reference, reference_name = references[key]
                    align_frames(reference, frame, reference_name, str(path))
                else:
                    references[key] = (frame, str(path))
            loaded[(spec.experiment, int(seed))] = (test, validation, threshold)
            for name in ("COMPLETED.json", *REQUIRED_ARTIFACTS):
                path = seed_dir / name
                manifest.append(
                    {
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    return loaded, manifest


def _material_ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def summarize_complete(
    review_dir: Path,
    scope: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260809,
    ece_bins: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a non-empty unique sequence")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    loaded, manifest = _load_inputs(review_dir, scope, seeds)
    specs = experiment_specs(scope)

    per_seed_rows: list[dict[str, object]] = []
    for spec in specs:
        for seed in seeds:
            test, _validation, threshold = loaded[(spec.experiment, int(seed))]
            for subset in SUBSETS:
                mask = subset_mask(test, subset)
                labels = test.loc[mask, f"y_{spec.task}"].to_numpy(int)
                probability = test.loc[mask, f"p_{spec.task}"].to_numpy(float)
                per_seed_rows.append(
                    {
                        "family": spec.family,
                        "split": spec.split,
                        "subset": subset,
                        "variant": spec.variant,
                        "experiment": spec.experiment,
                        "task": spec.task,
                        "seed": int(seed),
                        "n_samples": int(mask.sum()),
                        "n_positive": int(labels.sum()),
                        "prevalence": float(labels.mean()),
                        "validation_threshold": float(threshold),
                        **classification_metrics(labels, probability, threshold, ece_bins),
                    }
                )
    per_seed = pd.DataFrame(per_seed_rows)

    summary_rows: list[dict[str, object]] = []
    group_columns = ["family", "split", "subset", "variant", "experiment", "task"]
    for keys, group in per_seed.groupby(group_columns, sort=False):
        row: dict[str, object] = dict(zip(group_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for column in (
            "n_samples",
            "n_positive",
            "prevalence",
            "validation_threshold",
            *METRICS,
        ):
            values = group[column].to_numpy(float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)
    experiment_summary = pd.DataFrame(summary_rows)

    spec_lookup = {(spec.family, spec.split, spec.variant): spec for spec in specs}
    ensemble: dict[tuple[str, str, str], dict[str, object]] = {}
    for spec in specs:
        test_reference, validation_reference, _ = loaded[(spec.experiment, int(seeds[0]))]
        validation_probability = np.mean(
            [
                loaded[(spec.experiment, int(seed))][1][f"p_{spec.task}"].to_numpy(float)
                for seed in seeds
            ],
            axis=0,
        )
        test_probability = np.mean(
            [
                loaded[(spec.experiment, int(seed))][0][f"p_{spec.task}"].to_numpy(float)
                for seed in seeds
            ],
            axis=0,
        )
        ensemble[(spec.family, spec.split, spec.variant)] = {
            "test": test_reference,
            "validation": validation_reference,
            "test_probability": test_probability,
            "validation_probability": validation_probability,
            "threshold": best_mcc_threshold(
                validation_reference[f"y_{spec.task}"].to_numpy(int),
                validation_probability,
            ),
        }

    paired_rows: list[dict[str, object]] = []
    pairing_rows: list[dict[str, object]] = []
    comparison_index = 0
    for split in DEPLOYMENT_EXPERIMENTS:
        for comparison in pair_specs(scope):
            reference_key = (comparison.family, split, comparison.reference_variant)
            candidate_key = (comparison.family, split, comparison.candidate_variant)
            if reference_key not in ensemble or candidate_key not in ensemble:
                continue
            reference_spec = spec_lookup[reference_key]
            candidate_spec = spec_lookup[candidate_key]
            reference = ensemble[reference_key]
            candidate = ensemble[candidate_key]
            validation_frame = reference["validation"]
            test_frame = reference["test"]
            for subset_index, subset in enumerate(SUBSETS):
                mask = subset_mask(test_frame, subset)
                labels = test_frame.loc[mask, f"y_{comparison.task}"].to_numpy(int)
                reference_probability = np.asarray(reference["test_probability"])[mask]
                candidate_probability = np.asarray(candidate["test_probability"])[mask]
                reference_threshold = float(reference["threshold"])
                candidate_threshold = float(candidate["threshold"])
                reference_metrics = classification_metrics(
                    labels, reference_probability, reference_threshold, ece_bins
                )
                candidate_metrics = classification_metrics(
                    labels, candidate_probability, candidate_threshold, ece_bins
                )
                bootstrap = _bootstrap_metric_differences(
                    labels,
                    reference_probability,
                    candidate_probability,
                    reference_threshold,
                    candidate_threshold,
                    bootstrap_replicates,
                    bootstrap_seed + comparison_index * 10 + subset_index,
                    ece_bins,
                )
                comparison_index += 1

                seed_group = per_seed.loc[
                    (per_seed["family"] == comparison.family)
                    & (per_seed["split"] == split)
                    & (per_seed["subset"] == subset)
                ]
                pairing_rows.append(
                    {
                        "family": comparison.family,
                        "split": split,
                        "subset": subset,
                        "pair": comparison.pair,
                        "task": comparison.task,
                        "reference_experiment": reference_spec.experiment,
                        "candidate_experiment": candidate_spec.experiment,
                        "n_samples": int(mask.sum()),
                        "n_positive": int(labels.sum()),
                        "test_material_ids_sha256": _material_ids_sha256(
                            test_frame.loc[mask, "material_id"].tolist()
                        ),
                        "validation_n_samples": len(validation_frame),
                        "validation_material_ids_sha256": _material_ids_sha256(
                            validation_frame["material_id"].tolist()
                        ),
                        "paired_across_variants_and_seeds": True,
                        "reference_ensemble_threshold": reference_threshold,
                        "candidate_ensemble_threshold": candidate_threshold,
                    }
                )
                for metric in METRICS:
                    reference_seed = (
                        seed_group.loc[
                            seed_group["variant"] == comparison.reference_variant,
                            ["seed", metric],
                        ]
                        .set_index("seed")
                        .sort_index()
                    )
                    candidate_seed = (
                        seed_group.loc[
                            seed_group["variant"] == comparison.candidate_variant,
                            ["seed", metric],
                        ]
                        .set_index("seed")
                        .sort_index()
                    )
                    if reference_seed.index.tolist() != candidate_seed.index.tolist():
                        raise ValueError(
                            f"Seed mismatch for {comparison.pair}/{split}/{subset}/{metric}"
                        )
                    differences = (
                        candidate_seed[metric].to_numpy(float)
                        - reference_seed[metric].to_numpy(float)
                    )
                    ci_lower = bootstrap[metric]["ci_lower"]
                    ci_upper = bootstrap[metric]["ci_upper"]
                    higher_is_better = metric in {"pr_auc", "roc_auc", "mcc"}
                    significant_favor = (
                        bool(ci_lower > 0.0)
                        if higher_is_better
                        else bool(ci_upper < 0.0)
                    )
                    row: dict[str, object] = {
                        "family": comparison.family,
                        "split": split,
                        "subset": subset,
                        "pair": comparison.pair,
                        "task": comparison.task,
                        "metric": metric,
                        "direction": (
                            "higher_is_better" if higher_is_better else "lower_is_better"
                        ),
                        "difference_definition": "candidate_minus_reference",
                        "reference_variant": comparison.reference_variant,
                        "candidate_variant": comparison.candidate_variant,
                        "reference_experiment": reference_spec.experiment,
                        "candidate_experiment": candidate_spec.experiment,
                        "n_samples": int(mask.sum()),
                        "n_positive": int(labels.sum()),
                        "n_seeds": len(seeds),
                        "reference_ensemble_threshold": reference_threshold,
                        "candidate_ensemble_threshold": candidate_threshold,
                        "reference_seed_mean": float(reference_seed[metric].mean()),
                        "candidate_seed_mean": float(candidate_seed[metric].mean()),
                        "mean_paired_seed_difference": float(differences.mean()),
                        "std_paired_seed_difference": float(differences.std(ddof=1)),
                        "reference_ensemble_metric": reference_metrics[metric],
                        "candidate_ensemble_metric": candidate_metrics[metric],
                        "ensemble_difference_candidate_minus_reference": (
                            candidate_metrics[metric] - reference_metrics[metric]
                        ),
                        "paired_bootstrap_ci_lower": ci_lower,
                        "paired_bootstrap_ci_upper": ci_upper,
                        "paired_bootstrap_two_sided_p": bootstrap[metric][
                            "two_sided_p"
                        ],
                        "bootstrap_replicates_requested": int(bootstrap_replicates),
                        "bootstrap_replicates_valid": bootstrap[metric][
                            "bootstrap_replicates_valid"
                        ],
                        "ci95_significantly_favors_candidate": significant_favor,
                    }
                    for seed, value in zip(seeds, differences):
                        row[f"difference_seed_{seed}"] = float(value)
                    paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)
    pairing_audit = pd.DataFrame(pairing_rows)
    summary: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "state": "completed",
        "analysis_type": "strict_review_gnn_five_seed_paired_summary",
        "scope": scope,
        "review_results_dir": str(review_dir.resolve()),
        "seeds": [int(seed) for seed in seeds],
        "seed_count": len(seeds),
        "default_review_five_seed_set": list(seeds) == list(DEFAULT_SEEDS),
        "metrics": list(METRICS),
        "subsets": {
            "all_test": "all rows in each strict-split held-out test partition",
            "thermo_positive": (
                "rows with y_thermo=1, verified equivalent to E_hull<=0.10 eV/atom"
            ),
        },
        "comparison_definition": "candidate minus reference",
        "ensemble_threshold_rule": (
            "Maximum-MCC threshold fitted once on the corresponding five-seed "
            "mean validation probability, then frozen for held-out test evaluation"
        ),
        "bootstrap": {
            "method": (
                "paired material-level percentile bootstrap on five-seed mean "
                "held-out test probabilities"
            ),
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "confidence_level": 0.95,
        },
        "counts": {
            "experiments": len(specs),
            "seed_runs": len(specs) * len(seeds),
            "per_seed_metric_rows": len(per_seed),
            "experiment_summary_rows": len(experiment_summary),
            "paired_metric_rows": len(paired),
            "pairing_audit_rows": len(pairing_audit),
        },
        "pairing_audit": pairing_audit.to_dict(orient="records"),
        "input_manifest": manifest,
        "headline_pr_auc": paired.loc[
            paired["metric"] == "pr_auc",
            [
                "family",
                "split",
                "subset",
                "pair",
                "reference_ensemble_metric",
                "candidate_ensemble_metric",
                "ensemble_difference_candidate_minus_reference",
                "paired_bootstrap_ci_lower",
                "paired_bootstrap_ci_upper",
                "paired_bootstrap_two_sided_p",
            ],
        ].to_dict(orient="records"),
    }
    return per_seed, experiment_summary, paired, pairing_audit, summary


def write_complete_outputs(
    output_dir: Path,
    progress: pd.DataFrame,
    results: tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]
    ],
) -> None:
    per_seed, experiment_summary, paired, pairing_audit, summary = results
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    experiment_summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    pairing_audit.to_csv(output_dir / "pairing_audit.csv", index=False)
    write_json_atomic(output_dir / "summary.json", summary)
    with pd.ExcelWriter(output_dir / "summary.xlsx", engine="openpyxl") as writer:
        progress.to_excel(writer, sheet_name="progress", index=False)
        experiment_summary.to_excel(writer, sheet_name="experiment_summary", index=False)
        per_seed.to_excel(writer, sheet_name="per_seed_metrics", index=False)
        paired.to_excel(writer, sheet_name="paired_comparisons", index=False)
        pairing_audit.to_excel(writer, sheet_name="pairing_audit", index=False)


def execute(
    review_dir: Path,
    output_dir: Path,
    scope: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260809,
    ece_bins: int = 15,
    allow_pending: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress, status = inspect_progress(review_dir, scope, seeds)
    progress.to_csv(output_dir / "progress.csv", index=False)
    write_json_atomic(output_dir / "status.json", status)
    if status["state"] != "completed":
        print(json.dumps(status, ensure_ascii=False, indent=2))
        if status["state"] == "pending" and allow_pending:
            return 0
        return 2 if status["state"] == "pending" else 1

    validating_status = dict(status)
    validating_status.update(
        {
            "generated_at_utc": utc_now(),
            "state": "validating",
            "summary_artifacts_valid": False,
        }
    )
    write_json_atomic(output_dir / "status.json", validating_status)
    try:
        results = summarize_complete(
            review_dir,
            scope,
            seeds,
            bootstrap_replicates,
            bootstrap_seed,
            ece_bins,
        )
        write_complete_outputs(output_dir, progress, results)
    except Exception as error:
        invalid_status = dict(status)
        invalid_status.update(
            {
                "generated_at_utc": utc_now(),
                "state": "invalid",
                "summary_artifacts_valid": False,
                "analysis_error": f"{type(error).__name__}: {error}",
            }
        )
        write_json_atomic(output_dir / "status.json", invalid_status)
        raise

    completed_status = dict(status)
    completed_status.update(
        {
            "generated_at_utc": utc_now(),
            "state": "completed",
            "summary_artifacts_valid": True,
            "bootstrap_replicates": int(bootstrap_replicates),
            "output_files": [
                "progress.csv",
                "per_seed_metrics.csv",
                "experiment_summary.csv",
                "paired_comparisons.csv",
                "pairing_audit.csv",
                "summary.json",
                "summary.xlsx",
            ],
        }
    )
    write_json_atomic(output_dir / "status.json", completed_status)
    print(json.dumps(completed_status, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--scope", choices=SCOPES, default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Exit 0 for pending runs while still marking summary artifacts invalid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    review_dir = project_root / "results" / "review"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else review_dir / f"strict_gnn_summary_{args.scope}"
    )
    return execute(
        review_dir,
        output_dir,
        args.scope,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        ece_bins=args.ece_bins,
        allow_pending=args.allow_pending,
    )


if __name__ == "__main__":
    raise SystemExit(main())
