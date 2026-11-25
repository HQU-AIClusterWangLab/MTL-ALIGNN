"""Summarize the paired XGBoost deployment baselines requested in review.

The script is deliberately read-only with respect to model outputs. It loads
the frozen R11--R18 validation and test predictions, verifies exact
material-level pairing, fits each five-seed ensemble threshold once on its mean
validation probability, and compares Dynamic-only against Dynamic+E_hull on
both the complete test set and the deployment-relevant Thermo-positive subset
(E_hull <= 0.10 eV/atom).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from train.metrics import (
    best_mcc_threshold,
    binary_nll,
    expected_calibration_error,
)


DEFAULT_SEEDS = (2025, 2026, 2027, 2028, 2029)
METRICS = ("pr_auc", "roc_auc", "mcc", "nll", "brier", "ece")
EXPERIMENTS = {
    "random": {
        "dynamic_only": "R11_dynamic_only_xgb_random",
        "dynamic_ehull": "R12_dynamic_ehull_xgb_random",
    },
    "composition": {
        "dynamic_only": "R13_dynamic_only_xgb_composition",
        "dynamic_ehull": "R14_dynamic_ehull_xgb_composition",
    },
    "chemical_system": {
        "dynamic_only": "R15_dynamic_only_xgb_chemical",
        "dynamic_ehull": "R16_dynamic_ehull_xgb_chemical",
    },
    "structure_family": {
        "dynamic_only": "R17_dynamic_only_xgb_structure",
        "dynamic_ehull": "R18_dynamic_ehull_xgb_structure",
    },
}


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    source: Path | str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{source} missing required columns: {missing}")


def read_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(
        frame,
        ("material_id", "ehull", "y_thermo", "y_dynamic", "p_dynamic"),
        path,
    )
    frame = frame.copy()
    frame["material_id"] = frame["material_id"].astype(str)
    if frame["material_id"].duplicated().any():
        duplicates = frame.loc[
            frame["material_id"].duplicated(keep=False), "material_id"
        ].head(5)
        raise ValueError(f"Duplicate material_id values in {path}: {duplicates.tolist()}")
    for column in ("ehull", "y_thermo", "y_dynamic", "p_dynamic"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(f"Non-finite {column} values in {path}")
    if not frame["y_thermo"].isin([0, 1]).all():
        raise ValueError(f"Non-binary y_thermo values in {path}")
    if not frame["y_dynamic"].isin([0, 1]).all():
        raise ValueError(f"Non-binary y_dynamic values in {path}")
    if not frame["p_dynamic"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"p_dynamic outside [0, 1] in {path}")
    return frame.sort_values("material_id", kind="mergesort").reset_index(drop=True)


def read_threshold(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    try:
        threshold = float(value["dynamic"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid dynamic threshold in {path}") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Dynamic threshold outside [0, 1] in {path}: {threshold}")
    return threshold


def align_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    reference_name: str,
    candidate_name: str,
) -> None:
    left_ids = reference["material_id"].tolist()
    right_ids = candidate["material_id"].tolist()
    if left_ids != right_ids:
        left_only = sorted(set(left_ids) - set(right_ids))[:5]
        right_only = sorted(set(right_ids) - set(left_ids))[:5]
        raise ValueError(
            f"material_id mismatch for {reference_name} vs {candidate_name}; "
            f"left_only={left_only}, right_only={right_only}"
        )
    for column in ("y_thermo", "y_dynamic"):
        if not np.array_equal(
            reference[column].to_numpy(int), candidate[column].to_numpy(int)
        ):
            raise ValueError(
                f"{column} mismatch for {reference_name} vs {candidate_name}"
            )
    if not np.allclose(
        reference["ehull"].to_numpy(float),
        candidate["ehull"].to_numpy(float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"ehull mismatch for {reference_name} vs {candidate_name}")


def classification_metrics(
    labels: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    threshold: float,
    ece_bins: int = 15,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probability must be equally sized non-empty vectors")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive")
    two_classes = np.unique(y).size == 2
    prediction = (p >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)) if two_classes else math.nan,
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else math.nan,
        "mcc": float(matthews_corrcoef(y, prediction)) if two_classes else 0.0,
        "nll": binary_nll(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, n_bins=ece_bins),
    }


def subset_mask(frame: pd.DataFrame, subset: str) -> np.ndarray:
    if subset == "all_test":
        return np.ones(len(frame), dtype=bool)
    if subset == "thermo_positive":
        by_label = frame["y_thermo"].to_numpy(int) == 1
        by_threshold = frame["ehull"].to_numpy(float) <= 0.10 + 1.0e-12
        if not np.array_equal(by_label, by_threshold):
            mismatch = frame.loc[by_label != by_threshold, "material_id"].head(5)
            raise ValueError(
                "y_thermo does not equal ehull <= 0.10 for material_id values "
                f"{mismatch.tolist()}"
            )
        return by_label
    raise ValueError(f"Unknown subset: {subset}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_metric_differences(
    labels: np.ndarray,
    p_dynamic_only: np.ndarray,
    p_dynamic_ehull: np.ndarray,
    threshold_dynamic_only: float,
    threshold_dynamic_ehull: float,
    replicates: int,
    seed: int,
    ece_bins: int,
) -> dict[str, dict[str, float]]:
    """Paired material bootstrap of five-seed ensemble predictions."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    arrays = (labels, p_dynamic_only, p_dynamic_ehull)
    if any(np.asarray(item).ndim != 1 for item in arrays):
        raise ValueError("bootstrap inputs must be one-dimensional")
    if len({len(item) for item in arrays}) != 1 or len(labels) == 0:
        raise ValueError("bootstrap inputs must have equal non-zero length")

    rng = np.random.default_rng(seed)
    differences: dict[str, list[float]] = {metric: [] for metric in METRICS}
    n_samples = len(labels)
    for _ in range(replicates):
        indices = rng.integers(0, n_samples, size=n_samples)
        sampled_y = labels[indices]
        only = classification_metrics(
            sampled_y,
            p_dynamic_only[indices],
            threshold_dynamic_only,
            ece_bins,
        )
        ehull = classification_metrics(
            sampled_y,
            p_dynamic_ehull[indices],
            threshold_dynamic_ehull,
            ece_bins,
        )
        for metric in METRICS:
            difference = ehull[metric] - only[metric]
            if math.isfinite(difference):
                differences[metric].append(float(difference))

    output: dict[str, dict[str, float]] = {}
    for metric, values in differences.items():
        sample = np.asarray(values, dtype=float)
        if sample.size == 0:
            output[metric] = {
                "bootstrap_replicates_valid": 0,
                "ci_lower": math.nan,
                "ci_upper": math.nan,
                "two_sided_p": math.nan,
            }
            continue
        # The +1 correction avoids reporting an impossible exact p=0 with a
        # finite Monte Carlo sample.
        probability_nonpositive = float(
            (np.count_nonzero(sample <= 0.0) + 1) / (sample.size + 1)
        )
        probability_nonnegative = float(
            (np.count_nonzero(sample >= 0.0) + 1) / (sample.size + 1)
        )
        output[metric] = {
            "bootstrap_replicates_valid": int(sample.size),
            "ci_lower": float(np.quantile(sample, 0.025)),
            "ci_upper": float(np.quantile(sample, 0.975)),
            "two_sided_p": min(
                1.0,
                2.0 * min(probability_nonpositive, probability_nonnegative),
            ),
        }
    return output


def _load_inputs(
    review_dir: Path,
    seeds: Sequence[int],
) -> tuple[
    dict[
        tuple[str, str, int],
        tuple[pd.DataFrame, pd.DataFrame, float],
    ],
    list[dict[str, object]],
]:
    loaded: dict[
        tuple[str, str, int],
        tuple[pd.DataFrame, pd.DataFrame, float],
    ] = {}
    manifest: list[dict[str, object]] = []
    for split, variants in EXPERIMENTS.items():
        split_references: dict[str, tuple[pd.DataFrame, str]] = {}
        for variant, experiment in variants.items():
            for seed in seeds:
                seed_dir = review_dir / experiment / f"seed_{seed}"
                prediction_path = seed_dir / "test_predictions.csv"
                validation_path = seed_dir / "val_predictions.csv"
                threshold_path = seed_dir / "thresholds.json"
                frame = read_prediction(prediction_path)
                validation = read_prediction(validation_path)
                threshold = read_threshold(threshold_path)
                for partition, current, path in (
                    ("test", frame, prediction_path),
                    ("validation", validation, validation_path),
                ):
                    if partition not in split_references:
                        split_references[partition] = (current, str(path))
                    else:
                        reference, reference_name = split_references[partition]
                        align_frames(reference, current, reference_name, str(path))
                loaded[(split, variant, seed)] = (
                    frame,
                    validation,
                    threshold,
                )
                for path in (prediction_path, validation_path, threshold_path):
                    manifest.append(
                        {
                            "path": str(path.resolve()),
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )
    return loaded, manifest


def summarize(
    review_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260809,
    ece_bins: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    loaded, manifest = _load_inputs(review_dir, seeds)

    per_seed_rows: list[dict[str, object]] = []
    for split, variants in EXPERIMENTS.items():
        for variant, experiment in variants.items():
            for seed in seeds:
                frame, _, threshold = loaded[(split, variant, seed)]
                for subset in ("all_test", "thermo_positive"):
                    mask = subset_mask(frame, subset)
                    labels = frame.loc[mask, "y_dynamic"].to_numpy(int)
                    probability = frame.loc[mask, "p_dynamic"].to_numpy(float)
                    metrics = classification_metrics(
                        labels, probability, threshold, ece_bins
                    )
                    per_seed_rows.append(
                        {
                            "split": split,
                            "subset": subset,
                            "variant": variant,
                            "experiment": experiment,
                            "seed": int(seed),
                            "n_samples": int(mask.sum()),
                            "n_positive": int(labels.sum()),
                            "prevalence": float(labels.mean()),
                            "validation_threshold": float(threshold),
                            **metrics,
                        }
                    )
    per_seed = pd.DataFrame(per_seed_rows)

    summary_rows: list[dict[str, object]] = []
    group_columns = ["split", "subset", "variant", "experiment"]
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
    baseline_summary = pd.DataFrame(summary_rows)

    paired_rows: list[dict[str, object]] = []
    pairing_audit: list[dict[str, object]] = []
    for split_index, split in enumerate(EXPERIMENTS):
        reference, validation_reference, _ = loaded[
            (split, "dynamic_only", seeds[0])
        ]
        validation_labels = validation_reference["y_dynamic"].to_numpy(int)
        validation_probability: dict[str, np.ndarray] = {}
        ensemble_threshold: dict[str, float] = {}
        for variant in ("dynamic_only", "dynamic_ehull"):
            validation_probability[variant] = np.mean(
                [
                    loaded[(split, variant, seed)][1]["p_dynamic"].to_numpy(
                        float
                    )
                    for seed in seeds
                ],
                axis=0,
            )
            ensemble_threshold[variant] = best_mcc_threshold(
                validation_labels,
                validation_probability[variant],
            )
        for subset_index, subset in enumerate(("all_test", "thermo_positive")):
            mask = subset_mask(reference, subset)
            labels = reference.loc[mask, "y_dynamic"].to_numpy(int)
            probability: dict[str, np.ndarray] = {}
            for variant in ("dynamic_only", "dynamic_ehull"):
                probability[variant] = np.mean(
                    [
                        loaded[(split, variant, seed)][0]
                        .loc[mask, "p_dynamic"]
                        .to_numpy(float)
                        for seed in seeds
                    ],
                    axis=0,
                )
            ensemble_metrics = {
                variant: classification_metrics(
                    labels,
                    probability[variant],
                    ensemble_threshold[variant],
                    ece_bins,
                )
                for variant in ("dynamic_only", "dynamic_ehull")
            }
            group_seed_rows = per_seed.loc[
                (per_seed["split"] == split) & (per_seed["subset"] == subset)
            ]
            seed_differences: dict[str, list[float]] = {}
            for metric in METRICS:
                only = (
                    group_seed_rows.loc[
                        group_seed_rows["variant"] == "dynamic_only",
                        ["seed", metric],
                    ]
                    .set_index("seed")
                    .sort_index()
                )
                ehull = (
                    group_seed_rows.loc[
                        group_seed_rows["variant"] == "dynamic_ehull",
                        ["seed", metric],
                    ]
                    .set_index("seed")
                    .sort_index()
                )
                if only.index.tolist() != ehull.index.tolist():
                    raise ValueError(f"Seed mismatch for {split}/{subset}/{metric}")
                seed_differences[metric] = (
                    ehull[metric].to_numpy(float) - only[metric].to_numpy(float)
                ).tolist()

            bootstrap = _bootstrap_metric_differences(
                labels,
                probability["dynamic_only"],
                probability["dynamic_ehull"],
                ensemble_threshold["dynamic_only"],
                ensemble_threshold["dynamic_ehull"],
                bootstrap_replicates,
                bootstrap_seed + split_index * 100 + subset_index,
                ece_bins,
            )
            pairing_audit.append(
                {
                    "split": split,
                    "subset": subset,
                    "n_samples": int(mask.sum()),
                    "n_positive": int(labels.sum()),
                    "material_ids_sha256": hashlib.sha256(
                        "\n".join(reference.loc[mask, "material_id"]).encode("utf-8")
                    ).hexdigest(),
                    "paired_across_variants_and_seeds": True,
                    "validation_n_samples": len(validation_reference),
                    "validation_material_ids_sha256": hashlib.sha256(
                        "\n".join(validation_reference["material_id"]).encode("utf-8")
                    ).hexdigest(),
                    "validation_paired_across_variants_and_seeds": True,
                    "dynamic_only_ensemble_threshold": ensemble_threshold[
                        "dynamic_only"
                    ],
                    "dynamic_ehull_ensemble_threshold": ensemble_threshold[
                        "dynamic_ehull"
                    ],
                }
            )
            for metric in METRICS:
                differences = np.asarray(seed_differences[metric], dtype=float)
                row: dict[str, object] = {
                    "split": split,
                    "subset": subset,
                    "metric": metric,
                    "direction": (
                        "higher_is_better"
                        if metric in {"pr_auc", "roc_auc", "mcc"}
                        else "lower_is_better"
                    ),
                    "n_samples": int(mask.sum()),
                    "n_positive": int(labels.sum()),
                    "n_seeds": len(seeds),
                    "dynamic_only_ensemble_threshold": ensemble_threshold[
                        "dynamic_only"
                    ],
                    "dynamic_ehull_ensemble_threshold": ensemble_threshold[
                        "dynamic_ehull"
                    ],
                    "dynamic_only_seed_mean": float(
                        group_seed_rows.loc[
                            group_seed_rows["variant"] == "dynamic_only", metric
                        ].mean()
                    ),
                    "dynamic_ehull_seed_mean": float(
                        group_seed_rows.loc[
                            group_seed_rows["variant"] == "dynamic_ehull", metric
                        ].mean()
                    ),
                    "mean_paired_seed_difference_ehull_minus_only": float(
                        differences.mean()
                    ),
                    "std_paired_seed_difference": (
                        float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
                    ),
                    "dynamic_only_ensemble_metric": ensemble_metrics[
                        "dynamic_only"
                    ][metric],
                    "dynamic_ehull_ensemble_metric": ensemble_metrics[
                        "dynamic_ehull"
                    ][metric],
                    "ensemble_difference_ehull_minus_only": (
                        ensemble_metrics["dynamic_ehull"][metric]
                        - ensemble_metrics["dynamic_only"][metric]
                    ),
                    "paired_bootstrap_ci_lower": bootstrap[metric]["ci_lower"],
                    "paired_bootstrap_ci_upper": bootstrap[metric]["ci_upper"],
                    "paired_bootstrap_two_sided_p": bootstrap[metric]["two_sided_p"],
                    "bootstrap_replicates_requested": int(bootstrap_replicates),
                    "bootstrap_replicates_valid": bootstrap[metric][
                        "bootstrap_replicates_valid"
                    ],
                }
                for seed, value in zip(seeds, differences):
                    row[f"difference_seed_{seed}"] = float(value)
                paired_rows.append(row)

    paired_comparisons = pd.DataFrame(paired_rows)
    deployment_subset = per_seed.loc[
        per_seed["subset"] == "thermo_positive"
    ].reset_index(drop=True)
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "frozen_prediction_paired_baseline_summary",
        "review_results_dir": str(review_dir.resolve()),
        "experiments": EXPERIMENTS,
        "seeds": [int(seed) for seed in seeds],
        "metrics": list(METRICS),
        "ece_bins": int(ece_bins),
        "subsets": {
            "all_test": "all rows in each strict-split test partition",
            "thermo_positive": (
                "deployment subset with y_thermo=1, verified exactly equivalent "
                "to ehull<=0.10 eV/atom"
            ),
        },
        "comparison": "dynamic_ehull minus dynamic_only",
        "bootstrap": {
            "method": (
                "paired material-level percentile bootstrap on five-seed mean "
                "test ensemble probabilities; each ensemble threshold is fitted "
                "once by maximum MCC on the corresponding five-seed mean "
                "validation probability and then frozen for test evaluation"
            ),
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "confidence_level": 0.95,
        },
        "pairing_audit": pairing_audit,
        "input_manifest": manifest,
        "headline_pr_auc": paired_comparisons.loc[
            paired_comparisons["metric"] == "pr_auc",
            [
                "split",
                "subset",
                "mean_paired_seed_difference_ehull_minus_only",
                "ensemble_difference_ehull_minus_only",
                "paired_bootstrap_ci_lower",
                "paired_bootstrap_ci_upper",
            ],
        ].to_dict(orient="records"),
    }
    return baseline_summary, deployment_subset, paired_comparisons, summary


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def write_outputs(
    output_dir: Path,
    baseline_summary: pd.DataFrame,
    deployment_subset: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    deployment_subset.to_csv(
        output_dir / "deployment_subset_metrics.csv", index=False
    )
    paired_comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    with pd.ExcelWriter(output_dir / "summary.xlsx", engine="openpyxl") as writer:
        baseline_summary.to_excel(writer, sheet_name="baseline_summary", index=False)
        deployment_subset.to_excel(writer, sheet_name="deployment_subset", index=False)
        paired_comparisons.to_excel(writer, sheet_name="paired_comparisons", index=False)
        pd.DataFrame(summary["pairing_audit"]).to_excel(
            writer, sheet_name="pairing_audit", index=False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    review_dir = project_root / "results" / "review"
    results = summarize(
        review_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        ece_bins=args.ece_bins,
    )
    write_outputs(review_dir, *results)
    baseline_summary, deployment_subset, paired_comparisons, _ = results
    print(
        json.dumps(
            {
                "output_dir": str(review_dir),
                "baseline_rows": len(baseline_summary),
                "deployment_rows": len(deployment_subset),
                "comparison_rows": len(paired_comparisons),
                "bootstrap_replicates": args.bootstrap_replicates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
