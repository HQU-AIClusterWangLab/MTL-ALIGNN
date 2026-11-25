from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, matthews_corrcoef

from models.uncertainty import aggregate_binary_probabilities
from train.metrics import (
    best_mcc_threshold,
    binary_nll,
    classification_metrics,
    sigmoid,
    uncertainty_metrics,
)


UNCERTAINTY_SCORES = {
    "predictive_std": "std",
    "predictive_entropy": "predictive_entropy",
    "mutual_information": "mutual_information",
}


def seed_from_dir(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


def available_seeds(root: Path) -> set[int]:
    seeds = set()
    for path in root.glob("seed_*"):
        if not path.is_dir():
            continue
        try:
            seed = seed_from_dir(path)
        except (ValueError, IndexError):
            continue
        if (path / "val_predictions.csv").exists() and (path / "test_predictions.csv").exists():
            seeds.add(seed)
    return seeds


def load_seed_metrics(root: Path, label: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = root / f"seed_{seed}" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data["test_joint"]
        rows.append(
            {
                "evaluation": label,
                "seed": int(seed),
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "mcc": metrics["mcc"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "nll": metrics["nll"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
            }
        )
    return pd.DataFrame(rows)


def load_prediction_ensemble(root: Path, seeds: list[int], split: str):
    frames = []
    for seed in seeds:
        path = root / f"seed_{seed}" / f"{split}_predictions.csv"
        frame = pd.read_csv(path).sort_values("material_id").reset_index(drop=True)
        required = {"material_id", "y_joint", "p_joint", "logit_joint"}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{path} missing required columns: {sorted(missing)}")
        frames.append(frame)

    base = frames[0].copy()
    for frame, seed in zip(frames[1:], seeds[1:]):
        if frame["material_id"].astype(str).tolist() != base["material_id"].astype(str).tolist():
            raise RuntimeError(f"Prediction order/material set changed for seed {seed} in {root}.")
        if not np.array_equal(frame["y_joint"].to_numpy(int), base["y_joint"].to_numpy(int)):
            raise RuntimeError(f"Joint labels changed across seeds in {root}.")

    probabilities = np.vstack([frame["p_joint"].to_numpy(float) for frame in frames])
    logits = np.vstack([frame["logit_joint"].to_numpy(float) for frame in frames])
    return base, probabilities, logits


def fit_temperature(labels, logits_matrix: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    logits_matrix = np.asarray(logits_matrix, dtype=float)
    if logits_matrix.ndim == 1:
        logits_matrix = logits_matrix[None, :]

    def objective(log_temperature):
        temperature = float(np.exp(log_temperature))
        probability = sigmoid(logits_matrix / temperature).mean(axis=0)
        return binary_nll(labels, probability)

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError(result)
    return float(np.exp(result.x))


def safe_pr_auc(labels, probability) -> float:
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0 or np.unique(labels).size < 2:
        return math.nan
    return float(average_precision_score(labels, probability))


def selective_summary(
    evaluation: str,
    score_name: str,
    labels: np.ndarray,
    probability: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float,
    coverages=(1.00, 0.95, 0.90, 0.80, 0.70),
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    order = np.argsort(uncertainty, kind="mergesort")
    rows = []
    for coverage in coverages:
        retained = max(1, int(np.floor(float(coverage) * len(labels))))
        idx = order[:retained]
        y = labels[idx]
        p = probability[idx]
        pred = (p >= threshold).astype(int)
        error_rate = float((pred != y).mean())
        rows.append(
            {
                "evaluation": evaluation,
                "uncertainty_score": score_name,
                "coverage": float(coverage),
                "retained_count": int(retained),
                "risk": error_rate,
                "accuracy": 1.0 - error_rate,
                "mcc": float(matthews_corrcoef(y, pred)) if np.unique(y).size > 1 else math.nan,
                "pr_auc": safe_pr_auc(y, p),
                "mean_uncertainty": float(np.mean(uncertainty[idx])),
                "median_uncertainty": float(np.median(uncertainty[idx])),
            }
        )
    return pd.DataFrame(rows)


def evaluate_ensemble(root: Path, label: str, seeds: list[int], calibration_bins: int):
    val, val_prob_matrix, val_logit_matrix = load_prediction_ensemble(root, seeds, "val")
    test, test_prob_matrix, test_logit_matrix = load_prediction_ensemble(root, seeds, "test")

    val_stats = aggregate_binary_probabilities(val_prob_matrix)
    test_stats = aggregate_binary_probabilities(test_prob_matrix)

    temperature = fit_temperature(val["y_joint"], val_logit_matrix)
    val_probability = sigmoid(val_logit_matrix / temperature).mean(axis=0)
    test_probability = sigmoid(test_logit_matrix / temperature).mean(axis=0)
    threshold = best_mcc_threshold(val["y_joint"], val_probability)

    classification = classification_metrics(
        test["y_joint"], test_probability, threshold, calibration_bins
    )
    ensemble_row = {
        "evaluation": label,
        "n_members": len(seeds),
        "seeds": ",".join(str(seed) for seed in seeds),
        "temperature": temperature,
        "threshold": threshold,
        "test_count": int(len(test)),
        "positive_rate": float(test["y_joint"].mean()),
        **classification,
    }

    uncertainty_rows = []
    selective_frames = []
    score_arrays = {}
    for score_name, stats_key in UNCERTAINTY_SCORES.items():
        score = np.asarray(test_stats[stats_key], dtype=float)
        score_arrays[score_name] = score
        metrics = uncertainty_metrics(
            test["y_joint"], test_probability, score, threshold
        )
        uncertainty_rows.append(
            {
                "evaluation": label,
                "uncertainty_score": score_name,
                "mean_uncertainty": float(np.mean(score)),
                "median_uncertainty": float(np.median(score)),
                "std_uncertainty": float(np.std(score, ddof=1)) if len(score) > 1 else 0.0,
                **metrics,
            }
        )
        selective_frames.append(
            selective_summary(
                label,
                score_name,
                test["y_joint"].to_numpy(int),
                test_probability,
                score,
                threshold,
            )
        )

    detailed = test[[c for c in test.columns if c in {"material_id", "row_id", "formula", "y_joint", "y_thermo", "y_dynamic"}]].copy()
    detailed["p_joint_calibrated"] = test_probability
    detailed["pred_joint"] = (test_probability >= threshold).astype(int)
    detailed["is_error"] = (detailed["pred_joint"].to_numpy(int) != detailed["y_joint"].to_numpy(int)).astype(int)
    for score_name, score in score_arrays.items():
        detailed[score_name] = score

    return (
        ensemble_row,
        pd.DataFrame(uncertainty_rows),
        pd.concat(selective_frames, ignore_index=True),
        detailed,
    )


def paired_seed_differences(detail: pd.DataFrame, iid_label: str = "IID") -> pd.DataFrame:
    iid = detail[detail["evaluation"] == iid_label].set_index("seed")
    metrics = [
        "pr_auc",
        "roc_auc",
        "mcc",
        "f1",
        "balanced_accuracy",
        "nll",
        "brier",
        "ece",
    ]
    rows = []
    for label in sorted(set(detail["evaluation"]) - {iid_label}):
        other = detail[detail["evaluation"] == label].set_index("seed")
        common = sorted(set(iid.index) & set(other.index))
        for seed in common:
            row = {"evaluation": label, "seed": int(seed)}
            for metric in metrics:
                row[f"delta_{metric}_ood_minus_iid"] = float(other.loc[seed, metric] - iid.loc[seed, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def uncertainty_shift_table(detailed_by_eval: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "IID" not in detailed_by_eval:
        return pd.DataFrame()
    iid = detailed_by_eval["IID"]
    rows = []
    for label, frame in detailed_by_eval.items():
        if label == "IID":
            continue
        for score_name in UNCERTAINTY_SCORES:
            iid_values = iid[score_name].to_numpy(float)
            ood_values = frame[score_name].to_numpy(float)
            test = mannwhitneyu(ood_values, iid_values, alternative="two-sided")
            effect = 2.0 * float(test.statistic) / (len(ood_values) * len(iid_values)) - 1.0
            rows.append(
                {
                    "evaluation": label,
                    "uncertainty_score": score_name,
                    "iid_mean": float(np.mean(iid_values)),
                    "ood_mean": float(np.mean(ood_values)),
                    "mean_difference_ood_minus_iid": float(np.mean(ood_values) - np.mean(iid_values)),
                    "iid_median": float(np.median(iid_values)),
                    "ood_median": float(np.median(ood_values)),
                    "median_difference_ood_minus_iid": float(np.median(ood_values) - np.median(iid_values)),
                    "mann_whitney_u": float(test.statistic),
                    "mann_whitney_pvalue": float(test.pvalue),
                    "rank_biserial_effect_positive_means_ood_higher": effect,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iid-dir", required=True)
    parser.add_argument("--ood-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-bins", type=int, default=15)
    args = parser.parse_args()

    iid_dir = Path(args.iid_dir)
    ood_dirs = [Path(value) for value in args.ood_dir]
    roots = [("IID", iid_dir)] + [(path.name, path) for path in ood_dirs]

    seed_sets = [available_seeds(root) for _, root in roots]
    common_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if len(common_seeds) < 2:
        raise RuntimeError(
            f"Need at least two common completed seeds across IID/OOD runs. Found: {common_seeds}"
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Fair per-seed comparison: only identical seeds across all evaluation scenarios.
    seed_frames = [load_seed_metrics(root, label, common_seeds) for label, root in roots]
    detail = pd.concat(seed_frames, ignore_index=True)
    detail.to_csv(output / "ood_seed_metrics_common_seeds.csv", index=False)

    seed_summary_rows = []
    for label, group in detail.groupby("evaluation", sort=False):
        row = {"evaluation": label, "n_seeds": len(group), "seeds": ",".join(map(str, common_seeds))}
        for column in group.select_dtypes(include=[np.number]).columns:
            if column == "seed":
                continue
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        seed_summary_rows.append(row)
    seed_summary = pd.DataFrame(seed_summary_rows)
    seed_summary.to_csv(output / "ood_seed_summary_common_seeds.csv", index=False)

    paired = paired_seed_differences(detail)
    paired.to_csv(output / "ood_paired_seed_differences.csv", index=False)

    ensemble_rows = []
    uncertainty_frames = []
    selective_frames = []
    detailed_by_eval = {}
    for label, root in roots:
        ensemble_row, uncertainty, selective, detailed_predictions = evaluate_ensemble(
            root, label, common_seeds, args.calibration_bins
        )
        ensemble_rows.append(ensemble_row)
        uncertainty_frames.append(uncertainty)
        selective_frames.append(selective)
        detailed_by_eval[label] = detailed_predictions
        detailed_predictions.to_csv(
            output / f"{label}_ensemble_test_predictions.csv", index=False
        )

    ensemble_summary = pd.DataFrame(ensemble_rows)
    ensemble_summary.to_csv(output / "ood_ensemble_summary_common_members.csv", index=False)

    uncertainty_summary = pd.concat(uncertainty_frames, ignore_index=True)
    uncertainty_summary.to_csv(output / "ood_uncertainty_quality.csv", index=False)

    selective_summary_table = pd.concat(selective_frames, ignore_index=True)
    selective_summary_table.to_csv(output / "ood_selective_prediction.csv", index=False)

    shift = uncertainty_shift_table(detailed_by_eval)
    shift.to_csv(output / "ood_uncertainty_shift_vs_iid.csv", index=False)

    metadata = {
        "status": "completed",
        "common_seeds": common_seeds,
        "n_common_seeds": len(common_seeds),
        "iid_directory": str(iid_dir),
        "ood_directories": [str(path) for path in ood_dirs],
        "ensemble_interpretation": (
            "IID and each OOD scenario are compared using the same common seeds. "
            "Each scenario forms an equal-size Deep Ensemble from those members. "
            "Temperature and MCC threshold are fitted only on that scenario's validation split; test data are not used for selection."
        ),
        "primary_uq_outputs": [
            "ood_uncertainty_quality.csv",
            "ood_uncertainty_shift_vs_iid.csv",
            "ood_selective_prediction.csv",
        ],
    }
    (output / "OOD_EVALUATION.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Common seeds:", common_seeds)
    print("\n=== Equal-member ensemble performance ===")
    print(ensemble_summary.to_string(index=False))
    print("\n=== Uncertainty quality ===")
    print(uncertainty_summary.to_string(index=False))
    if not shift.empty:
        print("\n=== OOD uncertainty shift vs IID ===")
        print(shift.to_string(index=False))


if __name__ == "__main__":
    main()
