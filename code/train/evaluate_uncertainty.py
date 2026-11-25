from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models.uncertainty import aggregate_binary_probabilities, binary_entropy
from train.common import save_json
from train.metrics import (
    best_mcc_threshold,
    binary_nll,
    classification_metrics,
    risk_coverage_curve,
    sigmoid,
    uncertainty_metrics,
)


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


def calibrate_from_logits(
    val: pd.DataFrame,
    test: pd.DataFrame,
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    calibration_bins: int,
):
    temperature = fit_temperature(val["y_joint"], val_logits)
    val_probability = sigmoid(val_logits / temperature).mean(axis=0)
    test_probability = sigmoid(test_logits / temperature).mean(axis=0)
    threshold = best_mcc_threshold(val["y_joint"], val_probability)
    metrics = classification_metrics(
        test["y_joint"], test_probability, threshold, calibration_bins
    )
    return temperature, threshold, val_probability, test_probability, metrics


def sample_matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = sorted(
        [column for column in frame if column.startswith(prefix)],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if not columns:
        raise RuntimeError(f"No columns with prefix {prefix!r}.")
    return frame[columns].to_numpy(dtype=float).T


def evaluate_score(
    method: str,
    score_name: str,
    val,
    test,
    val_probability,
    test_probability,
    threshold,
    val_score,
    test_score,
    output_dir: Path,
):
    val_metrics = uncertainty_metrics(
        val["y_joint"], val_probability, val_score, threshold
    )
    test_metrics = uncertainty_metrics(
        test["y_joint"], test_probability, test_score, threshold
    )
    curve = risk_coverage_curve(
        test["y_joint"], test_probability, test_score, threshold
    )
    curve["method"] = method
    curve["uncertainty_score"] = score_name
    curve.to_csv(
        output_dir / f"risk_coverage_{method}_{score_name}.csv", index=False
    )
    return {
        "method": method,
        "uncertainty_score": score_name,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def mc_sample_ablation(
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    calibration_bins: int,
):
    val_probability_matrix = sample_matrix(val, "p_joint_sample_")
    test_probability_matrix = sample_matrix(test, "p_joint_sample_")
    val_logit_matrix = sample_matrix(val, "logit_joint_sample_")
    test_logit_matrix = sample_matrix(test, "logit_joint_sample_")
    counts = [
        count
        for count in (5, 10, 20, 30, 50, 100)
        if count <= val_probability_matrix.shape[0]
    ]
    rows = []
    for count in counts:
        val_stats = aggregate_binary_probabilities(
            val_probability_matrix[:count]
        )
        test_stats = aggregate_binary_probabilities(
            test_probability_matrix[:count]
        )
        temperature, threshold, _val_p, test_p, classification = calibrate_from_logits(
            val,
            test,
            val_logit_matrix[:count],
            test_logit_matrix[:count],
            calibration_bins,
        )
        uncertainty = uncertainty_metrics(
            test["y_joint"], test_p, test_stats["std"], threshold
        )
        rows.append(
            {
                "mc_samples": count,
                "temperature": temperature,
                **classification,
                **uncertainty,
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "mc_sample_count_ablation.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-experiment-dir", required=True)
    parser.add_argument("--single-experiment-dir", required=True)
    parser.add_argument("--calibration-bins", type=int, default=15)
    args = parser.parse_args()

    full_dir = Path(args.full_experiment_dir)
    single_dir = Path(args.single_experiment_dir)
    output_dir = full_dir / "uncertainty_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    ensemble_val = pd.read_csv(
        full_dir / "ensemble" / "val_ensemble_predictions.csv"
    )
    ensemble_test = pd.read_csv(
        full_dir / "ensemble" / "test_ensemble_predictions.csv"
    )
    mc_val = pd.read_csv(full_dir / "mc_dropout" / "val_mc_predictions.csv")
    mc_test = pd.read_csv(full_dir / "mc_dropout" / "test_mc_predictions.csv")
    single_seed_dirs = sorted(single_dir.glob("seed_*"))
    if not single_seed_dirs:
        raise RuntimeError(f"No single-model seed outputs in {single_dir}")
    single_val = pd.read_csv(single_seed_dirs[0] / "val_predictions.csv")
    single_test = pd.read_csv(single_seed_dirs[0] / "test_predictions.csv")

    method_rows = []
    uncertainty_rows = []

    # Single model temperature scaling uses the native model logit.
    single_values = calibrate_from_logits(
        single_val,
        single_test,
        single_val["logit_joint"].to_numpy(float)[None, :],
        single_test["logit_joint"].to_numpy(float)[None, :],
        args.calibration_bins,
    )
    single_temperature, single_threshold, single_val_p, single_test_p, single_metrics = single_values
    method_rows.append(
        {
            "method": "single_model",
            "temperature": single_temperature,
            "threshold": single_threshold,
            **single_metrics,
        }
    )
    uncertainty_rows.append(
        evaluate_score(
            "single_model",
            "predictive_entropy",
            single_val,
            single_test,
            single_val_p,
            single_test_p,
            single_threshold,
            binary_entropy(single_val_p),
            binary_entropy(single_test_p),
            output_dir,
        )
    )

    mc_val_logits = sample_matrix(mc_val, "logit_joint_sample_")
    mc_test_logits = sample_matrix(mc_test, "logit_joint_sample_")
    mc_values = calibrate_from_logits(
        mc_val,
        mc_test,
        mc_val_logits,
        mc_test_logits,
        args.calibration_bins,
    )
    mc_temperature, mc_threshold, mc_val_p, mc_test_p, mc_metrics = mc_values
    method_rows.append(
        {
            "method": "mc_dropout",
            "temperature": mc_temperature,
            "threshold": mc_threshold,
            **mc_metrics,
        }
    )
    mc_scores = {
        "predictive_std": "p_joint_std",
        "predictive_entropy": "joint_predictive_entropy",
        "mutual_information": "joint_mutual_information",
    }
    for score_name, column in mc_scores.items():
        uncertainty_rows.append(
            evaluate_score(
                "mc_dropout",
                score_name,
                mc_val,
                mc_test,
                mc_val_p,
                mc_test_p,
                mc_threshold,
                mc_val[column],
                mc_test[column],
                output_dir,
            )
        )

    ensemble_thresholds = json.loads(
        (full_dir / "ensemble" / "thresholds.json").read_text(encoding="utf-8")
    )
    ensemble_temperatures = json.loads(
        (full_dir / "ensemble" / "temperatures.json").read_text(encoding="utf-8")
    )
    ensemble_threshold = float(ensemble_thresholds["joint"])
    ensemble_val_p = ensemble_val["p_joint_calibrated"].to_numpy(float)
    ensemble_test_p = ensemble_test["p_joint_calibrated"].to_numpy(float)
    ensemble_metrics = classification_metrics(
        ensemble_test["y_joint"],
        ensemble_test_p,
        ensemble_threshold,
        args.calibration_bins,
    )
    method_rows.append(
        {
            "method": "deep_ensemble",
            "temperature": ensemble_temperatures["joint"],
            "threshold": ensemble_threshold,
            **ensemble_metrics,
        }
    )
    ensemble_scores = {
        "predictive_std": "p_joint_std",
        "predictive_entropy": "joint_predictive_entropy",
        "mutual_information": "joint_mutual_information",
    }
    for score_name, column in ensemble_scores.items():
        uncertainty_rows.append(
            evaluate_score(
                "deep_ensemble",
                score_name,
                ensemble_val,
                ensemble_test,
                ensemble_val_p,
                ensemble_test_p,
                ensemble_threshold,
                ensemble_val[column],
                ensemble_test[column],
                output_dir,
            )
        )

    method_table = pd.DataFrame(method_rows)
    uncertainty_table = pd.DataFrame(uncertainty_rows)
    method_table.to_csv(
        output_dir / "classification_and_calibration.csv", index=False
    )
    uncertainty_table.to_csv(output_dir / "uncertainty_quality.csv", index=False)

    candidates = uncertainty_table[
        uncertainty_table["method"] == "deep_ensemble"
    ].sort_values(["val_aurc", "val_error_auprc"], ascending=[True, False])
    selected = str(candidates.iloc[0]["uncertainty_score"])
    selection = {
        "method": "deep_ensemble",
        "score": selected,
        "column": ensemble_scores[selected],
        "selection_set": "validation",
        "selection_rule": "minimum validation AURC; error AUPRC tie-break",
    }
    save_json(selection, output_dir / "uncertainty_selection.json")
    mc_sample_ablation(mc_val, mc_test, output_dir, args.calibration_bins)
    save_json(
        {"status": "completed", **selection}, output_dir / "COMPLETED.json"
    )
    print(method_table.to_string(index=False))
    print(uncertainty_table.to_string(index=False))
    print(f"Selected ensemble uncertainty score: {selection}")


if __name__ == "__main__":
    main()
