from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from train.common import save_json
from train.metrics import ranking_metrics


def robust_uncertainty_normalization(values, lower: float, upper: float):
    values = np.asarray(values, dtype=float)
    scale = max(float(upper - lower), 1e-12)
    return np.clip((values - lower) / scale, 0.0, 1.0)


def mean_precision_objective(labels, score, k_values) -> float:
    metrics = ranking_metrics(labels, score, k_values)
    values = [
        metrics[f"precision_at_{k}"]
        for k in k_values
        if not np.isnan(metrics[f"precision_at_{k}"])
    ]
    return float(np.mean(values))


def stratified_bootstrap_indices(labels, generator):
    labels = np.asarray(labels, dtype=int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    sampled = np.concatenate(
        [
            generator.choice(positive, size=len(positive), replace=True),
            generator.choice(negative, size=len(negative), replace=True),
        ]
    )
    generator.shuffle(sampled)
    return sampled


def bootstrap_ranking(
    labels,
    score,
    k_values,
    iterations: int = 1000,
    seed: int = 2025,
):
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    generator = np.random.default_rng(seed)
    records = []
    for _ in range(iterations):
        indices = stratified_bootstrap_indices(labels, generator)
        records.append(ranking_metrics(labels[indices], score[indices], k_values))
    frame = pd.DataFrame(records)
    return {
        column: {
            "lower_95": float(frame[column].quantile(0.025)),
            "upper_95": float(frame[column].quantile(0.975)),
        }
        for column in frame
    }


def paired_bootstrap_difference(
    labels,
    score_left,
    score_right,
    k_values,
    iterations: int = 2000,
    seed: int = 2026,
):
    labels = np.asarray(labels, dtype=int)
    left = np.asarray(score_left, dtype=float)
    right = np.asarray(score_right, dtype=float)
    generator = np.random.default_rng(seed)
    rows = []
    for _ in range(iterations):
        indices = stratified_bootstrap_indices(labels, generator)
        left_metrics = ranking_metrics(labels[indices], left[indices], k_values)
        right_metrics = ranking_metrics(labels[indices], right[indices], k_values)
        rows.append(
            {
                f"precision_difference_at_{k}": (
                    left_metrics[f"precision_at_{k}"]
                    - right_metrics[f"precision_at_{k}"]
                )
                for k in k_values
            }
        )
    frame = pd.DataFrame(rows)
    result = []
    for k in k_values:
        column = f"precision_difference_at_{k}"
        values = frame[column]
        result.append(
            {
                "K": int(k),
                "difference": "risk_aware_minus_joint_probability",
                "mean_difference": float(values.mean()),
                "lower_95": float(values.quantile(0.025)),
                "upper_95": float(values.quantile(0.975)),
                "probability_difference_gt_zero": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(result)


def evaluate_scope(frame, scores: dict[str, np.ndarray], k_values, scope: str):
    rows = []
    bootstrap = {}
    labels = frame["y_joint"].to_numpy(int)
    for method, values in scores.items():
        metrics = ranking_metrics(labels, values, k_values)
        rows.append({"scope": scope, "method": method, **metrics})
        bootstrap[method] = bootstrap_ranking(labels, values, k_values)
    paired = paired_bootstrap_difference(
        labels,
        scores["risk_aware"],
        scores["joint_probability"],
        k_values,
    )
    paired.insert(0, "scope", scope)
    return rows, bootstrap, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-experiment-dir", required=True)
    args = parser.parse_args()

    root = Path(args.full_experiment_dir)
    output = root / "ranking"
    output.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(root / "ensemble" / "val_ensemble_predictions.csv")
    test = pd.read_csv(root / "ensemble" / "test_ensemble_predictions.csv")
    uncertainty_selection = json.loads(
        (root / "uncertainty_evaluation" / "uncertainty_selection.json").read_text(
            encoding="utf-8"
        )
    )
    uncertainty_column = uncertainty_selection["column"]
    resolved = yaml.safe_load(
        (root / "resolved_experiment.yaml").read_text(encoding="utf-8")
    )
    ranking_cfg = resolved["ranking"]
    quantile_low, quantile_high = ranking_cfg["robust_quantiles"]
    lower = float(val[uncertainty_column].quantile(quantile_low))
    upper = float(val[uncertainty_column].quantile(quantile_high))
    val_uncertainty = robust_uncertainty_normalization(
        val[uncertainty_column], lower, upper
    )
    test_uncertainty = robust_uncertainty_normalization(
        test[uncertainty_column], lower, upper
    )

    val_eligible = val["y_thermo"].to_numpy(int) == 1
    validation_k = ranking_cfg["validation_k"]
    lambda_rows = []
    for risk_lambda in ranking_cfg["lambdas"]:
        score = (
            val["p_joint_calibrated"].to_numpy(float)
            - float(risk_lambda) * val_uncertainty
        )
        objective = mean_precision_objective(
            val.loc[val_eligible, "y_joint"], score[val_eligible], validation_k
        )
        lambda_rows.append(
            {
                "lambda": float(risk_lambda),
                "validation_mean_precision": objective,
            }
        )
    lambda_table = pd.DataFrame(lambda_rows).sort_values(
        ["validation_mean_precision", "lambda"], ascending=[False, True]
    )
    selected_lambda = float(lambda_table.iloc[0]["lambda"])
    lambda_table.to_csv(output / "lambda_selection.csv", index=False)

    generator = np.random.default_rng(42)
    scores = {
        "random": generator.random(len(test)),
        "joint_probability": test["p_joint_calibrated"].to_numpy(float),
        "dynamic_probability": test["p_dynamic_calibrated"].to_numpy(float),
        "product_fusion": (
            test["p_thermo_calibrated"] * test["p_dynamic_calibrated"]
        ).to_numpy(float),
        "minimum_fusion": np.minimum(
            test["p_thermo_calibrated"], test["p_dynamic_calibrated"]
        ),
        "risk_aware": (
            test["p_joint_calibrated"].to_numpy(float)
            - selected_lambda * test_uncertainty
        ),
    }
    report_k = ranking_cfg["report_k"]
    all_rows, all_bootstrap, all_paired = evaluate_scope(
        test, scores, report_k, "all_test"
    )
    eligible_mask = test["y_thermo"].to_numpy(int) == 1
    eligible_frame = test.loc[eligible_mask].reset_index(drop=True)
    eligible_scores = {
        method: values[eligible_mask] for method, values in scores.items()
    }
    eligible_rows, eligible_bootstrap, eligible_paired = evaluate_scope(
        eligible_frame,
        eligible_scores,
        report_k,
        "thermo_positive_test",
    )
    table = pd.DataFrame(all_rows + eligible_rows)
    table.to_csv(output / "ranking_comparison.csv", index=False)
    pd.concat([all_paired, eligible_paired], ignore_index=True).to_csv(
        output / "paired_risk_vs_probability_bootstrap.csv", index=False
    )
    save_json(
        {
            "all_test": all_bootstrap,
            "thermo_positive_test": eligible_bootstrap,
        },
        output / "ranking_bootstrap_ci.json",
    )
    configuration = {
        "uncertainty_method": uncertainty_selection["method"],
        "uncertainty_score": uncertainty_selection["score"],
        "uncertainty_column": uncertainty_column,
        "uncertainty_q05": lower,
        "uncertainty_q95": upper,
        "lambda": selected_lambda,
        "lambda_selection_set": "validation thermo-positive subset",
        "lambda_selection_objective": (
            "mean Precision@K over configured validation K values"
        ),
    }
    save_json(configuration, output / "ranking_configuration.json")

    detailed = test.copy()
    detailed["uncertainty_normalized"] = test_uncertainty
    for method, values in scores.items():
        detailed[f"score_{method}"] = values
        detailed[f"rank_{method}"] = pd.Series(values).rank(
            method="first", ascending=False
        ).astype(int)
    detailed.to_csv(output / "test_ranking_predictions.csv", index=False)
    save_json({"status": "completed", **configuration}, output / "COMPLETED.json")
    print(lambda_table.to_string(index=False))
    print(table.to_string(index=False))
    print(configuration)


if __name__ == "__main__":
    main()
