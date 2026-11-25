from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def sigmoid(values):
    array = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -60.0, 60.0)))


def logit(probability, epsilon: float = 1e-7):
    probability = np.clip(
        np.asarray(probability, dtype=float),
        epsilon,
        1.0 - epsilon,
    )
    return np.log(probability / (1.0 - probability))


def binary_nll(y_true, probability, epsilon: float = 1e-7) -> float:
    labels = np.asarray(y_true, dtype=int)
    probability = np.clip(
        np.asarray(probability, dtype=float),
        epsilon,
        1.0 - epsilon,
    )
    return float(
        -np.mean(
            labels * np.log(probability)
            + (1 - labels) * np.log(1 - probability)
        )
    )


def expected_calibration_error(
    y_true,
    probability,
    n_bins: int = 15,
) -> float:
    labels = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for index, (lower, upper) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        if index == 0:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability > lower) & (probability <= upper)
        if not mask.any():
            continue
        empirical = labels[mask].mean()
        confidence = probability[mask].mean()
        result += mask.mean() * abs(empirical - confidence)
    return float(result)


def reliability_table(
    y_true,
    probability,
    n_bins: int = 15,
) -> pd.DataFrame:
    labels = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for index, (lower, upper) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        if index == 0:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability > lower) & (probability <= upper)
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "n": int(mask.sum()),
                "mean_confidence": float(probability[mask].mean())
                if mask.any()
                else math.nan,
                "empirical_positive_rate": float(labels[mask].mean())
                if mask.any()
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def best_mcc_threshold(y_true, probability) -> float:
    labels = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 199),
                probability,
                np.array([0.5]),
            ]
        )
    )
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        predictions = (probability >= threshold).astype(int)
        score = matthews_corrcoef(labels, predictions)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(
    y_true,
    probability,
    threshold: float = 0.5,
    n_bins: int = 15,
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predictions = (probability >= threshold).astype(int)
    result = {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(labels, probability)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "nll": binary_nll(labels, probability),
        "brier": float(brier_score_loss(labels, probability)),
        "ece": expected_calibration_error(labels, probability, n_bins),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels, probability))
        if np.unique(labels).size == 2
        else math.nan
    )
    return result


def risk_coverage_curve(
    y_true,
    probability,
    uncertainty,
    threshold: float,
) -> pd.DataFrame:
    labels = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    predictions = (probability >= threshold).astype(int)
    errors = (predictions != labels).astype(float)
    order = np.argsort(uncertainty, kind="mergesort")
    sorted_errors = errors[order]
    count = np.arange(1, len(errors) + 1)
    return pd.DataFrame(
        {
            "coverage": count / len(errors),
            "risk": np.cumsum(sorted_errors) / count,
            "retained_count": count,
        }
    )


def uncertainty_metrics(
    y_true,
    probability,
    uncertainty,
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    predictions = (probability >= threshold).astype(int)
    errors = (predictions != labels).astype(int)
    result: dict[str, float] = {}
    if np.unique(errors).size == 2:
        result["error_auroc"] = float(
            roc_auc_score(errors, uncertainty)
        )
        result["error_auprc"] = float(
            average_precision_score(errors, uncertainty)
        )
    else:
        result["error_auroc"] = math.nan
        result["error_auprc"] = math.nan

    curve = risk_coverage_curve(
        labels,
        probability,
        uncertainty,
        threshold,
    )
    # Discrete AURC is the mean selective risk over all attainable coverage
    # levels. This is the standard finite-sample estimator used for selective
    # prediction comparisons.
    aurc = float(curve["risk"].mean())

    oracle_errors = np.sort(errors)
    count = np.arange(1, len(errors) + 1)
    oracle_risk = np.cumsum(oracle_errors) / count
    oracle_aurc = float(np.mean(oracle_risk))
    result["aurc"] = aurc
    result["eaurc"] = aurc - oracle_aurc
    for coverage_level in (0.9, 0.8):
        retained = max(1, int(np.floor(coverage_level * len(errors))))
        order = np.argsort(uncertainty, kind="mergesort")
        result[f"risk_at_{int(coverage_level * 100)}_coverage"] = float(
            errors[order[:retained]].mean()
        )
    correct = errors == 0
    wrong = errors == 1
    result["correct_uncertainty_median"] = (
        float(np.median(uncertainty[correct])) if correct.any() else math.nan
    )
    result["error_uncertainty_median"] = (
        float(np.median(uncertainty[wrong])) if wrong.any() else math.nan
    )
    absolute_probability_error = np.abs(labels - probability)
    correlation = spearmanr(
        uncertainty,
        absolute_probability_error,
        nan_policy="omit",
    )
    result["spearman_uncertainty_abs_probability_error"] = float(
        correlation.statistic
    )
    result["spearman_pvalue"] = float(correlation.pvalue)
    return result


def precision_at_k(y_true, score, k: int) -> float:
    labels = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    k = min(int(k), len(labels))
    if k <= 0:
        return math.nan
    indices = np.argsort(-score, kind="mergesort")[:k]
    return float(labels[indices].mean())


def ranking_metrics(y_true, score, k_values) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    result = {}
    base_rate = float(labels.mean())
    for k in k_values:
        k_effective = min(int(k), len(labels))
        precision = precision_at_k(labels, score, k_effective)
        result[f"precision_at_{k}"] = precision
        result[f"yield_at_{k}"] = precision * k_effective
        result[f"enrichment_at_{k}"] = (
            precision / base_rate if base_rate > 0 else math.nan
        )
    return result
