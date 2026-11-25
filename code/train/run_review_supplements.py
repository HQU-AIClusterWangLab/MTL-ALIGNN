"""Reproducible, no-retraining analyses requested during peer review.

This module consumes only frozen prediction CSVs and processed tabular data.
It deliberately does not import the training dataset or unknown-screening entry
points, so running it cannot mutate model checkpoints or regenerate labels.
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
from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train.metrics import best_mcc_threshold, binary_nll, sigmoid


COMMON_SEEDS = (2025, 2026, 2027)
TASKS = ("thermo", "dynamic", "joint")
EHULL_THRESHOLDS = (0.05, 0.075, 0.10, 0.125, 0.15)
TOP_K_VALUES = (10, 20, 50, 100)
TOLERANCE = 1.0e-10


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _entropy(probability: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _logit(probability: np.ndarray, epsilon: float = 1.0e-7) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(p / (1.0 - p))


def _safe_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path | str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{source} missing required columns: {missing}")


def read_prediction(path: Path, required: Iterable[str] = ()) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(frame, {"material_id", *required}, path)
    if frame["material_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate material_id values in {path}")
    return frame.sort_values("material_id", kind="mergesort").reset_index(drop=True)


def align_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_ids = left["material_id"].astype(str).tolist()
    right_ids = right["material_id"].astype(str).tolist()
    if left_ids != right_ids:
        left_only = sorted(set(left_ids) - set(right_ids))[:5]
        right_only = sorted(set(right_ids) - set(left_ids))[:5]
        raise ValueError(
            f"material_id mismatch for {left_name} vs {right_name}; "
            f"left_only={left_only}, right_only={right_only}"
        )
    if "y_joint" in left and "y_joint" in right:
        if not np.array_equal(left["y_joint"].to_numpy(int), right["y_joint"].to_numpy(int)):
            raise ValueError(f"y_joint mismatch for {left_name} vs {right_name}")
    return left, right


def load_seed_frames(
    experiment_dir: Path,
    seeds: Sequence[int],
    split: str,
    required: Iterable[str] = ("y_joint", "p_joint"),
) -> tuple[list[pd.DataFrame], list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for seed in seeds:
        path = experiment_dir / f"seed_{seed}" / f"{split}_predictions.csv"
        frames.append(read_prediction(path, required))
        paths.append(path)
    for index in range(1, len(frames)):
        align_pair(frames[0], frames[index], str(paths[0]), str(paths[index]))
    return frames, paths


def aggregate_members(
    frames: Sequence[pd.DataFrame],
    temperatures: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if not frames:
        raise ValueError("At least one frozen prediction frame is required")
    metadata = [
        "material_id",
        "row_id",
        "formula",
        "reduced_formula",
        "chemical_system",
        "ehull",
        "spacegroup_symbol",
        "spacegroup_number",
        "layergroup",
        "structure_family",
        "y_thermo",
        "y_dynamic",
        "y_joint",
    ]
    result = frames[0][[column for column in metadata if column in frames[0]]].copy()
    logits_by_task: dict[str, np.ndarray] = {}
    for task in TASKS:
        probability_column = f"p_{task}"
        if not all(probability_column in frame for frame in frames):
            continue
        probability_matrix = np.vstack(
            [frame[probability_column].to_numpy(float) for frame in frames]
        )
        logit_matrix = np.vstack(
            [
                frame[f"logit_{task}"].to_numpy(float)
                if f"logit_{task}" in frame
                else _logit(frame[probability_column].to_numpy(float))
                for frame in frames
            ]
        )
        logits_by_task[task] = logit_matrix
        mean = probability_matrix.mean(axis=0)
        result[f"p_{task}_mean"] = mean
        result[f"p_{task}_std"] = probability_matrix.std(axis=0, ddof=0)
        result[f"{task}_predictive_entropy"] = _entropy(mean)
        result[f"{task}_expected_entropy"] = _entropy(probability_matrix).mean(axis=0)
        result[f"{task}_mutual_information"] = np.maximum(
            result[f"{task}_predictive_entropy"]
            - result[f"{task}_expected_entropy"],
            0.0,
        )
        if temperatures and task in temperatures:
            calibrated_matrix = sigmoid(logit_matrix / float(temperatures[task]))
            calibrated_mean = calibrated_matrix.mean(axis=0)
            result[f"p_{task}_calibrated"] = calibrated_mean
            result[f"p_{task}_calibrated_std"] = calibrated_matrix.std(axis=0, ddof=0)
            result[f"{task}_calibrated_predictive_entropy"] = _entropy(calibrated_mean)
            result[f"{task}_calibrated_expected_entropy"] = _entropy(
                calibrated_matrix
            ).mean(axis=0)
            result[f"{task}_calibrated_mutual_information"] = np.maximum(
                result[f"{task}_calibrated_predictive_entropy"]
                - result[f"{task}_calibrated_expected_entropy"],
                0.0,
            )
    return result, logits_by_task


def fit_temperature(labels: Iterable[int], logits_matrix: np.ndarray) -> float:
    """Fit one shared positive temperature using validation labels only."""
    y = np.asarray(labels, dtype=int)
    logits = np.asarray(logits_matrix, dtype=float)
    if logits.ndim == 1:
        logits = logits[None, :]
    if logits.shape[1] != len(y):
        raise ValueError("Logit matrix and labels are not aligned")

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probability = sigmoid(logits / temperature).mean(axis=0)
        return binary_nll(y, probability)

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature fit failed: {result.message}")
    return float(np.exp(result.x))


def paired_bootstrap_difference(
    labels: Iterable[int],
    full_probability: Iterable[float],
    comparator_probability: Iterable[float],
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired material-level bootstrap for Full minus comparator."""
    y = np.asarray(labels, dtype=int)
    full = np.asarray(full_probability, dtype=float)
    other = np.asarray(comparator_probability, dtype=float)
    if not (len(y) == len(full) == len(other)):
        raise ValueError("labels and paired predictions must have equal length")
    scorers = {
        "pr_auc": average_precision_score,
        "roc_auc": roc_auc_score,
    }
    if metric not in scorers:
        raise ValueError(f"Unsupported metric: {metric}")
    scorer = scorers[metric]
    full_point = float(scorer(y, full))
    comparator_point = float(scorer(y, other))
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(int(replicates)):
        indices = rng.integers(0, len(y), size=len(y))
        sampled_y = y[indices]
        if np.unique(sampled_y).size < 2:
            continue
        differences.append(
            float(scorer(sampled_y, full[indices]) - scorer(sampled_y, other[indices]))
        )
    if not differences:
        raise ValueError("No valid bootstrap replicate contained both classes")
    values = np.asarray(differences, dtype=float)
    return {
        "full_metric": full_point,
        "comparator_metric": comparator_point,
        "difference_full_minus_comparator": full_point - comparator_point,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "bootstrap_mean_difference": float(values.mean()),
        "probability_difference_gt_zero": float((values > 0).mean()),
        "probability_difference_lt_zero": float((values < 0).mean()),
        "two_sided_bootstrap_p": float(min(1.0, 2.0 * min(
            (np.count_nonzero(values <= 0) + 1) / (len(values) + 1),
            (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)))),
        "bootstrap_standard_error": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "bootstrap_replicates_requested": int(replicates),
        "bootstrap_replicates_valid": int(len(values)),
    }


def logic_violation_magnitudes(
    p_thermo: Iterable[float],
    p_dynamic: Iterable[float],
    p_joint: Iterable[float],
) -> dict[str, np.ndarray]:
    thermo = np.asarray(p_thermo, dtype=float)
    dynamic = np.asarray(p_dynamic, dtype=float)
    joint = np.asarray(p_joint, dtype=float)
    if not (len(thermo) == len(dynamic) == len(joint)):
        raise ValueError("Logical-consistency probabilities must be aligned")
    v1 = np.maximum(joint - thermo, 0.0)
    v2 = np.maximum(joint - dynamic, 0.0)
    v3 = np.maximum(thermo + dynamic - 1.0 - joint, 0.0)
    return {
        "V1_pJ_gt_pT": v1,
        "V2_pJ_gt_pD": v2,
        "V3_pJ_below_Frechet": v3,
        "ANY_max_magnitude": np.maximum.reduce([v1, v2, v3]),
    }


def summarize_logic_violations(
    model: str,
    probability_scale: str,
    p_thermo: Iterable[float],
    p_dynamic: Iterable[float],
    p_joint: Iterable[float],
    tolerance: float = TOLERANCE,
) -> pd.DataFrame:
    magnitudes = logic_violation_magnitudes(p_thermo, p_dynamic, p_joint)
    rows = []
    for name, magnitude in magnitudes.items():
        mask = magnitude > tolerance
        rows.append(
            {
                "status": "ok",
                "model": model,
                "probability_scale": probability_scale,
                "violation": name,
                "definition": {
                    "V1_pJ_gt_pT": "max(pJ-pT,0)",
                    "V2_pJ_gt_pD": "max(pJ-pD,0)",
                    "V3_pJ_below_Frechet": "max(pT+pD-1-pJ,0)",
                    "ANY_max_magnitude": "max(V1,V2,V3)",
                }[name],
                "n": int(len(magnitude)),
                "violation_count": int(mask.sum()),
                "violation_rate": float(mask.mean()),
                "mean_magnitude_all": float(magnitude.mean()),
                "mean_magnitude_violating": float(magnitude[mask].mean())
                if mask.any()
                else 0.0,
                "p95_magnitude_all": float(np.quantile(magnitude, 0.95)),
                "max_magnitude": float(magnitude.max(initial=0.0)),
                "tolerance": float(tolerance),
            }
        )
    return pd.DataFrame(rows)


def fixed_bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(n_bins) + 1)


def frozen_quantile_edges(
    values: Iterable[float],
    n_bins: int,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> np.ndarray:
    array = _finite(values)
    if len(array) == 0:
        raise ValueError("Cannot derive quantile bins from no finite values")
    edges = np.quantile(array, np.linspace(0.0, 1.0, int(n_bins) + 1))
    # Preserve the requested bin count even for ties; searchsorted handles empty bins.
    edges[0] = (
        float(lower_bound)
        if lower_bound is not None
        else min(edges[0], float(array.min()))
    )
    edges[-1] = (
        float(upper_bound)
        if upper_bound is not None
        else max(edges[-1], float(array.max()))
    )
    if edges[0] > float(array.min()) or edges[-1] < float(array.max()):
        raise ValueError("Frozen outer bin bounds do not cover validation values")
    return edges.astype(float)


def reliability_rows(
    labels: Iterable[int],
    probability: Iterable[float],
    edges: Sequence[float],
) -> pd.DataFrame:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probability, dtype=float)
    boundaries = np.asarray(edges, dtype=float)
    if len(boundaries) < 2:
        raise ValueError("At least two reliability boundaries are required")
    rows = []
    for index in range(len(boundaries) - 1):
        lower, upper = float(boundaries[index]), float(boundaries[index + 1])
        if index == len(boundaries) - 2:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        count = int(mask.sum())
        mean_probability = float(p[mask].mean()) if count else math.nan
        observed = float(y[mask].mean()) if count else math.nan
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "fraction": count / len(y) if len(y) else math.nan,
                "mean_probability": mean_probability,
                "observed_positive_rate": observed,
                "absolute_gap": abs(mean_probability - observed) if count else math.nan,
            }
        )
    return pd.DataFrame(rows)


def ece_from_reliability(table: pd.DataFrame) -> float:
    valid = table["count"] > 0
    if not valid.any():
        return math.nan
    weights = table.loc[valid, "fraction"].to_numpy(float)
    gaps = table.loc[valid, "absolute_gap"].to_numpy(float)
    return float(np.sum(weights * gaps))


def classwise_ece(labels: Iterable[int], probability: Iterable[float], n_bins: int) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probability, dtype=float)
    edges = fixed_bin_edges(n_bins)
    positive = ece_from_reliability(reliability_rows(y, p, edges))
    negative = ece_from_reliability(reliability_rows(1 - y, 1.0 - p, edges))
    return float(np.nanmean([positive, negative]))


def calibration_metrics(
    labels: Iterable[int],
    probability: Iterable[float],
    threshold: float,
    n_bins: int,
    adaptive_edges: Sequence[float],
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    fixed = reliability_rows(y, p, fixed_bin_edges(n_bins))
    adaptive = reliability_rows(y, p, adaptive_edges)
    return {
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "nll": binary_nll(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece_from_reliability(fixed),
        "classwise_ece": classwise_ece(y, p, n_bins),
        "adaptive_ece_frozen_validation_bins": ece_from_reliability(adaptive),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else math.nan,
        "error_count": int((pred != y).sum()),
        "error_rate": float((pred != y).mean()),
        "mcc": float(matthews_corrcoef(y, pred)),
    }


def probability_uncertainty_table(
    labels: Iterable[int],
    probability: Iterable[float],
    uncertainty: Iterable[float],
    threshold: float,
    uncertainty_edges: Sequence[float],
    probability_edges: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> pd.DataFrame:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probability, dtype=float)
    u = np.asarray(uncertainty, dtype=float)
    if not (len(y) == len(p) == len(u)):
        raise ValueError("labels, probability, and uncertainty must be aligned")
    p_edges = np.asarray(probability_edges, dtype=float)
    u_edges = np.asarray(uncertainty_edges, dtype=float)
    predictions = (p >= threshold).astype(int)
    rows = []
    for p_index in range(len(p_edges) - 1):
        p_lower, p_upper = p_edges[p_index], p_edges[p_index + 1]
        p_mask = (p >= p_lower) & (p <= p_upper) if p_index == len(p_edges) - 2 else (p >= p_lower) & (p < p_upper)
        for u_index in range(len(u_edges) - 1):
            u_lower, u_upper = u_edges[u_index], u_edges[u_index + 1]
            u_mask = (u >= u_lower) & (u <= u_upper) if u_index == len(u_edges) - 2 else (u >= u_lower) & (u < u_upper)
            mask = p_mask & u_mask
            count = int(mask.sum())
            errors = (predictions[mask] != y[mask]) if count else np.array([], dtype=bool)
            rows.append(
                {
                    "probability_bin": p_index,
                    "probability_lower": float(p_lower),
                    "probability_upper": float(p_upper),
                    "uncertainty_bin": u_index,
                    "uncertainty_lower": float(u_lower),
                    "uncertainty_upper": float(u_upper),
                    "count": count,
                    "positive_count": int(y[mask].sum()) if count else 0,
                    "error_count": int(errors.sum()) if count else 0,
                    "error_rate": float(errors.mean()) if count else math.nan,
                    "false_positive_count": int(((predictions[mask] == 1) & (y[mask] == 0)).sum()) if count else 0,
                    "false_negative_count": int(((predictions[mask] == 0) & (y[mask] == 1)).sum()) if count else 0,
                    "mean_probability": float(p[mask].mean()) if count else math.nan,
                    "mean_uncertainty": float(u[mask].mean()) if count else math.nan,
                }
            )
    return pd.DataFrame(rows)


def quadrant_counts(ehull: Iterable[float], dynamic: Iterable[int], threshold: float) -> dict[str, float | int]:
    energy = np.asarray(ehull, dtype=float)
    dynamic_label = np.asarray(dynamic, dtype=int)
    valid = np.isfinite(energy) & np.isin(dynamic_label, [0, 1])
    thermo_positive = energy[valid] <= float(threshold)
    dynamic_positive = dynamic_label[valid] == 1
    t_plus_d_plus = int((thermo_positive & dynamic_positive).sum())
    t_plus_d_minus = int((thermo_positive & ~dynamic_positive).sum())
    t_minus_d_plus = int((~thermo_positive & dynamic_positive).sum())
    t_minus_d_minus = int((~thermo_positive & ~dynamic_positive).sum())
    thermo_count = t_plus_d_plus + t_plus_d_minus
    return {
        "labeled_valid_count": int(valid.sum()),
        "Tplus_Dplus": t_plus_d_plus,
        "Tplus_Dminus": t_plus_d_minus,
        "Tminus_Dplus": t_minus_d_plus,
        "Tminus_Dminus": t_minus_d_minus,
        "thermo_positive_count": thermo_count,
        "dynamic_positive_count": int(dynamic_positive.sum()),
        "joint_positive_count": t_plus_d_plus,
        "p_Dplus_given_Tplus": t_plus_d_plus / thermo_count if thermo_count else math.nan,
    }


def run_bootstrap_experiment(
    project_root: Path,
    replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, list[Path]]:
    full_dir = project_root / "results/main/M06_full_mtl"
    comparators = {
        "M04_single_alignn": project_root / "results/main/M04_single_alignn",
        "M05_equal_mtl": project_root / "results/main/M05_equal_mtl",
        "A03_no_logic": project_root / "results/ablation/A03_no_logic",
    }
    rows = []
    sources: list[Path] = []
    for comparator_index, (name, comparator_dir) in enumerate(comparators.items()):
        full_frames, full_paths = load_seed_frames(full_dir, COMMON_SEEDS, "test")
        other_frames, other_paths = load_seed_frames(comparator_dir, COMMON_SEEDS, "test")
        sources.extend(full_paths + other_paths)
        for seed_index, seed in enumerate(COMMON_SEEDS):
            full, other = align_pair(full_frames[seed_index], other_frames[seed_index], "Full", name)
            for metric_index, metric in enumerate(("pr_auc", "roc_auc")):
                result = paired_bootstrap_difference(
                    full["y_joint"],
                    full["p_joint"],
                    other["p_joint"],
                    metric,
                    replicates,
                    bootstrap_seed + 1000 * comparator_index + 10 * seed_index + metric_index,
                )
                rows.append(
                    {
                        "status": "ok",
                        "comparison": f"M06_full_mtl minus {name}",
                        "comparator": name,
                        "scope": "paired_same_seed",
                        "seed": seed,
                        "ensemble_members": 1,
                        "metric": metric,
                        "n_material_ids": len(full),
                        "bootstrap_unit": "material_id",
                        "bootstrap_seed": bootstrap_seed + 1000 * comparator_index + 10 * seed_index + metric_index,
                        **result,
                    }
                )
        full_ensemble = np.vstack([frame["p_joint"].to_numpy(float) for frame in full_frames]).mean(axis=0)
        other_ensemble = np.vstack([frame["p_joint"].to_numpy(float) for frame in other_frames]).mean(axis=0)
        for metric_index, metric in enumerate(("pr_auc", "roc_auc")):
            result = paired_bootstrap_difference(
                full_frames[0]["y_joint"],
                full_ensemble,
                other_ensemble,
                metric,
                replicates,
                bootstrap_seed + 10000 + 1000 * comparator_index + metric_index,
            )
            rows.append(
                {
                    "status": "ok",
                    "comparison": f"M06_full_mtl minus {name}",
                    "comparator": name,
                    "scope": "common_seed_probability_mean_ensemble",
                    "seed": "2025,2026,2027",
                    "ensemble_members": 3,
                    "metric": metric,
                    "n_material_ids": len(full_frames[0]),
                    "bootstrap_unit": "material_id",
                    "bootstrap_seed": bootstrap_seed + 10000 + 1000 * comparator_index + metric_index,
                    **result,
                }
            )
    return pd.DataFrame(rows), sorted(set(sources))


def run_logic_experiment(project_root: Path) -> tuple[pd.DataFrame, list[Path], dict]:
    roots = {
        "M06_full_mtl": project_root / "results/main/M06_full_mtl",
        "A03_no_logic": project_root / "results/ablation/A03_no_logic",
    }
    frames = []
    sources: list[Path] = []
    fitted: dict[str, dict[str, float]] = {}
    for name, root in roots.items():
        val_members, val_paths = load_seed_frames(root, COMMON_SEEDS, "val", required=("y_joint", "p_joint"))
        test_members, test_paths = load_seed_frames(root, COMMON_SEEDS, "test", required=("y_joint", "p_joint"))
        sources.extend(val_paths + test_paths)
        raw_val, val_logits = aggregate_members(val_members)
        missing_tasks = sorted(set(TASKS) - set(val_logits))
        if missing_tasks:
            frames.append(
                pd.DataFrame(
                    [{"status": "missing_fields", "model": name, "missing_tasks": ",".join(missing_tasks)}]
                )
            )
            continue
        temperatures = {
            task: fit_temperature(raw_val[f"y_{task}"], val_logits[task])
            for task in TASKS
        }
        fitted[name] = temperatures
        test, _ = aggregate_members(test_members, temperatures)
        for scale, suffix in (("raw", "mean"), ("calibrated", "calibrated")):
            summary = summarize_logic_violations(
                name,
                scale,
                test[f"p_thermo_{suffix}"],
                test[f"p_dynamic_{suffix}"],
                test[f"p_joint_{suffix}"],
            )
            summary["seeds"] = ",".join(map(str, COMMON_SEEDS))
            summary["ensemble_members"] = len(COMMON_SEEDS)
            summary["calibration_fit_source"] = (
                f"{root.relative_to(project_root)}/seed_*/val_predictions.csv"
                if scale == "calibrated"
                else "not_applicable_raw"
            )
            summary["temperature_thermo"] = temperatures["thermo"] if scale == "calibrated" else 1.0
            summary["temperature_dynamic"] = temperatures["dynamic"] if scale == "calibrated" else 1.0
            summary["temperature_joint"] = temperatures["joint"] if scale == "calibrated" else 1.0
            frames.append(summary)
    return pd.concat(frames, ignore_index=True, sort=False), sorted(set(sources)), fitted


def build_frozen_ood_evaluations(
    project_root: Path,
    n_bins: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    list[Path],
    dict[str, pd.DataFrame],
]:
    roots = {
        "IID": project_root / "results/main/M06_full_mtl",
        "chemical_OOD": project_root / "results/optional/O01_chemical_system_ood",
        "structure_OOD": project_root / "results/optional/O02_structure_family_ood",
    }
    protocol = (
        "scenario-specific development-validation-to-held-out-test "
        "calibration transfer"
    )
    sources: list[Path] = []
    calibration_rows = []
    reliability_frames = []
    taxonomy_frames = []
    pu_frames = []
    detailed_by_eval: dict[str, pd.DataFrame] = {}
    evaluation_counts = {}
    scenario_parameters: dict[str, dict] = {}
    for evaluation, root in roots.items():
        val_members, val_paths = load_seed_frames(root, COMMON_SEEDS, "val")
        test_members, test_paths = load_seed_frames(root, COMMON_SEEDS, "test")
        sources.extend(val_paths + test_paths)
        raw_val, val_logits = aggregate_members(val_members)
        temperature = fit_temperature(raw_val["y_joint"], val_logits["joint"])
        validation, _ = aggregate_members(val_members, {"joint": temperature})
        raw_threshold = best_mcc_threshold(
            validation["y_joint"], validation["p_joint_mean"]
        )
        calibrated_threshold = best_mcc_threshold(
            validation["y_joint"], validation["p_joint_calibrated"]
        )
        high_confidence_margin = float(
            np.quantile(
                np.abs(
                    validation["p_joint_calibrated"].to_numpy(float)
                    - calibrated_threshold
                ),
                0.75,
            )
        )
        raw_adaptive_edges = frozen_quantile_edges(
            validation["p_joint_mean"], n_bins, lower_bound=0.0, upper_bound=1.0
        )
        calibrated_adaptive_edges = frozen_quantile_edges(
            validation["p_joint_calibrated"],
            n_bins,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        uncertainty_edges = frozen_quantile_edges(
            validation["p_joint_calibrated_std"],
            5,
            lower_bound=0.0,
            upper_bound=0.5,
        )
        fit_source = "|".join(
            path.relative_to(project_root).as_posix() for path in val_paths
        )
        run_directory = root.relative_to(project_root).as_posix()
        held_out_role = (
            "IID test"
            if evaluation == "IID"
            else f"held-out {evaluation.replace('_', ' ')} test"
        )
        test, _ = aggregate_members(test_members, {"joint": temperature})
        detailed_by_eval[evaluation] = test
        evaluation_counts[evaluation] = int(len(test))
        scenario_parameters[evaluation] = {
            "run_directory": run_directory,
            "seeds": list(COMMON_SEEDS),
            "ensemble_members": len(COMMON_SEEDS),
            "development_validation_count": int(len(validation)),
            "held_out_test_count": int(len(test)),
            "held_out_test_role": held_out_role,
            "temperature_joint": temperature,
            "raw_decision_threshold_joint": raw_threshold,
            "calibrated_decision_threshold_joint": calibrated_threshold,
            "high_confidence_definition": (
                "abs(pJ_calibrated - scenario-frozen decision threshold) >= "
                "scenario development-validation 75th-percentile margin"
            ),
            "high_confidence_margin_quantile": 0.75,
            "high_confidence_margin_cutoff": high_confidence_margin,
            "adaptive_reliability_edges_raw": raw_adaptive_edges.tolist(),
            "adaptive_reliability_edges_calibrated": calibrated_adaptive_edges.tolist(),
            "uncertainty_quintile_edges_calibrated_std": uncertainty_edges.tolist(),
            "fit_source": [
                path.relative_to(project_root).as_posix() for path in val_paths
            ],
            "test_or_OOD_labels_used_for_fit": False,
        }
        for scale, probability_column, threshold, adaptive_edges in (
            ("raw", "p_joint_mean", raw_threshold, raw_adaptive_edges),
            (
                "calibrated_with_scenario_validation_temperature",
                "p_joint_calibrated",
                calibrated_threshold,
                calibrated_adaptive_edges,
            ),
        ):
            metrics = calibration_metrics(
                test["y_joint"], test[probability_column], threshold, n_bins, adaptive_edges
            )
            calibration_rows.append(
                {
                    "status": "ok",
                    "evaluation": evaluation,
                    "scenario_run_directory": run_directory,
                    "held_out_test_role": held_out_role,
                    "calibration_protocol": protocol,
                    "probability_scale": scale,
                    "temperature": 1.0 if scale == "raw" else temperature,
                    "fitted_temperature_joint": temperature,
                    "temperature_fit_domain": (
                        "scenario_development_validation_only"
                        if scale != "raw"
                        else "not_applicable_raw"
                    ),
                    "threshold_fit_domain": "scenario_development_validation_only",
                    "development_validation_fit_source": fit_source,
                    "development_validation_count": len(validation),
                    "high_confidence_margin_cutoff": high_confidence_margin,
                    "seeds": ",".join(map(str, COMMON_SEEDS)),
                    "ensemble_members": len(COMMON_SEEDS),
                    **metrics,
                }
            )
            for scheme, edges in (
                ("equal_width", fixed_bin_edges(n_bins)),
                ("adaptive_frozen_scenario_validation_quantiles", adaptive_edges),
            ):
                table = reliability_rows(test["y_joint"], test[probability_column], edges)
                table.insert(0, "status", "ok")
                table.insert(1, "evaluation", evaluation)
                table.insert(2, "probability_scale", scale)
                table.insert(3, "binning_scheme", scheme)
                table["bin_edges_source"] = (
                    "fixed_[0,1]"
                    if scheme == "equal_width"
                    else "scenario_development_validation_only"
                )
                table["calibration_protocol"] = protocol
                table["scenario_run_directory"] = run_directory
                table["held_out_test_role"] = held_out_role
                table["development_validation_fit_source"] = fit_source
                table["development_validation_count"] = len(validation)
                table["temperature"] = 1.0 if scale == "raw" else temperature
                table["decision_threshold"] = threshold
                table["high_confidence_margin_cutoff"] = high_confidence_margin
                reliability_frames.append(table)

        prediction = (test["p_joint_calibrated"].to_numpy(float) >= calibrated_threshold).astype(int)
        labels = test["y_joint"].to_numpy(int)
        margin = np.abs(test["p_joint_calibrated"].to_numpy(float) - calibrated_threshold)
        error_mask = prediction != labels
        taxonomy = test.loc[error_mask].copy()
        taxonomy["status"] = "ok"
        taxonomy["evaluation"] = evaluation
        taxonomy["prediction"] = prediction[error_mask]
        taxonomy["error_type"] = np.where(
            (prediction[error_mask] == 1) & (labels[error_mask] == 0), "FP", "FN"
        )
        taxonomy["probability_margin_from_frozen_threshold"] = margin[error_mask]
        taxonomy["high_confidence_margin_cutoff"] = high_confidence_margin
        taxonomy["is_high_confidence_error"] = margin[error_mask] >= high_confidence_margin
        taxonomy["decision_threshold"] = calibrated_threshold
        taxonomy["temperature"] = temperature
        taxonomy["calibration_protocol"] = protocol
        taxonomy["scenario_run_directory"] = run_directory
        taxonomy["held_out_test_role"] = held_out_role
        taxonomy["development_validation_count"] = len(validation)
        taxonomy["development_validation_fit_source"] = fit_source
        taxonomy["freeze_source"] = "same-scenario common-seed development validation only"
        taxonomy["embedding_distance_status"] = "not_available_config_save_embeddings_false"
        desired = [
            "status",
            "evaluation",
            "material_id",
            "row_id",
            "formula",
            "chemical_system",
            "structure_family",
            "layergroup",
            "ehull",
            "y_thermo",
            "y_dynamic",
            "y_joint",
            "prediction",
            "error_type",
            "p_joint_mean",
            "p_joint_calibrated",
            "p_joint_std",
            "p_joint_calibrated_std",
            "joint_predictive_entropy",
            "joint_mutual_information",
            "joint_calibrated_predictive_entropy",
            "joint_calibrated_mutual_information",
            "probability_margin_from_frozen_threshold",
            "high_confidence_margin_cutoff",
            "is_high_confidence_error",
            "decision_threshold",
            "temperature",
            "calibration_protocol",
            "scenario_run_directory",
            "held_out_test_role",
            "development_validation_count",
            "development_validation_fit_source",
            "freeze_source",
            "embedding_distance_status",
        ]
        taxonomy_frames.append(taxonomy[[column for column in desired if column in taxonomy]])

        pu = probability_uncertainty_table(
            labels,
            test["p_joint_calibrated"],
            test["p_joint_calibrated_std"],
            calibrated_threshold,
            uncertainty_edges,
        )
        pu.insert(0, "status", "ok")
        pu.insert(1, "evaluation", evaluation)
        pu["probability_scale"] = "calibrated_with_scenario_validation_temperature"
        pu["uncertainty_measure"] = "std_across_3_temperature_scaled_member_probabilities"
        pu["uncertainty_bin_edges_source"] = "scenario_development_validation_only"
        pu["calibration_protocol"] = protocol
        pu["scenario_run_directory"] = run_directory
        pu["held_out_test_role"] = held_out_role
        pu["development_validation_fit_source"] = fit_source
        pu["development_validation_count"] = len(validation)
        pu["temperature"] = temperature
        pu["decision_threshold"] = calibrated_threshold
        pu["high_confidence_margin_cutoff"] = high_confidence_margin
        pu_frames.append(pu)

    calibration_protocol = {
        "protocol": protocol,
        "shared_seeds": list(COMMON_SEEDS),
        "ensemble_members_per_scenario": len(COMMON_SEEDS),
        "scenario_parameters": scenario_parameters,
        "scenario_validation_fit_performed": True,
        "test_or_OOD_refit_performed": False,
        "evaluation_counts": evaluation_counts,
        "interpretation": (
            "Each frozen training scenario fits its own temperature, raw/calibrated "
            "decision thresholds, high-confidence margin, and reporting bins on "
            "that scenario's non-test development-validation ensemble. Those "
            "parameters are frozen and applied only to the matching held-out test "
            "set. No test or OOD labels are used for fitting, and no single "
            "temperature is claimed to transfer across differently trained models."
        ),
    }
    return (
        pd.DataFrame(calibration_rows),
        pd.concat(reliability_frames, ignore_index=True, sort=False),
        pd.concat(taxonomy_frames, ignore_index=True, sort=False),
        pd.concat(pu_frames, ignore_index=True, sort=False),
        calibration_protocol,
        sorted(set(sources)),
        detailed_by_eval,
    )


def continuous_shift_rows(
    known: pd.DataFrame,
    target: pd.DataFrame,
    target_name: str,
    columns: Sequence[str],
) -> list[dict]:
    rows = []
    for column in columns:
        if column not in known or column not in target:
            rows.append(
                {
                    "status": "missing_field",
                    "record_type": "continuous_shift",
                    "comparison": f"labeled_vs_{target_name}",
                    "feature": column,
                }
            )
            continue
        a = _finite(pd.to_numeric(known[column], errors="coerce"))
        b = _finite(pd.to_numeric(target[column], errors="coerce"))
        if len(a) < 2 or len(b) < 2:
            rows.append(
                {
                    "status": "insufficient_finite_values",
                    "record_type": "continuous_shift",
                    "comparison": f"labeled_vs_{target_name}",
                    "feature": column,
                    "known_n": len(a),
                    "target_n": len(b),
                }
            )
            continue
        pooled_sd = math.sqrt((float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2.0)
        iqr = float(np.quantile(a, 0.75) - np.quantile(a, 0.25))
        ks = ks_2samp(a, b, alternative="two-sided", method="auto")
        wd = float(wasserstein_distance(a, b))
        rows.append(
            {
                "status": "ok",
                "record_type": "continuous_shift",
                "comparison": f"labeled_vs_{target_name}",
                "feature": column,
                "known_n": len(a),
                "target_n": len(b),
                "known_mean": float(a.mean()),
                "target_mean": float(b.mean()),
                "known_std": float(a.std(ddof=1)),
                "target_std": float(b.std(ddof=1)),
                "standardized_mean_difference_target_minus_known": (float(b.mean() - a.mean()) / pooled_sd) if pooled_sd > 0 else 0.0,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein_distance_raw_units": wd,
                "wasserstein_distance_over_known_IQR": wd / iqr if iqr > 0 else math.nan,
            }
        )
    return rows


def categorical_shift_rows(
    known: pd.DataFrame,
    target: pd.DataFrame,
    target_name: str,
    columns: Sequence[str],
    top_levels: int = 20,
) -> list[dict]:
    rows = []
    for column in columns:
        if column not in known or column not in target:
            rows.append(
                {
                    "status": "missing_field",
                    "record_type": "categorical_shift",
                    "comparison": f"labeled_vs_{target_name}",
                    "feature": column,
                }
            )
            continue
        a = known[column].fillna("<MISSING>").astype(str)
        b = target[column].fillna("<MISSING>").astype(str)
        a_frequency = a.value_counts(normalize=True)
        b_frequency = b.value_counts(normalize=True)
        support = sorted(set(a_frequency.index) | set(b_frequency.index))
        pa = np.asarray([a_frequency.get(level, 0.0) for level in support], dtype=float)
        pb = np.asarray([b_frequency.get(level, 0.0) for level in support], dtype=float)
        midpoint = 0.5 * (pa + pb)
        positive_a = pa > 0
        positive_b = pb > 0
        kl_a = np.sum(
            pa[positive_a] * np.log(pa[positive_a] / midpoint[positive_a])
        )
        kl_b = np.sum(
            pb[positive_b] * np.log(pb[positive_b] / midpoint[positive_b])
        )
        total_variation = 0.5 * float(np.abs(pa - pb).sum())
        rows.append(
            {
                "status": "ok",
                "record_type": "categorical_summary",
                "comparison": f"labeled_vs_{target_name}",
                "feature": column,
                "known_n": len(a),
                "target_n": len(b),
                "category_support_union": len(support),
                "total_variation_distance": total_variation,
                "jensen_shannon_divergence_nats": float(0.5 * (kl_a + kl_b)),
            }
        )
        ranked = sorted(support, key=lambda level: max(a_frequency.get(level, 0.0), b_frequency.get(level, 0.0)), reverse=True)
        kept = ranked[:top_levels]
        for level in kept:
            af = float(a_frequency.get(level, 0.0))
            bf = float(b_frequency.get(level, 0.0))
            rows.append(
                {
                    "status": "ok",
                    "record_type": "categorical_frequency",
                    "comparison": f"labeled_vs_{target_name}",
                    "feature": column,
                    "level": level,
                    "known_n": len(a),
                    "target_n": len(b),
                    "known_fraction": af,
                    "target_fraction": bf,
                    "fraction_difference_target_minus_known": bf - af,
                }
            )
        omitted = set(ranked[top_levels:])
        if omitted:
            af = float(a.isin(omitted).mean())
            bf = float(b.isin(omitted).mean())
            rows.append(
                {
                    "status": "ok",
                    "record_type": "categorical_frequency",
                    "comparison": f"labeled_vs_{target_name}",
                    "feature": column,
                    "level": f"<OTHER_{len(omitted)}_LEVELS>",
                    "known_n": len(a),
                    "target_n": len(b),
                    "known_fraction": af,
                    "target_fraction": bf,
                    "fraction_difference_target_minus_known": bf - af,
                }
            )
    return rows


def diagnostic_domain_classifier_rows(
    known: pd.DataFrame,
    target: pd.DataFrame,
    target_name: str,
    feature_columns: Sequence[str],
    folds: int,
    seed: int,
) -> list[dict]:
    usable = [column for column in feature_columns if column in known and column in target]
    missing = sorted(set(feature_columns) - set(usable))
    if not usable:
        return [
            {
                "status": "missing_fields",
                "record_type": "diagnostic_known_vs_unknown_classifier",
                "comparison": f"labeled_vs_{target_name}",
                "missing_features": ",".join(missing),
            }
        ]
    x = pd.concat([known[usable], target[usable]], ignore_index=True).apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    y = np.concatenate([np.zeros(len(known), dtype=int), np.ones(len(target), dtype=int)])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    rows = []
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed + fold,
                    ),
                ),
            ]
        )
        model.fit(x.iloc[train_index], y[train_index])
        probability = model.predict_proba(x.iloc[test_index])[:, 1]
        score = float(roc_auc_score(y[test_index], probability))
        scores.append(score)
        rows.append(
            {
                "status": "ok",
                "record_type": "diagnostic_known_vs_unknown_classifier_fold",
                "comparison": f"labeled_vs_{target_name}",
                "fold": fold,
                "cv_auroc": score,
                "train_n": len(train_index),
                "test_n": len(test_index),
                "feature_count": len(usable),
                "classifier": "median-imputed standardized class-balanced logistic regression",
                "interpretation": "domain-shift diagnostic only; not a material-property model",
                "cv_seed": seed,
            }
        )
    rows.append(
        {
            "status": "ok",
            "record_type": "diagnostic_known_vs_unknown_classifier_summary",
            "comparison": f"labeled_vs_{target_name}",
            "fold": "mean_std",
            "cv_auroc": float(np.mean(scores)),
            "cv_auroc_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "known_n": len(known),
            "target_n": len(target),
            "feature_count": len(usable),
            "missing_feature_count": len(missing),
            "classifier": "median-imputed standardized class-balanced logistic regression",
            "interpretation": "domain-shift diagnostic only; not a material-property model",
            "cv_seed": seed,
        }
    )
    return rows


def run_pool_shift(
    project_root: Path,
    cv_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, list[Path], dict]:
    supervised_path = project_root / "data/processed/supervised.csv"
    unknown_path = project_root / "data/processed/unknown_pool.csv"
    eligible_path = project_root / "data/processed/unknown_thermo_eligible.csv"
    feature_path = project_root / "data/processed/tabular_feature_columns.json"
    supervised = pd.read_csv(supervised_path)
    unknown = pd.read_csv(unknown_path)
    eligible_file = pd.read_csv(eligible_path)
    feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
    eligible = unknown[pd.to_numeric(unknown["ehull"], errors="coerce") <= 0.10].copy()
    file_ids = set(eligible_file["material_id"].astype(str))
    derived_ids = set(eligible["material_id"].astype(str))
    eligible_match = file_ids == derived_ids
    continuous = list(dict.fromkeys(["ehull", *feature_columns, "cell_c", "cell_alpha", "cell_beta", "cell_volume", "mass_per_volume_proxy"]))
    categorical = [
        "anonymous_formula",
        "chemical_system",
        "structure_family",
        "layergroup",
        "bravais_type",
        "source_label",
    ]
    rows: list[dict] = []
    for target_name, target in (("unknown_all", unknown), ("unknown_thermo_eligible", eligible)):
        rows.extend(continuous_shift_rows(supervised, target, target_name, continuous))
        rows.extend(categorical_shift_rows(supervised, target, target_name, categorical))
        rows.extend(
            diagnostic_domain_classifier_rows(
                supervised,
                target,
                target_name,
                feature_columns,
                cv_folds,
                seed,
            )
        )
    metadata = {
        "labeled_count": int(len(supervised)),
        "unknown_all_count": int(len(unknown)),
        "unknown_thermo_eligible_count_derived_ehull_le_0.10": int(len(eligible)),
        "unknown_thermo_eligible_file_count": int(len(eligible_file)),
        "eligible_file_matches_derived_ids": eligible_match,
        "diagnostic_classifier": "Stratified CV Known-vs-Unknown domain classifier; explicitly not a material-property model",
        "embedding_distance": {
            "status": "not_run_missing_saved_embeddings",
            "reason": "config.yaml sets outputs.save_embeddings=false; no embedding distance was fabricated",
        },
    }
    return pd.DataFrame(rows), [supervised_path, unknown_path, eligible_path, feature_path], metadata


def _shannon(values: pd.Series) -> float:
    counts = values.fillna("<MISSING>").astype(str).value_counts(normalize=True)
    return float(-(counts * np.log(counts)).sum()) if len(counts) else math.nan


def _read_json_mapping(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _required_finite_number(mapping: dict, key: str, source: Path) -> float:
    if key not in mapping:
        raise KeyError(f"{source} missing required key: {key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{source}:{key} must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{source}:{key} must be finite")
    return value


def load_unknown_ensemble_calibration(
    project_root: Path,
    prediction: pd.DataFrame,
    probability_column: str,
) -> tuple[dict, list[Path]]:
    """Strictly bind all-pool calibrated pJ to its matching five-member ensemble."""
    calibrated_aliases = {"pJ_calibrated", "p_joint_calibrated"}
    if probability_column not in calibrated_aliases:
        raise ValueError(
            "Threshold sensitivity requires calibrated all-pool pJ; "
            f"received {probability_column!r}"
        )
    require_columns(
        prediction,
        {"p_joint_calibrated"},
        "all-pool Unknown predictions",
    )
    if probability_column == "pJ_calibrated":
        require_columns(
            prediction,
            {"pJ_calibrated"},
            "all-pool Unknown predictions",
        )
        if not np.allclose(
            prediction["pJ_calibrated"].to_numpy(float),
            prediction["p_joint_calibrated"].to_numpy(float),
            rtol=0.0,
            atol=1.0e-12,
            equal_nan=True,
        ):
            raise ValueError("pJ_calibrated is not an exact alias of p_joint_calibrated")

    member_indices = sorted(
        int(column.removeprefix("p_joint_member_"))
        for column in prediction.columns
        if column.startswith("p_joint_member_")
        and column.removeprefix("p_joint_member_").isdigit()
    )
    if not member_indices or member_indices != list(range(1, len(member_indices) + 1)):
        raise ValueError(
            "All-pool predictions must contain consecutive p_joint_member_1..N columns"
        )
    detected_members = len(member_indices)

    ensemble_dir = project_root / "results/main/M06_full_mtl/ensemble"
    thresholds_path = ensemble_dir / "thresholds.json"
    temperatures_path = ensemble_dir / "temperatures.json"
    metrics_path = ensemble_dir / "ensemble_metrics.json"
    screening_summary_path = (
        project_root
        / "results/main/M10_unknown_screening/screening_summary.json"
    )
    thresholds = _read_json_mapping(thresholds_path)
    temperatures = _read_json_mapping(temperatures_path)
    metrics = _read_json_mapping(metrics_path)
    screening_summary = _read_json_mapping(screening_summary_path)
    threshold = _required_finite_number(thresholds, "joint", thresholds_path)
    temperature = _required_finite_number(
        temperatures, "joint", temperatures_path
    )

    member_count = metrics.get("member_count")
    member_paths = metrics.get("member_paths")
    calibrated_thresholds = metrics.get("calibrated_thresholds")
    metric_temperatures = metrics.get("temperatures")
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count <= 0
    ):
        raise TypeError(f"{metrics_path}:member_count must be a positive integer")
    if not isinstance(member_paths, list) or len(member_paths) != member_count:
        raise ValueError(
            f"{metrics_path}:member_paths must contain exactly member_count entries"
        )
    if not isinstance(calibrated_thresholds, dict):
        raise TypeError(f"{metrics_path}:calibrated_thresholds must be an object")
    if not isinstance(metric_temperatures, dict):
        raise TypeError(f"{metrics_path}:temperatures must be an object")
    metrics_threshold = _required_finite_number(
        calibrated_thresholds, "joint", metrics_path
    )
    metrics_temperature = _required_finite_number(
        metric_temperatures, "joint", metrics_path
    )
    if not math.isclose(threshold, metrics_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("thresholds.json and ensemble_metrics.json disagree for joint")
    if not math.isclose(
        temperature, metrics_temperature, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("temperatures.json and ensemble_metrics.json disagree for joint")
    if detected_members != member_count:
        raise ValueError(
            "Cross-ensemble threshold mismatch: all-pool predictions contain "
            f"{detected_members} members but calibration metadata contains "
            f"{member_count}"
        )
    if screening_summary.get("pJ_calibrated_alias") != "p_joint_calibrated":
        raise ValueError(
            f"{screening_summary_path}:pJ_calibrated_alias must be "
            "'p_joint_calibrated'"
        )

    return (
        {
            "probability_scale": "calibrated",
            "canonical_probability_column": "p_joint_calibrated",
            "selected_probability_column": probability_column,
            "ensemble_members": member_count,
            "member_columns_detected": detected_members,
            "member_paths": member_paths,
            "temperature_joint": temperature,
            "calibrated_decision_threshold_joint": threshold,
            "temperature_application": metrics.get("temperature_application"),
            "threshold_source": str(thresholds_path.relative_to(project_root)),
            "temperature_source": str(temperatures_path.relative_to(project_root)),
            "ensemble_metadata_source": str(metrics_path.relative_to(project_root)),
            "screening_summary_source": str(
                screening_summary_path.relative_to(project_root)
            ),
        },
        [
            thresholds_path,
            temperatures_path,
            metrics_path,
            screening_summary_path,
        ],
    )


def run_threshold_sensitivity(
    project_root: Path,
) -> tuple[pd.DataFrame, list[Path], dict]:
    supervised_path = project_root / "data/processed/supervised.csv"
    unknown_path = project_root / "data/processed/unknown_pool.csv"
    prediction_path = project_root / "results/main/M10_unknown_screening/unknown_all_predictions.csv"
    supervised = pd.read_csv(supervised_path)
    unknown = pd.read_csv(unknown_path)
    prediction = read_prediction(prediction_path)
    if len(prediction) != 8803:
        raise ValueError(f"Expected direct all-pool frozen predictions for 8,803 Unknown samples, found {len(prediction)}")
    p_candidates = ["pJ_calibrated", "p_joint_calibrated"]
    probability_column = next((column for column in p_candidates if column in prediction), None)
    if probability_column is None:
        raise KeyError(f"{prediction_path} lacks all pJ candidates {p_candidates}")
    ensemble_calibration, calibration_sources = load_unknown_ensemble_calibration(
        project_root,
        prediction,
        probability_column,
    )
    p_joint_threshold = ensemble_calibration[
        "calibrated_decision_threshold_joint"
    ]
    metadata_columns = ["material_id", "ehull", "chemical_system", "structure_family", "layergroup"]
    merged = unknown[metadata_columns].merge(
        prediction[["material_id", probability_column]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(unknown) or len(merged) != len(prediction):
        raise ValueError(
            f"All-pool pJ coverage mismatch: unknown={len(unknown)}, prediction={len(prediction)}, matched={len(merged)}"
        )
    full_pool_positive_count = int(
        (merged[probability_column] >= p_joint_threshold).sum()
    )
    rows = []
    top_sets: dict[tuple[float, int], set[str]] = {}
    for threshold in EHULL_THRESHOLDS:
        quadrant = quadrant_counts(supervised["ehull"], supervised["dynamic_label"], threshold)
        candidates = merged[pd.to_numeric(merged["ehull"], errors="coerce") <= threshold].copy()
        candidates = candidates.sort_values(
            [probability_column, "material_id"], ascending=[False, True], kind="mergesort"
        )
        rows.append(
            {
                "status": "ok",
                "record_type": "ehull_threshold_statistics",
                "ehull_threshold_eV_per_atom": threshold,
                "pJ_source_column": probability_column,
                "unknown_full_pool_count": len(merged),
                "unknown_candidate_count": len(candidates),
                "unknown_predicted_joint_positive_count": int((candidates[probability_column] >= p_joint_threshold).sum()),
                "frozen_pJ_decision_threshold": p_joint_threshold,
                **quadrant,
            }
        )
        for k in TOP_K_VALUES:
            selected = candidates.head(min(k, len(candidates)))
            ids = set(selected["material_id"].astype(str))
            top_sets[(threshold, k)] = ids
            elements = set()
            for chemical_system in selected["chemical_system"].fillna("").astype(str):
                elements.update(part for part in chemical_system.split("-") if part)
            rows.append(
                {
                    "status": "ok",
                    "record_type": "unknown_topk_diversity",
                    "ehull_threshold_eV_per_atom": threshold,
                    "requested_k": k,
                    "effective_k": len(selected),
                    "pJ_source_column": probability_column,
                    "unique_chemical_systems": int(selected["chemical_system"].nunique(dropna=False)),
                    "unique_structure_families": int(selected["structure_family"].nunique(dropna=False)),
                    "unique_layergroups": int(selected["layergroup"].nunique(dropna=False)),
                    "unique_elements": len(elements),
                    "structure_family_shannon_nats": _shannon(selected["structure_family"]),
                    "minimum_selected_pJ": float(selected[probability_column].min()) if len(selected) else math.nan,
                    "mean_selected_pJ": float(selected[probability_column].mean()) if len(selected) else math.nan,
                }
            )
    for left_index, left in enumerate(EHULL_THRESHOLDS):
        for right in EHULL_THRESHOLDS[left_index + 1 :]:
            for k in TOP_K_VALUES:
                left_ids, right_ids = top_sets[(left, k)], top_sets[(right, k)]
                intersection = len(left_ids & right_ids)
                union = len(left_ids | right_ids)
                rows.append(
                    {
                        "status": "ok",
                        "record_type": "unknown_topk_overlap",
                        "ehull_threshold_left": left,
                        "ehull_threshold_right": right,
                        "requested_k": k,
                        "left_effective_k": len(left_ids),
                        "right_effective_k": len(right_ids),
                        "overlap_count": intersection,
                        "overlap_fraction_of_smaller_set": intersection / min(len(left_ids), len(right_ids)) if min(len(left_ids), len(right_ids)) else math.nan,
                        "jaccard": intersection / union if union else math.nan,
                        "pJ_source_column": probability_column,
                    }
                )
    table = pd.DataFrame(rows)
    provenance_columns = {
        "pJ_probability_scale": ensemble_calibration["probability_scale"],
        "pJ_ensemble_members": ensemble_calibration["ensemble_members"],
        "pJ_member_columns_detected": ensemble_calibration[
            "member_columns_detected"
        ],
        "pJ_temperature_joint": ensemble_calibration["temperature_joint"],
        "pJ_calibrated_decision_threshold_joint": p_joint_threshold,
        "pJ_threshold_source": ensemble_calibration["threshold_source"],
        "pJ_temperature_source": ensemble_calibration["temperature_source"],
        "pJ_ensemble_metadata_source": ensemble_calibration[
            "ensemble_metadata_source"
        ],
    }
    for column, value in provenance_columns.items():
        table[column] = value
    metadata = {
        "unknown_prediction_count": int(len(prediction)),
        "unknown_pool_count": int(len(unknown)),
        "matched_direct_pJ_count": int(len(merged)),
        "unknown_full_pool_predicted_joint_positive_count": (
            full_pool_positive_count
        ),
        "unknown_full_pool_predicted_joint_positive_fraction": (
            full_pool_positive_count / len(merged)
        ),
        "pJ_source_column": probability_column,
        "pJ_ensemble_calibration": ensemble_calibration,
        "ehull_thresholds_eV_per_atom": list(EHULL_THRESHOLDS),
        "top_k_values": list(TOP_K_VALUES),
        "interpretation": (
            "All 8,803 Unknown materials retain calibrated pJ predictions. "
            "The decision threshold and temperature come from the same frozen "
            "five-member M06 ensemble used to generate those predictions; E_hull "
            "thresholds are post-hoc candidate filters only."
        ),
    }
    return (
        table,
        [
            supervised_path,
            unknown_path,
            prediction_path,
            *calibration_sources,
        ],
        metadata,
    )


def file_fingerprint(path: Path, project_root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def write_workbook(tables: dict[str, pd.DataFrame], output_path: Path) -> dict:
    sheet_names = {
        "bootstrap_summary": "bootstrap",
        "logic_violations": "logic_violations",
        "error_taxonomy": "error_taxonomy",
        "calibration_shift": "calibration_shift",
        "reliability_bins": "reliability_bins",
        "pool_shift": "pool_shift",
        "threshold_sensitivity": "threshold_sensitivity",
        "probability_uncertainty_bins": "pJ_uncertainty_bins",
    }
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name, frame in tables.items():
                frame.to_excel(writer, sheet_name=sheet_names[name][:31], index=False)
        return {"status": "ok", "path": str(output_path)}
    except ImportError as error:
        return {"status": "skipped_missing_openpyxl", "error": str(error)}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run frozen-prediction, no-retraining peer-review supplements."
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root / "results/review_supplements").resolve()
    output.mkdir(parents=True, exist_ok=True)
    component_status: dict[str, dict] = {}
    input_sources: list[Path] = []

    def run_component(name: str, function, *function_args):
        try:
            value = function(*function_args)
            component_status[name] = {"status": "ok"}
            return value
        except Exception as error:  # keep every requested output auditable
            component_status[name] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            return None

    bootstrap_result = run_component(
        "bootstrap",
        run_bootstrap_experiment,
        root,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    if bootstrap_result is None:
        bootstrap = pd.DataFrame([{"status": "error", **component_status["bootstrap"]}])
    else:
        bootstrap, sources = bootstrap_result
        input_sources.extend(sources)

    logic_result = run_component("logic_violations", run_logic_experiment, root)
    if logic_result is None:
        logic = pd.DataFrame([{"status": "error", **component_status["logic_violations"]}])
        logic_temperatures = {}
    else:
        logic, sources, logic_temperatures = logic_result
        input_sources.extend(sources)

    ood_result = run_component(
        "scenario_specific_development_calibration_and_errors",
        build_frozen_ood_evaluations,
        root,
        args.calibration_bins,
    )
    if ood_result is None:
        status = component_status[
            "scenario_specific_development_calibration_and_errors"
        ]
        calibration = pd.DataFrame([{"status": "error", **status}])
        reliability = pd.DataFrame([{"status": "error", **status}])
        taxonomy = pd.DataFrame([{"status": "error", **status}])
        probability_uncertainty = pd.DataFrame([{"status": "error", **status}])
        calibration_protocol_details = {}
    else:
        (
            calibration,
            reliability,
            taxonomy,
            probability_uncertainty,
            calibration_protocol_details,
            sources,
            _,
        ) = ood_result
        input_sources.extend(sources)

    pool_result = run_component(
        "pool_shift",
        run_pool_shift,
        root,
        args.cv_folds,
        args.bootstrap_seed,
    )
    if pool_result is None:
        pool_shift = pd.DataFrame([{"status": "error", **component_status["pool_shift"]}])
        pool_metadata = {}
    else:
        pool_shift, sources, pool_metadata = pool_result
        input_sources.extend(sources)

    threshold_result = run_component(
        "threshold_sensitivity",
        run_threshold_sensitivity,
        root,
    )
    if threshold_result is None:
        threshold_sensitivity = pd.DataFrame(
            [{"status": component_status["threshold_sensitivity"]["status"], **component_status["threshold_sensitivity"]}]
        )
        threshold_metadata = {}
    else:
        threshold_sensitivity, sources, threshold_metadata = threshold_result
        input_sources.extend(sources)

    tables = {
        "bootstrap_summary": bootstrap,
        "logic_violations": logic,
        "error_taxonomy": taxonomy,
        "calibration_shift": calibration,
        "reliability_bins": reliability,
        "pool_shift": pool_shift,
        "threshold_sensitivity": threshold_sensitivity,
        "probability_uncertainty_bins": probability_uncertainty,
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    workbook_status = write_workbook(tables, output / "summary.xlsx")
    unique_sources = sorted({path.resolve() for path in input_sources if path.exists()})
    source_manifest = [file_fingerprint(path, root) for path in unique_sources]
    overall_status = (
        "completed"
        if all(value.get("status") == "ok" for value in component_status.values())
        else "completed_with_recorded_component_errors"
    )
    summary = {
        "status": overall_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "output_directory": str(output),
        "method_guard": {
            "retraining_performed": False,
            "checkpoint_updated": False,
            "predictions_regenerated": False,
            "analysis_type": "read-only post-processing of frozen predictions and processed tabular data",
            "legacy_iid_split_limitation": {
                "status": "known_high_priority_scientific_limitation",
                "affected_frozen_runs": [
                    "M04_single_alignn",
                    "M05_equal_mtl",
                    "M06_full_mtl",
                    "A03_no_logic",
                ],
                "affected_postprocessing": [
                    "paired_bootstrap",
                    "logic_violations",
                    "legacy_IID_diagnostics",
                ],
                "canonical_composition_cross_split_groups": 399,
                "audit_finding": (
                    "These frozen runs use the legacy iid_split. A subsequent "
                    "audit identified 399 canonical-composition groups crossing "
                    "split boundaries."
                ),
                "interpretation_guard": (
                    "These post-hoc supplements cannot replace or be presented "
                    "as the four strict-split GNN main experiments."
                ),
            },
        },
        "shared_seeds": list(COMMON_SEEDS),
        "bootstrap": {
            "unit": "material_id",
            "replicates": args.bootstrap_replicates,
            "base_seed": args.bootstrap_seed,
            "difference_direction": "M06 Full minus comparator",
        },
        "calibration_protocol": calibration_protocol_details,
        "logic_calibration_temperatures_fit_on_each_model_IID_validation": logic_temperatures,
        "pool_shift": pool_metadata,
        "threshold_sensitivity": threshold_metadata,
        "component_status": component_status,
        "missing_embedding_policy": {
            "status": "not_run_missing_saved_embeddings",
            "reason": "No saved embeddings are present and config.yaml sets outputs.save_embeddings=false; embedding distances were not fabricated.",
        },
        "input_source_manifest": source_manifest,
        "outputs": [f"{name}.csv" for name in tables] + ["summary.json", "summary.xlsx"],
        "workbook": workbook_status,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_safe_json_value),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_safe_json_value))
    if overall_status != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
