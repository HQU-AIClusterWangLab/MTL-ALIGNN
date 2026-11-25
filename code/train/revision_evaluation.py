"""Validation-only calibration and matched-seed summaries for revision controls.

Four-state calibration fits ONE scalar on native 4-class logits. Dual-product
calibration fits the two marginals, then derives each member's joint probability.
Test labels are used only for the final metrics, never to choose a temperature,
threshold, fusion rule, seed subset, or uncertainty score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from models.uncertainty import aggregate_binary_probabilities
from train.aggregate_ensemble import base_columns, fit_temperature, member_logits
from train.common import save_json
from train.metrics import (
    best_mcc_threshold, classification_metrics, reliability_table,
    risk_coverage_curve, sigmoid, uncertainty_metrics,
)

TASKS = ("thermo", "dynamic", "joint")
STATES = ("00", "01", "10", "11")


def read_members(root: Path, seeds: list[int], split: str) -> list[pd.DataFrame]:
    frames = []
    for seed in seeds:
        folder = root / f"seed_{seed}"
        if not (folder / "COMPLETED.json").is_file():
            raise RuntimeError(f"Incomplete member: {folder}")
        frame = pd.read_csv(folder / f"{split}_predictions.csv")
        if frame.material_id.isna().any() or frame.material_id.duplicated().any():
            raise ValueError(f"Invalid or duplicate IDs: {folder}/{split}")
        frame = frame.sort_values("material_id").reset_index(drop=True)
        if frames:
            pd.testing.assert_frame_equal(
                frames[0][["material_id", *[f"y_{t}" for t in TASKS]]],
                frame[["material_id", *[f"y_{t}" for t in TASKS]]],
            )
        if not np.isfinite(frame.filter(regex=r"^(p_|logit_)").to_numpy(float)).all():
            raise ValueError(f"Non-finite prediction: {folder}/{split}")
        frames.append(frame)
    return frames


def state_matrix(frames: list[pd.DataFrame]) -> np.ndarray:
    return np.stack([f[[f"logit_state_{s}" for s in STATES]].to_numpy(float) for f in frames])


def fit_calibration(frames: list[pd.DataFrame], mode: str) -> dict[str, float]:
    if mode == "four_state":
        logits = state_matrix(frames)
        labels = (2 * frames[0].y_thermo + frames[0].y_dynamic).to_numpy(int)

        def nll(log_temperature):
            scaled = logits / np.exp(log_temperature)
            log_probs = scaled - logsumexp(scaled, axis=2, keepdims=True)
            log_ensemble = logsumexp(log_probs, axis=0) - np.log(len(frames))
            return -float(log_ensemble[np.arange(len(labels)), labels].mean())

        result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"Four-state temperature fit failed: {result}")
        return {"four_state": float(np.exp(result.x))}
    tasks = ("thermo", "dynamic") if mode == "dual_product" else TASKS
    return {t: fit_temperature(frames[0][f"y_{t}"], member_logits(frames, t))
            for t in tasks if f"p_{t}" in frames[0]}


def probability_matrices(frames, mode: str, temperatures=None) -> dict[str, np.ndarray]:
    temperatures = temperatures or {}
    if mode == "four_state":
        q = softmax(state_matrix(frames) / temperatures.get("four_state", 1.0), axis=2)
        return {"thermo": q[:, :, 2] + q[:, :, 3],
                "dynamic": q[:, :, 1] + q[:, :, 3], "joint": q[:, :, 3]}
    result = {}
    tasks = ("thermo", "dynamic") if mode == "dual_product" else TASKS
    for task in tasks:
        if f"p_{task}" not in frames[0]:
            continue
        result[task] = (
            sigmoid(member_logits(frames, task) / temperatures[task])
            if task in temperatures else np.stack([f[f"p_{task}"].to_numpy(float) for f in frames])
        )
    if mode == "dual_product":
        # Average member products; a product of ensemble means is a different rule.
        result["joint"] = result["thermo"] * result["dynamic"]
    return result


def violation_stats(probabilities: dict, tolerance=1e-7) -> dict:
    if not set(TASKS).issubset(probabilities):
        return {}
    t, d, j = (np.asarray(probabilities[task]) for task in TASKS)
    upper = np.maximum(np.maximum(j - t, j - d), 0)
    lower = np.maximum(t + d - 1 - j, 0)
    return {"any_violation": float(((upper > tolerance) | (lower > tolerance)).mean()),
            "upper_violation": float((upper > tolerance).mean()),
            "lower_violation": float((lower > tolerance).mean()),
            "mean_violation_magnitude": float((upper + lower).mean()),
            "tolerance": tolerance}


def evaluate_experiment(root: Path, seeds: list[int], mode: str, output: Path, bins=15):
    val_frames = read_members(root, seeds, "val")
    test_frames = read_members(root, seeds, "test")
    if set(val_frames[0].material_id) & set(test_frames[0].material_id):
        raise ValueError(f"Validation/test overlap in {root}")
    temperatures = fit_calibration(val_frames, mode)
    output.mkdir(parents=True, exist_ok=True)
    ensemble_frames = {"val": val_frames[0][base_columns(val_frames[0])].copy(),
                       "test": test_frames[0][base_columns(test_frames[0])].copy()}
    metrics_rows, reliability_rows, risk_rows = [], [], []
    thresholds, violations = {}, {}
    for stage, temps in (("raw", None), ("calibrated", temperatures)):
        matrices = {"val": probability_matrices(val_frames, mode, temps),
                    "test": probability_matrices(test_frames, mode, temps)}
        thresholds[stage] = {}
        for split, values in matrices.items():
            violations[f"{split}_{stage}_members"] = violation_stats(values)
            violations[f"{split}_{stage}_ensemble"] = violation_stats({t: v.mean(0) for t, v in values.items()})
        for task in matrices["val"]:
            val_p = matrices["val"][task].mean(0)
            threshold = best_mcc_threshold(val_frames[0][f"y_{task}"], val_p)
            thresholds[stage][task] = threshold
            for split, values in matrices.items():
                frame = ensemble_frames[split]
                stats = aggregate_binary_probabilities(values[task])
                y = frame[f"y_{task}"].to_numpy(int)
                p = stats["mean"]
                frame[f"p_{task}_{'mean' if stage == 'raw' else stage}"] = p
                for key in ("std", "predictive_entropy", "expected_entropy", "mutual_information"):
                    frame[f"{task}_{stage}_{key}"] = stats[key]
                row = {"split": split, "stage": stage, "task": task,
                       "members": len(seeds), "n": len(y),
                       **classification_metrics(y, p, threshold, bins)}
                if split == "test":
                    for score in ("std", "predictive_entropy", "mutual_information"):
                        row.update({f"{score}_{key}": value for key, value in
                                    uncertainty_metrics(y, p, stats[score], threshold).items()})
                        curve = risk_coverage_curve(y, p, stats[score], threshold)
                        curve = curve.iloc[np.unique(np.linspace(0, len(curve) - 1, min(201, len(curve))).astype(int))]
                        risk_rows.append(curve.assign(task=task, stage=stage, score=score))
                    reliability_rows.append(reliability_table(y, p, bins).assign(task=task, stage=stage))
                metrics_rows.append(row)
    for split, frame in ensemble_frames.items():
        frame.to_csv(output / f"{split}_ensemble_predictions.csv", index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(output / "test_reliability.csv", index=False)
    pd.concat(risk_rows, ignore_index=True).to_csv(output / "test_risk_coverage.csv", index=False)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output / "ensemble_metrics.csv", index=False)
    save_json({"seeds": seeds, "source": str(root), "output_mode": mode,
               "calibration_fit_split": "val", "threshold_fit_split": "val",
               "temperatures": temperatures, "thresholds": thresholds,
               "logical_violations": violations,
               "fusion_rule": "mean of member products" if mode == "dual_product" else mode},
              output / "calibration_and_logic.json")
    # Compute member summaries on precisely the requested seed subset.
    members = []
    for seed, val, test in zip(seeds, val_frames, test_frames):
        native = probability_matrices([test], mode)
        native_val = probability_matrices([val], mode)
        info = json.loads((root / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
        for task, p in native.items():
            threshold = best_mcc_threshold(val[f"y_{task}"], native_val[task][0])
            members.append({"seed": seed, "task": task, "best_epoch": info.get("best_epoch"),
                            "batch_size": info.get("batch_size"),
                            "parameter_count": info.get("parameter_count"),
                            "runtime_seconds": info.get("runtime_seconds"),
                            **classification_metrics(test[f"y_{task}"], p[0], threshold, bins)})
    member_frame = pd.DataFrame(members)
    member_frame.to_csv(output / "member_metrics.csv", index=False)
    return metrics, member_frame


def summarize_matrix(entries: list[dict], seeds: list[int], output: Path, bins=15):
    """Only completed, matched members enter comparisons; never silently cherry-pick."""
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "COMPLETED.json"
    if marker.exists():
        marker.unlink()
    all_metrics, all_members, audit = [], [], []
    references = {}
    for entry in entries:
        root = Path(entry["root"])
        # Validate IDs and labels across methods, separately for each frozen split.
        for part in ("val", "test"):
            frame = read_members(root, seeds, part)[0]
            columns = ["material_id", *[f"y_{t}" for t in TASKS]]
            key = (entry["split"], part)
            if key in references:
                pd.testing.assert_frame_equal(references[key], frame[columns])
            else:
                references[key] = frame[columns]
        metrics, members = evaluate_experiment(root, seeds, entry["mode"], output / entry["name"], bins)
        tags = {"experiment": entry["name"], "data_split": entry["split"],
                "method": entry["method"], "source": entry["source"]}
        all_metrics.append(metrics.assign(**tags))
        all_members.append(members.assign(**tags))
        audit.append({**tags, "root": str(root), "seeds": seeds})
    metrics = pd.concat(all_metrics, ignore_index=True)
    members = pd.concat(all_members, ignore_index=True)
    metrics.to_csv(output / "ensemble_comparison.csv", index=False)
    members.to_csv(output / "member_comparison.csv", index=False)
    numeric = ["pr_auc", "mcc", "f1", "nll", "brier", "ece"]
    summary = members.groupby(["data_split", "method", "experiment", "task"])[numeric].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "seed_mean_std.csv", index=False)
    comparisons = []
    for split, group in members[members.task == "joint"].groupby("data_split"):
        full = group[group.method == "Full"].set_index("seed")
        if full.empty:
            continue
        for method, method_group in group.groupby("method"):
            for row in method_group.to_dict("records"):
                comparisons.append({"data_split": split, "method": method, "seed": row["seed"],
                                    **{f"delta_{m}_vs_Full": row[m] - full.loc[row["seed"], m] for m in numeric}})
    pd.DataFrame(comparisons).to_csv(output / "paired_seed_differences.csv", index=False)
    save_json({"entries": audit, "seeds": seeds, "std_ddof": 1,
               "scope": "internal retrospective supplement; external DFT validation not yet performed",
               "ap_definition": "sklearn average_precision_score (stepwise AP)",
               "no_test_selection": True}, output / "summary_manifest.json")
    save_json({"status": "completed", "seeds": seeds, "experiments": len(entries)}, output / "COMPLETED.json")
