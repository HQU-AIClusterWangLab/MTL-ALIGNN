from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models.uncertainty import aggregate_binary_probabilities
from train.common import save_json
from train.metrics import (
    best_mcc_threshold,
    binary_nll,
    classification_metrics,
    logit,
    sigmoid,
)


TASKS = ["thermo", "dynamic", "joint"]


def load_members(experiment_dir: Path, split: str):
    paths = sorted(experiment_dir.glob(f"seed_*/{split}_predictions.csv"))
    if len(paths) < 2:
        raise RuntimeError(
            f"At least two member predictions are required in {experiment_dir}."
        )
    frames = [
        pd.read_csv(path).sort_values("material_id").reset_index(drop=True)
        for path in paths
    ]
    reference = frames[0]["material_id"].tolist()
    for path, frame in zip(paths[1:], frames[1:]):
        if frame["material_id"].tolist() != reference:
            raise RuntimeError(f"Member predictions are not aligned: {path}")
    return frames, paths


def member_logits(frames, task: str) -> np.ndarray:
    logits = []
    for frame in frames:
        column = f"logit_{task}"
        if column in frame:
            logits.append(frame[column].to_numpy(dtype=float))
        else:
            logits.append(logit(frame[f"p_{task}"].to_numpy(dtype=float)))
    return np.vstack(logits)


def fit_temperature(labels, logits_matrix: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    logits_matrix = np.asarray(logits_matrix, dtype=float)
    if logits_matrix.ndim == 1:
        logits_matrix = logits_matrix[None, :]

    def objective(log_temperature):
        temperature = float(np.exp(log_temperature))
        calibrated = sigmoid(logits_matrix / temperature).mean(axis=0)
        return binary_nll(labels, calibrated)

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result}")
    return float(np.exp(result.x))


def base_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
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
        "graph_path",
        "y_thermo",
        "y_dynamic",
        "y_joint",
    ]
    return [column for column in columns if column in frame]


def aggregate(frames, temperatures: dict[str, float] | None = None):
    result = frames[0][base_columns(frames[0])].copy()
    for task in TASKS:
        probability_column = f"p_{task}"
        if probability_column not in frames[0]:
            continue
        probability_matrix = np.vstack(
            [frame[probability_column].to_numpy(dtype=float) for frame in frames]
        )
        logits_matrix = member_logits(frames, task)
        stats = aggregate_binary_probabilities(probability_matrix)
        result[f"p_{task}_mean"] = stats["mean"]
        result[f"p_{task}_std"] = stats["std"]
        result[f"{task}_predictive_entropy"] = stats["predictive_entropy"]
        result[f"{task}_expected_entropy"] = stats["expected_entropy"]
        result[f"{task}_mutual_information"] = stats["mutual_information"]
        result[f"{task}_variation_ratio"] = stats["variation_ratio"]
        result[f"logit_{task}_mean"] = logits_matrix.mean(axis=0)
        if temperatures and task in temperatures:
            result[f"p_{task}_calibrated"] = sigmoid(
                logits_matrix / float(temperatures[task])
            ).mean(axis=0)
        for member_index, (probability, member_logit) in enumerate(
            zip(probability_matrix, logits_matrix), start=1
        ):
            result[f"p_{task}_member_{member_index}"] = probability
            result[f"logit_{task}_member_{member_index}"] = member_logit
    return result


def logical_violation_rates(frame: pd.DataFrame, suffix: str) -> dict:
    thermo = frame[f"p_thermo_{suffix}"].to_numpy(float)
    dynamic = frame[f"p_dynamic_{suffix}"].to_numpy(float)
    joint = frame[f"p_joint_{suffix}"].to_numpy(float)
    tolerance = 1e-10
    return {
        "joint_gt_thermo": float(np.mean(joint > thermo + tolerance)),
        "joint_gt_dynamic": float(np.mean(joint > dynamic + tolerance)),
        "frechet_lower_bound": float(
            np.mean(joint + tolerance < thermo + dynamic - 1.0)
        ),
        "any_violation": float(
            np.mean(
                (joint > thermo + tolerance)
                | (joint > dynamic + tolerance)
                | (joint + tolerance < thermo + dynamic - 1.0)
            )
        ),
    }


def ensemble_size_ablation(
    val_frames,
    test_frames,
    output_dir: Path,
    calibration_bins: int,
):
    detail_rows = []
    max_members = len(val_frames)
    requested_counts = sorted(set([1, 3, 5, max_members]))
    counts = [value for value in requested_counts if value <= max_members]
    for count in counts:
        combinations = list(itertools.combinations(range(max_members), count))
        # All subsets are feasible for five members; retain a deterministic cap
        # for future larger ensembles.
        combinations = combinations[:50]
        for subset_id, indices in enumerate(combinations, start=1):
            subset_val = [val_frames[index] for index in indices]
            subset_test = [test_frames[index] for index in indices]
            val_raw = aggregate(subset_val)
            temperatures = {
                task: fit_temperature(
                    val_raw[f"y_{task}"], member_logits(subset_val, task)
                )
                for task in TASKS
            }
            val = aggregate(subset_val, temperatures)
            test = aggregate(subset_test, temperatures)
            for task in TASKS:
                threshold = best_mcc_threshold(
                    val[f"y_{task}"], val[f"p_{task}_calibrated"]
                )
                metrics = classification_metrics(
                    test[f"y_{task}"],
                    test[f"p_{task}_calibrated"],
                    threshold,
                    calibration_bins,
                )
                detail_rows.append(
                    {
                        "members": count,
                        "subset_id": subset_id,
                        "member_indices": ",".join(str(i + 1) for i in indices),
                        "task": task,
                        "temperature": temperatures[task],
                        **metrics,
                    }
                )
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(output_dir / "ensemble_size_ablation_detail.csv", index=False)
    numeric = [
        column
        for column in detail.select_dtypes(include=[np.number]).columns
        if column not in {"members", "subset_id"}
    ]
    summary_rows = []
    for (members, task), group in detail.groupby(["members", "task"]):
        row = {"members": members, "task": task, "subsets": len(group)}
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "ensemble_size_ablation.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--calibration-bins", type=int, default=15)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = experiment_dir / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    val_frames, paths = load_members(experiment_dir, "val")
    test_frames, _ = load_members(experiment_dir, "test")

    val_raw = aggregate(val_frames)
    temperatures = {
        task: fit_temperature(
            val_raw[f"y_{task}"], member_logits(val_frames, task)
        )
        for task in TASKS
    }
    val = aggregate(val_frames, temperatures)
    test = aggregate(test_frames, temperatures)
    raw_thresholds = {
        task: best_mcc_threshold(val[f"y_{task}"], val[f"p_{task}_mean"])
        for task in TASKS
    }
    calibrated_thresholds = {
        task: best_mcc_threshold(
            val[f"y_{task}"], val[f"p_{task}_calibrated"]
        )
        for task in TASKS
    }

    metrics = {
        "member_count": len(val_frames),
        "member_paths": [str(path) for path in paths],
        "temperature_application": "shared temperature applied to each member logit before probability averaging",
        "temperatures": temperatures,
        "raw_thresholds": raw_thresholds,
        "calibrated_thresholds": calibrated_thresholds,
        "logical_violations_raw": logical_violation_rates(test, "mean"),
        "logical_violations_calibrated": logical_violation_rates(test, "calibrated"),
    }
    for task in TASKS:
        metrics[f"val_{task}_raw"] = classification_metrics(
            val[f"y_{task}"],
            val[f"p_{task}_mean"],
            raw_thresholds[task],
            args.calibration_bins,
        )
        metrics[f"val_{task}_calibrated"] = classification_metrics(
            val[f"y_{task}"],
            val[f"p_{task}_calibrated"],
            calibrated_thresholds[task],
            args.calibration_bins,
        )
        metrics[f"test_{task}_raw"] = classification_metrics(
            test[f"y_{task}"],
            test[f"p_{task}_mean"],
            raw_thresholds[task],
            args.calibration_bins,
        )
        metrics[f"test_{task}_calibrated"] = classification_metrics(
            test[f"y_{task}"],
            test[f"p_{task}_calibrated"],
            calibrated_thresholds[task],
            args.calibration_bins,
        )

    val.to_csv(output_dir / "val_ensemble_predictions.csv", index=False)
    test.to_csv(output_dir / "test_ensemble_predictions.csv", index=False)
    save_json(temperatures, output_dir / "temperatures.json")
    save_json(raw_thresholds, output_dir / "raw_thresholds.json")
    # Downstream scripts use calibrated thresholds.
    save_json(calibrated_thresholds, output_dir / "thresholds.json")
    save_json(metrics, output_dir / "ensemble_metrics.json")
    ensemble_size_ablation(
        val_frames, test_frames, output_dir, args.calibration_bins
    )
    save_json(
        {"status": "completed", "member_count": len(val_frames)},
        output_dir / "COMPLETED.json",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
