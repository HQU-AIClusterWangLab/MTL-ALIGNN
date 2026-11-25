"""Aggregate the five retuned single-task ALIGNN predictions without model inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train.aggregate_ensemble import fit_temperature, load_members, member_logits
from train.common import save_json
from train.metrics import best_mcc_threshold, classification_metrics, sigmoid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    val_frames, val_paths = load_members(root, "val")
    test_frames, test_paths = load_members(root, "test")
    if len(val_frames) != 5 or len(test_frames) != 5:
        raise ValueError("The single-task ensemble requires exactly five members")
    expected = {f"seed_{seed}" for seed in range(2025, 2030)}
    if {path.parent.name for path in val_paths + test_paths} != expected:
        raise ValueError("Unexpected single-task ensemble seed set")

    val_logits = member_logits(val_frames, "joint")
    val_y = val_frames[0]["y_joint"].to_numpy(dtype=int)
    temperature = fit_temperature(val_y, val_logits)
    predictions = {}
    for split, frames in (("val", val_frames), ("test", test_frames)):
        reference = frames[0]
        if any(
            not np.array_equal(reference["y_joint"], frame["y_joint"])
            for frame in frames[1:]
        ):
            raise ValueError(f"{split}: inconsistent Joint labels across members")
        logits = member_logits(frames, "joint")
        raw_members = np.vstack(
            [frame["p_joint"].to_numpy(dtype=float) for frame in frames]
        )
        predictions[split] = pd.DataFrame(
            {
                "material_id": reference["material_id"],
                "y_joint": reference["y_joint"],
                "p_joint_mean": raw_members.mean(axis=0),
                "p_joint_std": raw_members.std(axis=0, ddof=1),
                "p_joint_calibrated": sigmoid(logits / temperature).mean(axis=0),
            }
        )

    val = predictions["val"]
    thresholds = {
        "raw": best_mcc_threshold(val["y_joint"], val["p_joint_mean"]),
        "calibrated": best_mcc_threshold(
            val["y_joint"], val["p_joint_calibrated"]
        ),
    }
    metrics = {"member_count": 5, "temperature_joint": temperature, "thresholds": thresholds}
    output = root / "ensemble"
    output.mkdir(parents=True, exist_ok=True)
    for split, frame in predictions.items():
        frame.to_csv(output / f"{split}_ensemble_predictions.csv", index=False)
        for scale, column in (
            ("raw", "p_joint_mean"),
            ("calibrated", "p_joint_calibrated"),
        ):
            metrics[f"{split}_joint_{scale}"] = classification_metrics(
                frame["y_joint"], frame[column], thresholds[scale], 15
            )
    save_json(metrics, output / "ensemble_metrics.json")
    print(output / "ensemble_metrics.json")


if __name__ == "__main__":
    main()
