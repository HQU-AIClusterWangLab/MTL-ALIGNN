from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from train.common import save_json
from train.metrics import (
    best_mcc_threshold,
    classification_metrics,
)


def fusion_features(frame: pd.DataFrame) -> np.ndarray:
    thermo = frame["p_thermo_calibrated"].to_numpy(float)
    dynamic = frame["p_dynamic_calibrated"].to_numpy(float)
    return np.column_stack(
        [
            thermo,
            dynamic,
            thermo * dynamic,
            np.minimum(thermo, dynamic),
            np.maximum(0.0, thermo + dynamic - 1.0),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-experiment-dir", required=True)
    parser.add_argument("--calibration-bins", type=int, default=15)
    args = parser.parse_args()

    root = Path(args.full_experiment_dir)
    output = root / "joint_fusion"
    output.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(root / "ensemble" / "val_ensemble_predictions.csv")
    test = pd.read_csv(root / "ensemble" / "test_ensemble_predictions.csv")
    labels_val = val["y_joint"].to_numpy(int)
    labels_test = test["y_joint"].to_numpy(int)

    val_scores = {
        "direct_joint": val["p_joint_calibrated"].to_numpy(float),
        "product_fusion": (
            val["p_thermo_calibrated"] * val["p_dynamic_calibrated"]
        ).to_numpy(float),
        "minimum_fusion": np.minimum(
            val["p_thermo_calibrated"],
            val["p_dynamic_calibrated"],
        ),
        "frechet_lower_bound": np.maximum(
            0.0,
            val["p_thermo_calibrated"]
            + val["p_dynamic_calibrated"]
            - 1.0,
        ),
    }
    test_scores = {
        "direct_joint": test["p_joint_calibrated"].to_numpy(float),
        "product_fusion": (
            test["p_thermo_calibrated"] * test["p_dynamic_calibrated"]
        ).to_numpy(float),
        "minimum_fusion": np.minimum(
            test["p_thermo_calibrated"],
            test["p_dynamic_calibrated"],
        ),
        "frechet_lower_bound": np.maximum(
            0.0,
            test["p_thermo_calibrated"]
            + test["p_dynamic_calibrated"]
            - 1.0,
        ),
    }

    # Learned fusion is trained only on the validation set. Its validation
    # threshold is chosen from out-of-fold predictions to avoid fitting and
    # thresholding on exactly the same predictions.
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    learned_val = cross_val_predict(
        estimator,
        fusion_features(val),
        labels_val,
        cv=folds,
        method="predict_proba",
    )[:, 1]
    estimator.fit(fusion_features(val), labels_val)
    learned_test = estimator.predict_proba(fusion_features(test))[:, 1]
    val_scores["learned_fusion"] = learned_val
    test_scores["learned_fusion"] = learned_test

    rows = []
    prediction_output = test[
        [
            "material_id",
            "formula",
            "ehull",
            "y_thermo",
            "y_dynamic",
            "y_joint",
        ]
    ].copy()
    thresholds = {}
    for method in val_scores:
        threshold = best_mcc_threshold(labels_val, val_scores[method])
        thresholds[method] = threshold
        metrics = classification_metrics(
            labels_test,
            test_scores[method],
            threshold,
            args.calibration_bins,
        )
        rows.append({"method": method, **metrics})
        prediction_output[f"score_{method}"] = test_scores[method]

    table = pd.DataFrame(rows).sort_values(
        ["pr_auc", "mcc"], ascending=False
    )
    table.to_csv(output / "joint_fusion_comparison.csv", index=False)
    prediction_output.to_csv(
        output / "test_joint_fusion_predictions.csv",
        index=False,
    )
    save_json(thresholds, output / "thresholds.json")
    save_json(
        {
            "status": "completed",
            "learned_fusion_training": "validation set",
            "learned_fusion_threshold": "5-fold out-of-fold validation predictions",
        },
        output / "COMPLETED.json",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
