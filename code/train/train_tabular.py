from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from train.common import (
    capture_environment,
    completed,
    experiment_result_root,
    load_yaml,
    resolve_experiment,
    save_json,
    save_yaml,
    set_seed,
)
from train.metrics import best_mcc_threshold, classification_metrics


TASKS = ["thermo", "dynamic", "joint"]


def build_estimator(
    model_type: str,
    seed: int,
    positive_weight: float,
    cfg: dict,
    class_weighting: str,
):
    if model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 800)),
            max_depth=cfg.get("max_depth"),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 1)),
            max_features=cfg.get("max_features", "sqrt"),
            class_weight=(
                "balanced_subsample" if class_weighting == "balanced" else None
            ),
            n_jobs=int(cfg.get("n_jobs", 32)),
            random_state=seed,
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=int(cfg.get("n_estimators", 600)),
            max_depth=int(cfg.get("max_depth", 6)),
            learning_rate=float(cfg.get("learning_rate", 0.03)),
            subsample=float(cfg.get("subsample", 0.9)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            min_child_weight=float(cfg.get("min_child_weight", 1.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(
                positive_weight if class_weighting == "balanced" else 1.0
            ),
            tree_method="hist",
            device="cuda",
            n_jobs=int(cfg.get("n_jobs", 16)),
            random_state=seed,
        )
    else:
        raise ValueError(model_type)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )


def prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
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
        "thermo_label",
        "dynamic_label",
        "joint_label",
    ]
    result = frame[[column for column in columns if column in frame]].copy()
    return result.rename(
        columns={
            "thermo_label": "y_thermo",
            "dynamic_label": "y_dynamic",
            "joint_label": "y_joint",
        }
    )


def train_seed(experiment: dict, seed: int, output_dir: Path, force: bool):
    if completed(output_dir) and not force:
        print(f"SKIP completed: {output_dir}")
        return load_yaml(output_dir / "summary.yaml")
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        # A failed forced rerun must never inherit the previous success marker.
        # The marker is recreated only after every output of this run is saved.
        completion_marker = output_dir / "COMPLETED.json"
        if completion_marker.exists():
            completion_marker.unlink()
    set_seed(seed)

    processed = Path(experiment["paths"]["processed_dir"])
    frame = pd.read_csv(processed / "supervised.csv")
    assignments = pd.read_csv(
        processed / experiment.get("split_file", "iid_split.csv")
    )
    required_assignment_columns = {"material_id", "split"}
    missing_assignment_columns = required_assignment_columns - set(assignments.columns)
    if missing_assignment_columns:
        raise KeyError(
            "Split manifest is missing required columns: "
            f"{sorted(missing_assignment_columns)}"
        )
    if assignments["material_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate material_id rows.")
    frame = frame.merge(
        assignments[["material_id", "split"]],
        on="material_id",
        validate="one_to_one",
    )
    import json
    feature_columns = json.loads(
        (processed / "tabular_feature_columns.json").read_text(encoding="utf-8")
    )
    feature_columns = list(feature_columns)
    forbidden = {"ehull", "thermo_label", "dynamic_label", "joint_label"}
    if forbidden.intersection(feature_columns):
        raise RuntimeError("Target leakage in tabular feature list.")

    heads = list(experiment.get("heads", TASKS))
    invalid_heads = set(heads) - set(TASKS)
    if invalid_heads or not heads:
        raise ValueError(f"Invalid tabular prediction heads: {sorted(invalid_heads)}")

    additional_features = list(experiment.get("additional_features", []))
    if "ehull" in additional_features and heads != ["dynamic"]:
        raise RuntimeError(
            "ehull may only be added to the deployment-matched Dynamic-only "
            "baseline; using it for Thermo or Joint would leak the target definition."
        )
    disallowed_additional = set(additional_features) - set(frame.columns)
    if disallowed_additional:
        raise KeyError(
            f"Unknown additional feature columns: {sorted(disallowed_additional)}"
        )
    feature_columns.extend(
        column for column in additional_features if column not in feature_columns
    )

    train = frame[frame["split"] == "train"].copy()
    val = frame[frame["split"] == "val"].copy()
    test = frame[frame["split"] == "test"].copy()
    x_train = train[feature_columns]
    x_val = val[feature_columns]
    x_test = test[feature_columns]

    val_predictions = prediction_frame(val)
    test_predictions = prediction_frame(test)
    thresholds = {}
    metrics = {
        "seed": seed,
        "model_type": experiment["model_type"],
        "heads": heads,
        "feature_columns": feature_columns,
        "additional_features": additional_features,
    }
    start = time.time()

    for task in heads:
        label_column = f"{task}_label"
        y_train = train[label_column].to_numpy(dtype=int)
        positives = float((y_train == 1).sum())
        negatives = float((y_train == 0).sum())
        estimator = build_estimator(
            experiment["model_type"],
            seed,
            negatives / max(positives, 1.0),
            experiment.get("tabular_model", {}),
            str(experiment["training"].get("class_weighting", "none")).lower(),
        )
        fell_back_to_cpu = False
        fallback_reason = None
        try:
            estimator.fit(x_train, y_train)
        except Exception as exc:
            # XGBoost CUDA is optional; fall back to CPU hist without changing
            # the scientific model.
            if experiment["model_type"] != "xgboost":
                raise
            classifier = estimator.named_steps["classifier"]
            classifier.set_params(device="cpu")
            estimator.fit(x_train, y_train)
            fell_back_to_cpu = True
            fallback_reason = repr(exc)

        fitted_classifier = estimator.named_steps["classifier"]
        metrics.setdefault("estimator_backend", {})[task] = {
            "class": type(fitted_classifier).__name__,
            "device": fitted_classifier.get_params().get("device"),
            "tree_method": fitted_classifier.get_params().get("tree_method"),
            "fit_fell_back_to_cpu": fell_back_to_cpu,
            "fallback_reason": fallback_reason,
        }

        val_probability = estimator.predict_proba(x_val)[:, 1]
        test_probability = estimator.predict_proba(x_test)[:, 1]
        threshold = best_mcc_threshold(
            val[f"{task}_label"],
            val_probability,
        )
        thresholds[task] = threshold
        val_predictions[f"p_{task}"] = val_probability
        test_predictions[f"p_{task}"] = test_probability
        val_predictions[f"logit_{task}"] = np.log(
            np.clip(val_probability, 1e-7, 1 - 1e-7)
            / np.clip(1 - val_probability, 1e-7, 1 - 1e-7)
        )
        test_predictions[f"logit_{task}"] = np.log(
            np.clip(test_probability, 1e-7, 1 - 1e-7)
            / np.clip(1 - test_probability, 1e-7, 1 - 1e-7)
        )
        val_predictions[f"pred_{task}"] = (
            val_probability >= threshold
        ).astype(int)
        test_predictions[f"pred_{task}"] = (
            test_probability >= threshold
        ).astype(int)
        metrics[f"val_{task}"] = classification_metrics(
            val[f"{task}_label"],
            val_probability,
            threshold,
            int(experiment["uncertainty"]["calibration_bins"]),
        )
        metrics[f"test_{task}"] = classification_metrics(
            test[f"{task}_label"],
            test_probability,
            threshold,
            int(experiment["uncertainty"]["calibration_bins"]),
        )
        joblib.dump(estimator, output_dir / f"model_{task}.joblib")
        classifier = estimator.named_steps["classifier"]
        importance = getattr(classifier, "feature_importances_", None)
        if importance is not None:
            pd.DataFrame(
                {"feature": feature_columns, "importance": importance}
            ).sort_values("importance", ascending=False).to_csv(
                output_dir / f"feature_importance_{task}.csv", index=False
            )

    metrics["runtime_seconds"] = time.time() - start
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    save_json(thresholds, output_dir / "thresholds.json")
    save_json(metrics, output_dir / "metrics.json")
    save_yaml(experiment, output_dir / "resolved_experiment.yaml")
    save_json(capture_environment(), output_dir / "environment.json")
    summary = {
        "name": experiment["name"],
        "category": experiment["category"],
        "seed": seed,
        "status": "completed",
        "monitor_task": str(experiment.get("monitor_task", heads[0])),
    }
    monitor_task = summary["monitor_task"]
    if monitor_task not in heads:
        raise ValueError(
            f"monitor_task={monitor_task!r} is not one of the trained heads {heads}."
        )
    summary[f"test_{monitor_task}_pr_auc"] = metrics[f"test_{monitor_task}"]["pr_auc"]
    summary[f"test_{monitor_task}_mcc"] = metrics[f"test_{monitor_task}"]["mcc"]
    save_yaml(summary, output_dir / "summary.yaml")
    save_json(summary, output_dir / "COMPLETED.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project = load_yaml("config.yaml")
    exp_file = load_yaml(args.experiment)
    exp_file["experiment_file"] = args.experiment
    experiment = resolve_experiment(project, exp_file)
    output_root = experiment_result_root(experiment)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed is not None else experiment["seeds"]
    summaries = [
        train_seed(
            experiment,
            int(seed),
            output_root / f"seed_{seed}",
            args.force,
        )
        for seed in seeds
    ]
    save_json(summaries, output_root / "all_seed_summaries.json")


if __name__ == "__main__":
    main()
