from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


COMMON_SEEDS = (2025, 2026, 2027)
ENSEMBLE_SEEDS = (2025, 2026, 2027, 2028, 2029)
ENSEMBLE_SIZES = (1, 3, 5)
EXPECTED_TEST_COUNT = 1214
PROBABILITY_LOGIT_TOLERANCE = 1e-6
METRIC_TOLERANCE = 1e-10
REFERENCE_TOLERANCE = 1e-10

MODEL_SPECS = (
    ("Single ALIGNN", "Single", "M04_single_alignn"),
    ("Equal MTL", "Equal MTL", "M05_equal_mtl"),
    ("Full MTL", "Full MTL", "M06_full_mtl"),
)

DETAIL_COLUMNS = [
    "ensemble_size",
    "subset_id",
    "member_seeds",
    "joint_ap",
    "roc_auc",
    "nll",
    "brier",
    "n_test",
    "calibrated",
    "aggregation_method",
    "source_prediction_files",
]

SUMMARY_COLUMNS = [
    "ensemble_size",
    "n_subsets",
    "joint_ap_mean",
    "joint_ap_sd",
    "roc_auc_mean",
    "roc_auc_sd",
    "nll_mean",
    "nll_sd",
    "brier_mean",
    "brier_sd",
    "n_test",
    "calibrated",
    "member_pool_seeds",
    "aggregation_method",
]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the uncalibrated all-subset ensemble-size ablation and "
            "generate the two-panel Fig. 3 from frozen result files."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="Project root containing results/, scripts/, and data/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory (default: <project-root>/paper_figures/fig3).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, na_rep="NA")
    os.replace(temporary, path)


def atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise AssertionError(f"Missing columns in {source}: {missing}")


def extract_split_file(resolved_experiment: Path) -> str:
    matches = []
    for line in resolved_experiment.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("split_file:"):
            matches.append(stripped.split(":", 1)[1].strip().strip("'\""))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one split_file in {resolved_experiment}; found {matches}"
        )
    return matches[0]


def load_joint_predictions(path: Path, expected_count: int) -> tuple[pd.DataFrame, float]:
    frame = pd.read_csv(path)
    require_columns(frame, {"material_id", "y_joint", "p_joint", "logit_joint"}, path)
    if len(frame) != expected_count:
        raise AssertionError(
            f"Unexpected test count in {path}: {len(frame)} != {expected_count}"
        )
    if frame["material_id"].isna().any():
        raise AssertionError(f"Missing material_id in {path}")
    if frame["material_id"].duplicated().any():
        duplicates = frame.loc[frame["material_id"].duplicated(), "material_id"].tolist()
        raise AssertionError(f"Duplicate material_id in {path}: {duplicates[:5]}")
    calibrated_columns = [column for column in frame.columns if "calibrat" in column.lower()]
    if calibrated_columns:
        raise AssertionError(
            f"Per-seed frozen predictions unexpectedly contain calibrated columns in {path}: "
            f"{calibrated_columns}"
        )

    compact = frame.loc[:, ["material_id", "y_joint", "p_joint", "logit_joint"]].copy()
    compact["material_id"] = compact["material_id"].astype(str)
    compact["y_joint"] = pd.to_numeric(compact["y_joint"], errors="raise").astype(int)
    compact["p_joint"] = pd.to_numeric(compact["p_joint"], errors="raise").astype(float)
    compact["logit_joint"] = pd.to_numeric(
        compact["logit_joint"], errors="raise"
    ).astype(float)
    if not set(compact["y_joint"].unique()).issubset({0, 1}):
        raise AssertionError(f"Non-binary y_joint in {path}")
    if not np.isfinite(compact[["p_joint", "logit_joint"]].to_numpy(float)).all():
        raise AssertionError(f"Non-finite probability/logit in {path}")
    if not compact["p_joint"].between(0.0, 1.0, inclusive="both").all():
        raise AssertionError(f"p_joint outside [0, 1] in {path}")

    max_abs_diff = float(
        np.max(
            np.abs(
                compact["p_joint"].to_numpy(float)
                - sigmoid(compact["logit_joint"].to_numpy(float))
            )
        )
    )
    if max_abs_diff > PROBABILITY_LOGIT_TOLERANCE:
        raise AssertionError(
            f"p_joint is inconsistent with sigmoid(logit_joint) in {path}: "
            f"max_abs_diff={max_abs_diff:.12g}"
        )

    compact = compact.sort_values("material_id", kind="mergesort").reset_index(drop=True)
    return compact, max_abs_diff


def load_panel_a(project_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    results_root = project_root / "results" / "main"
    rows: list[dict[str, object]] = []
    prediction_audit: list[dict[str, object]] = []
    split_names: dict[str, str] = {}
    reference_ids: list[str] | None = None
    reference_labels: np.ndarray | None = None

    for model_name, tick_label, directory_name in MODEL_SPECS:
        model_dir = results_root / directory_name
        split_name = extract_split_file(model_dir / "resolved_experiment.yaml")
        split_names[model_name] = split_name
        used_seeds: list[int] = []
        for seed in COMMON_SEEDS:
            metrics_path = model_dir / f"seed_{seed}" / "metrics.json"
            completed_path = model_dir / f"seed_{seed}" / "COMPLETED.json"
            predictions_path = model_dir / f"seed_{seed}" / "test_predictions.csv"
            for required_path in (metrics_path, completed_path, predictions_path):
                if not required_path.is_file():
                    raise AssertionError(f"Required Panel A source is missing: {required_path}")

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
            if int(metrics.get("seed", -1)) != seed:
                raise AssertionError(f"Seed mismatch in {metrics_path}")
            if completed.get("status") != "completed":
                raise AssertionError(f"Run is not completed: {completed_path}")
            test_joint = metrics.get("test_joint")
            if not isinstance(test_joint, dict):
                raise AssertionError(f"Missing test_joint block in {metrics_path}")
            for field in ("pr_auc", "nll", "brier", "roc_auc"):
                if field not in test_joint:
                    raise AssertionError(f"Missing test_joint.{field} in {metrics_path}")
            joint_ap = float(test_joint["pr_auc"])
            if not 0.0 <= joint_ap <= 1.0:
                raise AssertionError(f"Joint AP outside [0,1] in {metrics_path}")
            completed_ap = float(completed["test_joint_pr_auc"])
            if not math.isclose(joint_ap, completed_ap, rel_tol=0.0, abs_tol=1e-14):
                raise AssertionError(
                    f"metrics/COMPLETED AP conflict for {model_name}, seed {seed}"
                )

            predictions, max_sigmoid_diff = load_joint_predictions(
                predictions_path, EXPECTED_TEST_COUNT
            )
            ids = predictions["material_id"].tolist()
            labels = predictions["y_joint"].to_numpy(int)
            if reference_ids is None:
                reference_ids = ids
                reference_labels = labels
            else:
                if ids != reference_ids:
                    raise AssertionError(
                        f"Panel A material IDs differ for {model_name}, seed {seed}"
                    )
                if not np.array_equal(labels, reference_labels):
                    raise AssertionError(
                        f"Panel A y_joint differs for {model_name}, seed {seed}"
                    )
            recomputed_ap = float(
                average_precision_score(labels, predictions["p_joint"].to_numpy(float))
            )
            if not math.isclose(
                recomputed_ap, joint_ap, rel_tol=0.0, abs_tol=METRIC_TOLERANCE
            ):
                raise AssertionError(
                    f"Panel A AP does not reproduce from frozen predictions for "
                    f"{model_name}, seed {seed}: {recomputed_ap} vs {joint_ap}"
                )

            rows.append(
                {
                    "model": model_name,
                    "tick_label": tick_label,
                    "seed": seed,
                    "joint_ap": joint_ap,
                    "nll": float(test_joint["nll"]),
                    "brier": float(test_joint["brier"]),
                    "roc_auc": float(test_joint["roc_auc"]),
                    "metrics_path": str(metrics_path.resolve()),
                    "predictions_path": str(predictions_path.resolve()),
                    "metrics_sha256": sha256_file(metrics_path),
                    "predictions_sha256": sha256_file(predictions_path),
                }
            )
            prediction_audit.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "max_abs_p_minus_sigmoid_logit": max_sigmoid_diff,
                    "recomputed_ap": recomputed_ap,
                }
            )
            used_seeds.append(seed)
        if tuple(used_seeds) != COMMON_SEEDS:
            raise AssertionError(f"Panel A did not use exactly common seeds for {model_name}")

    if set(split_names.values()) != {"iid_split.csv"}:
        raise AssertionError(f"Panel A experiments do not share iid_split.csv: {split_names}")
    if reference_ids is None or reference_labels is None:
        raise AssertionError("Panel A reference test set was not established")

    panel_a = pd.DataFrame(rows)
    counts = panel_a.groupby("model", sort=False)["seed"].agg(list)
    for model_name, seeds in counts.items():
        if tuple(seeds) != COMMON_SEEDS:
            raise AssertionError(f"Unexpected Panel A seeds for {model_name}: {seeds}")

    id_hash = sha256_lines(reference_ids)
    id_label_hash = sha256_lines(
        [f"{material_id}\t{int(label)}" for material_id, label in zip(reference_ids, reference_labels)]
    )
    audit = {
        "split_names": split_names,
        "n_test": len(reference_ids),
        "n_joint_positive": int(reference_labels.sum()),
        "sorted_material_id_sha256": id_hash,
        "sorted_material_id_label_sha256": id_label_hash,
        "prediction_checks": prediction_audit,
    }
    return panel_a, audit


def load_ensemble_members(
    project_root: Path,
) -> tuple[dict[int, pd.DataFrame], dict[str, object]]:
    full_dir = project_root / "results" / "main" / "M06_full_mtl"
    split_path = project_root / "data" / "processed" / "iid_split.csv"
    if not split_path.is_file():
        raise AssertionError(f"Missing frozen split manifest: {split_path}")
    split = pd.read_csv(split_path)
    require_columns(split, {"material_id", "split"}, split_path)
    split_test = split.loc[split["split"].astype(str).str.lower() == "test", "material_id"]
    split_ids = sorted(split_test.astype(str).tolist())
    if len(split_ids) != EXPECTED_TEST_COUNT or len(set(split_ids)) != EXPECTED_TEST_COUNT:
        raise AssertionError(
            f"Frozen test split does not contain {EXPECTED_TEST_COUNT} unique material IDs"
        )

    frames: dict[int, pd.DataFrame] = {}
    source_paths: dict[int, Path] = {}
    completed_paths: dict[int, Path] = {}
    probability_logit_diffs: dict[int, float] = {}
    reference_ids: list[str] | None = None
    reference_labels: np.ndarray | None = None
    for seed in ENSEMBLE_SEEDS:
        path = full_dir / f"seed_{seed}" / "test_predictions.csv"
        completed_path = full_dir / f"seed_{seed}" / "COMPLETED.json"
        if not path.is_file():
            raise AssertionError(f"Missing frozen ensemble member: {path}")
        if not completed_path.is_file():
            raise AssertionError(f"Missing completion record: {completed_path}")
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("status") != "completed" or int(completed.get("seed", -1)) != seed:
            raise AssertionError(f"Invalid completion record: {completed_path}")
        frame, max_abs_diff = load_joint_predictions(path, EXPECTED_TEST_COUNT)
        ids = frame["material_id"].tolist()
        labels = frame["y_joint"].to_numpy(int)
        if ids != split_ids:
            raise AssertionError(f"Member seed {seed} does not match iid_split.csv test IDs")
        if reference_ids is None:
            reference_ids = ids
            reference_labels = labels
        else:
            if ids != reference_ids:
                raise AssertionError(f"Member seed {seed} has a different material-ID set")
            if not np.array_equal(labels, reference_labels):
                raise AssertionError(f"Member seed {seed} has different y_joint labels")
        frames[seed] = frame
        source_paths[seed] = path.resolve()
        completed_paths[seed] = completed_path.resolve()
        probability_logit_diffs[seed] = max_abs_diff

    if tuple(frames) != ENSEMBLE_SEEDS:
        raise AssertionError(f"Unexpected ensemble member seeds: {tuple(frames)}")
    if reference_ids is None or reference_labels is None:
        raise AssertionError("Ensemble reference test set was not established")

    ensemble_dir = full_dir / "ensemble"
    ensemble_metrics_path = ensemble_dir / "ensemble_metrics.json"
    ensemble_predictions_path = ensemble_dir / "test_ensemble_predictions.csv"
    if not ensemble_metrics_path.is_file() or not ensemble_predictions_path.is_file():
        raise AssertionError("Missing existing five-member ensemble audit source")

    ensemble_metrics = json.loads(ensemble_metrics_path.read_text(encoding="utf-8"))
    member_paths = ensemble_metrics.get("member_paths", [])
    member_seed_order = tuple(int(Path(path).parent.name.split("_")[-1]) for path in member_paths)
    if member_seed_order != ENSEMBLE_SEEDS:
        raise AssertionError(
            f"ensemble_metrics member order differs from expected seeds: {member_seed_order}"
        )

    audit = {
        "split_path": str(split_path.resolve()),
        "split_sha256": sha256_file(split_path),
        "n_test": len(reference_ids),
        "n_joint_positive": int(reference_labels.sum()),
        "sorted_material_id_sha256": sha256_lines(reference_ids),
        "sorted_material_id_label_sha256": sha256_lines(
            [
                f"{material_id}\t{int(label)}"
                for material_id, label in zip(reference_ids, reference_labels)
            ]
        ),
        "source_paths": {seed: str(path) for seed, path in source_paths.items()},
        "source_sha256": {
            seed: sha256_file(path) for seed, path in source_paths.items()
        },
        "completed_paths": {
            seed: str(path) for seed, path in completed_paths.items()
        },
        "completed_sha256": {
            seed: sha256_file(path) for seed, path in completed_paths.items()
        },
        "probability_logit_max_abs_diff": probability_logit_diffs,
        "ensemble_metrics_path": str(ensemble_metrics_path.resolve()),
        "ensemble_metrics_sha256": sha256_file(ensemble_metrics_path),
        "ensemble_predictions_path": str(ensemble_predictions_path.resolve()),
        "ensemble_predictions_sha256": sha256_file(ensemble_predictions_path),
        "existing_raw_five_member_metrics": ensemble_metrics["test_joint_raw"],
    }
    return frames, audit


def compute_raw_ablation(
    project_root: Path,
    frames: dict[int, pd.DataFrame],
    binary_nll,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    labels = frames[ENSEMBLE_SEEDS[0]]["y_joint"].to_numpy(int)
    seed_to_row = {seed: index for index, seed in enumerate(ENSEMBLE_SEEDS)}
    probability_matrix = np.vstack(
        [frames[seed]["p_joint"].to_numpy(float) for seed in ENSEMBLE_SEEDS]
    )
    source_paths = {
        seed: (
            project_root
            / "results"
            / "main"
            / "M06_full_mtl"
            / f"seed_{seed}"
            / "test_predictions.csv"
        ).resolve()
        for seed in ENSEMBLE_SEEDS
    }
    aggregation_method = (
        "Material-ID-aligned arithmetic mean of stored uncalibrated member p_joint "
        "probabilities in probability space; size-level mean over all C(5,M) subsets"
    )
    detail_rows: list[dict[str, object]] = []

    for ensemble_size in ENSEMBLE_SIZES:
        combinations = list(itertools.combinations(ENSEMBLE_SEEDS, ensemble_size))
        expected_subsets = math.comb(len(ENSEMBLE_SEEDS), ensemble_size)
        if len(combinations) != expected_subsets:
            raise AssertionError(
                f"Subset enumeration failed for M={ensemble_size}: "
                f"{len(combinations)} != {expected_subsets}"
            )
        for subset_id, member_seeds in enumerate(combinations, start=1):
            indices = [seed_to_row[seed] for seed in member_seeds]
            raw_probability = probability_matrix[indices].mean(axis=0, dtype=np.float64)
            if not np.isfinite(raw_probability).all():
                raise AssertionError(f"Non-finite raw probability for subset {member_seeds}")
            if np.min(raw_probability) < 0.0 or np.max(raw_probability) > 1.0:
                raise AssertionError(f"Raw probability outside [0,1] for {member_seeds}")
            detail_rows.append(
                {
                    "ensemble_size": ensemble_size,
                    "subset_id": subset_id,
                    "member_seeds": ";".join(str(seed) for seed in member_seeds),
                    "joint_ap": float(average_precision_score(labels, raw_probability)),
                    "roc_auc": float(roc_auc_score(labels, raw_probability)),
                    "nll": float(binary_nll(labels, raw_probability)),
                    "brier": float(brier_score_loss(labels, raw_probability)),
                    "n_test": len(labels),
                    "calibrated": False,
                    "aggregation_method": aggregation_method,
                    "source_prediction_files": ";".join(
                        str(source_paths[seed]) for seed in member_seeds
                    ),
                }
            )

    detail = pd.DataFrame(detail_rows, columns=DETAIL_COLUMNS)
    if detail["calibrated"].astype(bool).any():
        raise AssertionError("Raw detail unexpectedly contains calibrated=True")
    actual_sizes = tuple(sorted(detail["ensemble_size"].unique().tolist()))
    if actual_sizes != ENSEMBLE_SIZES:
        raise AssertionError(f"Unexpected ensemble sizes: {actual_sizes}")
    expected_counts = {1: 5, 3: 10, 5: 1}
    actual_counts = detail.groupby("ensemble_size").size().to_dict()
    if actual_counts != expected_counts:
        raise AssertionError(f"Unexpected subset counts: {actual_counts}")

    summary_rows: list[dict[str, object]] = []
    for ensemble_size in ENSEMBLE_SIZES:
        group = detail.loc[detail["ensemble_size"] == ensemble_size]
        n_subsets = len(group)
        row: dict[str, object] = {
            "ensemble_size": ensemble_size,
            "n_subsets": n_subsets,
            "n_test": int(group["n_test"].iloc[0]),
            "calibrated": False,
            "member_pool_seeds": ";".join(str(seed) for seed in ENSEMBLE_SEEDS),
            "aggregation_method": aggregation_method,
        }
        for detail_column, summary_prefix in (
            ("joint_ap", "joint_ap"),
            ("roc_auc", "roc_auc"),
            ("nll", "nll"),
            ("brier", "brier"),
        ):
            values = group[detail_column].to_numpy(float)
            row[f"{summary_prefix}_mean"] = float(np.mean(values))
            row[f"{summary_prefix}_sd"] = (
                float(np.std(values, ddof=1)) if n_subsets > 1 else math.nan
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    if summary["calibrated"].astype(bool).any():
        raise AssertionError("Raw summary unexpectedly contains calibrated=True")
    if not summary.loc[summary["ensemble_size"] == 5, "joint_ap_sd"].isna().all():
        raise AssertionError("M=5 AP SD must be NA")
    if not summary.loc[summary["ensemble_size"] == 5, "nll_sd"].isna().all():
        raise AssertionError("M=5 NLL SD must be NA")
    if not summary.loc[summary["ensemble_size"] == 5, "brier_sd"].isna().all():
        raise AssertionError("M=5 Brier SD must be NA")

    singleton = detail.loc[detail["ensemble_size"] == 1].copy()
    singleton["seed"] = singleton["member_seeds"].astype(int)
    for _, row in singleton.iterrows():
        seed = int(row["seed"])
        metrics_path = (
            project_root
            / "results"
            / "main"
            / "M06_full_mtl"
            / f"seed_{seed}"
            / "metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        frozen_ap = float(metrics["test_joint"]["pr_auc"])
        if not math.isclose(
            float(row["joint_ap"]), frozen_ap, rel_tol=0.0, abs_tol=METRIC_TOLERANCE
        ):
            raise AssertionError(
                f"Raw singleton AP fails to reproduce seed {seed}: "
                f"{row['joint_ap']} vs {frozen_ap}"
            )

    ensemble_dir = project_root / "results" / "main" / "M06_full_mtl" / "ensemble"
    metrics_path = ensemble_dir / "ensemble_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    reference = metrics["test_joint_raw"]
    five = detail.loc[detail["ensemble_size"] == 5].iloc[0]
    differences = {
        "joint_ap": float(five["joint_ap"]) - float(reference["pr_auc"]),
        "nll": float(five["nll"]) - float(reference["nll"]),
        "brier": float(five["brier"]) - float(reference["brier"]),
        "roc_auc": float(five["roc_auc"]) - float(reference["roc_auc"]),
    }
    for metric, difference in differences.items():
        if abs(difference) > REFERENCE_TOLERANCE:
            raise AssertionError(
                f"Raw M=5 cross-check failed for {metric}: difference={difference:.12g}"
            )

    existing = pd.read_csv(
        ensemble_dir / "test_ensemble_predictions.csv",
        usecols=["material_id", "y_joint", "p_joint_mean"],
    )
    if existing["material_id"].duplicated().any():
        raise AssertionError("Duplicate material_id in test_ensemble_predictions.csv")
    existing["material_id"] = existing["material_id"].astype(str)
    existing = existing.sort_values("material_id", kind="mergesort").reset_index(drop=True)
    expected_ids = frames[ENSEMBLE_SEEDS[0]]["material_id"].tolist()
    if existing["material_id"].tolist() != expected_ids:
        raise AssertionError("Existing ensemble predictions have different material IDs")
    five_probability = probability_matrix.mean(axis=0, dtype=np.float64)
    max_probability_difference = float(
        np.max(np.abs(existing["p_joint_mean"].to_numpy(float) - five_probability))
    )
    if max_probability_difference > 1e-12:
        raise AssertionError(
            "Raw M=5 probabilities do not reproduce p_joint_mean: "
            f"max_abs_diff={max_probability_difference:.12g}"
        )

    crosscheck = {
        "reference_path": str(metrics_path.resolve()),
        "reference_values": {
            "joint_ap": float(reference["pr_auc"]),
            "nll": float(reference["nll"]),
            "brier": float(reference["brier"]),
            "roc_auc": float(reference["roc_auc"]),
        },
        "recomputed_values": {
            "joint_ap": float(five["joint_ap"]),
            "nll": float(five["nll"]),
            "brier": float(five["brier"]),
            "roc_auc": float(five["roc_auc"]),
        },
        "differences": differences,
        "max_abs_probability_difference_vs_existing_p_joint_mean": max_probability_difference,
    }
    return detail, summary, crosscheck


def print_terminal_audit(
    panel_a: pd.DataFrame,
    raw_detail: pd.DataFrame,
    raw_summary: pd.DataFrame,
    crosscheck: dict[str, object],
) -> None:
    print("================ FIG. 3 DATA AUDIT (RAW) ================")
    print("[Panel A]")
    for model_name, _, _ in MODEL_SPECS:
        group = panel_a.loc[panel_a["model"] == model_name].sort_values("seed")
        print(model_name)
        for row in group.itertuples(index=False):
            print(f"  seed {row.seed}: {row.joint_ap:.15f}")
            print(f"    source: {row.metrics_path} -> test_joint.pr_auc")
        values = group["joint_ap"].to_numpy(float)
        print(f"  mean: {np.mean(values):.15f}")
        print(f"  sample SD (ddof=1): {np.std(values, ddof=1):.15f}")
    print("  common seeds: [2025, 2026, 2027] exactly")
    print()
    print("[Panel B - uncalibrated probability-space ensembles]")
    for ensemble_size in ENSEMBLE_SIZES:
        group = raw_detail.loc[raw_detail["ensemble_size"] == ensemble_size]
        print(f"M={ensemble_size}")
        for row in group.itertuples(index=False):
            print(
                f"  subset {row.subset_id:02d} [{row.member_seeds}]: "
                f"AP={row.joint_ap:.15f}, ROC-AUC={row.roc_auc:.15f}, "
                f"NLL={row.nll:.15f}, Brier={row.brier:.15f}"
            )
        summary_row = raw_summary.loc[
            raw_summary["ensemble_size"] == ensemble_size
        ].iloc[0]
        sd_text = (
            f"{summary_row['joint_ap_sd']:.15f}"
            if pd.notna(summary_row["joint_ap_sd"])
            else "NA"
        )
        print(
            f"  mean AP={summary_row['joint_ap_mean']:.15f}; "
            f"sample SD={sd_text}; n_subsets={int(summary_row['n_subsets'])}"
        )
        print(
            f"  mean NLL={summary_row['nll_mean']:.15f}; "
            f"mean Brier={summary_row['brier_mean']:.15f}"
        )
    print()
    print("[Raw M=5 cross-check]")
    for metric in ("joint_ap", "nll", "brier", "roc_auc"):
        reference = crosscheck["reference_values"][metric]
        recomputed = crosscheck["recomputed_values"][metric]
        difference = crosscheck["differences"][metric]
        print(
            f"  {metric}: recomputed={recomputed:.15f}, "
            f"reference={reference:.15f}, difference={difference:.3e}"
        )
    print("  status: PASS")
    print("===========================================================")


def write_raw_ablation_sources(
    raw_detail: pd.DataFrame,
    raw_summary: pd.DataFrame,
    raw_detail_path: Path,
    raw_summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    atomic_to_csv(raw_detail, raw_detail_path)
    atomic_to_csv(raw_summary, raw_summary_path)
    detail_readback = pd.read_csv(raw_detail_path)
    summary_readback = pd.read_csv(raw_summary_path)
    require_columns(detail_readback, set(DETAIL_COLUMNS), raw_detail_path)
    require_columns(summary_readback, set(SUMMARY_COLUMNS), raw_summary_path)
    if detail_readback["calibrated"].astype(bool).any():
        raise AssertionError("Raw detail readback contains calibrated=True")
    if summary_readback["calibrated"].astype(bool).any():
        raise AssertionError("Raw summary readback contains calibrated=True")
    if tuple(sorted(summary_readback["ensemble_size"].astype(int).tolist())) != ENSEMBLE_SIZES:
        raise AssertionError("Raw summary readback does not contain sizes 1, 3, 5")
    return detail_readback, summary_readback


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def padded_limits(values: np.ndarray, top_fraction: float = 0.16) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = max(upper - lower, 0.01)
    return max(0.0, lower - 0.11 * span), min(1.0, upper + top_fraction * span)


def create_figure(
    panel_a: pd.DataFrame,
    raw_detail: pd.DataFrame,
    raw_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    configure_matplotlib()
    color_mean = "#24577A"
    color_points = "#82A9C0"
    color_seed = "#66717B"
    color_pair = "#C9CED3"
    color_grid = "#E7EAED"

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.22, top=0.82, wspace=0.34)

    ax = axes[0]
    x_positions = np.arange(len(MODEL_SPECS), dtype=float)
    panel_a_lookup = {
        (row.model, int(row.seed)): float(row.joint_ap)
        for row in panel_a.itertuples(index=False)
    }
    for seed in COMMON_SEEDS:
        seed_values = np.asarray(
            [panel_a_lookup[(model_name, seed)] for model_name, _, _ in MODEL_SPECS],
            dtype=float,
        )
        ax.plot(
            x_positions,
            seed_values,
            color=color_pair,
            linewidth=0.7,
            alpha=0.85,
            zorder=1,
        )
        ax.scatter(
            x_positions,
            seed_values,
            s=22,
            color=color_seed,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            zorder=3,
        )

    panel_a_means = []
    panel_a_sds = []
    panel_a_domain_values = panel_a["joint_ap"].to_numpy(float).tolist()
    for model_name, _, _ in MODEL_SPECS:
        values = panel_a.loc[panel_a["model"] == model_name, "joint_ap"].to_numpy(float)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        panel_a_means.append(mean)
        panel_a_sds.append(sd)
        panel_a_domain_values.extend([mean - sd, mean + sd])
    ax.errorbar(
        x_positions,
        panel_a_means,
        yerr=panel_a_sds,
        fmt="D",
        markersize=6.4,
        markerfacecolor=color_mean,
        markeredgecolor="white",
        markeredgewidth=0.7,
        ecolor=color_mean,
        elinewidth=1.25,
        capsize=3.2,
        capthick=1.1,
        linestyle="none",
        zorder=5,
    )
    ax.set_xticks(x_positions, [spec[1] for spec in MODEL_SPECS])
    ax.set_xlim(-0.42, 2.42)
    ax.set_ylim(*padded_limits(np.asarray(panel_a_domain_values), top_fraction=0.12))
    ax.set_ylabel("Joint average precision")
    ax.set_title("Task organization", pad=8)
    ax.text(
        0.03,
        0.965,
        "Common seeds, n = 3",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color="#4C5359",
    )
    ax.text(
        -0.17,
        1.10,
        "a",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[1]
    summary_sorted = raw_summary.sort_values("ensemble_size")
    sizes = summary_sorted["ensemble_size"].to_numpy(int)
    means = summary_sorted["joint_ap_mean"].to_numpy(float)
    sds = summary_sorted["joint_ap_sd"].to_numpy(float)
    panel_b_domain_values: list[float] = raw_detail["joint_ap"].to_numpy(float).tolist()
    for size, mean, sd in zip(sizes, means, sds):
        group = raw_detail.loc[raw_detail["ensemble_size"] == size].sort_values("subset_id")
        values = group["joint_ap"].to_numpy(float)
        if len(values) > 1:
            jitter_width = 0.12 if len(values) <= 5 else 0.17
            offsets = np.linspace(-jitter_width, jitter_width, len(values))
            ax.scatter(
                np.full(len(values), float(size)) + offsets,
                values,
                s=19,
                color=color_points,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.72,
                zorder=2,
            )
        if np.isfinite(sd):
            panel_b_domain_values.extend([mean - sd, mean + sd])

    ax.plot(
        sizes,
        means,
        color=color_mean,
        linewidth=1.45,
        marker="o",
        markersize=6.2,
        markerfacecolor=color_mean,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=5,
    )
    finite_sd = np.isfinite(sds)
    ax.errorbar(
        sizes[finite_sd],
        means[finite_sd],
        yerr=sds[finite_sd],
        fmt="none",
        ecolor=color_mean,
        elinewidth=1.2,
        capsize=3.2,
        capthick=1.05,
        zorder=4,
    )
    low_b, high_b = padded_limits(np.asarray(panel_b_domain_values), top_fraction=0.22)
    ax.set_ylim(low_b, high_b)
    label_offset = 0.025 * (high_b - low_b)
    for size, mean, sd in zip(sizes, means, sds):
        label_y = float(mean) + (float(sd) if np.isfinite(sd) else 0.0) + label_offset
        ax.text(
            float(size),
            label_y,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#2E3338",
            zorder=6,
        )
    ax.set_xticks([1, 3, 5])
    ax.set_xlim(0.55, 5.45)
    ax.set_xlabel("Number of ensemble members")
    ax.set_ylabel("Joint average precision")
    ax.set_title("Ensemble size", pad=8)
    ax.text(
        -0.17,
        1.10,
        "b",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#4E555B")
        axis.spines["bottom"].set_color("#4E555B")
        axis.yaxis.grid(True, color=color_grid, linewidth=0.55, zorder=0)
        axis.xaxis.grid(False)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.tick_params(pad=3)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_bbox = figure.bbox
    for text_artist in figure.findobj(match=mpl.text.Text):
        if not text_artist.get_visible() or not text_artist.get_text().strip():
            continue
        bbox = text_artist.get_window_extent(renderer=renderer)
        if (
            bbox.x0 < figure_bbox.x0 - 2
            or bbox.y0 < figure_bbox.y0 - 2
            or bbox.x1 > figure_bbox.x1 + 2
            or bbox.y1 > figure_bbox.y1 + 2
        ):
            raise AssertionError(f"Figure text is clipped: {text_artist.get_text()!r}")

    paths = {
        "pdf": output_dir / "Fig3_task_ensemble.pdf",
        "png": output_dir / "Fig3_task_ensemble.png",
        "svg": output_dir / "Fig3_task_ensemble.svg",
    }
    figure.savefig(
        paths["pdf"],
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Title": "Fig. 3 - Task organization and raw ensemble size",
            "Creator": "plot_fig3_task_ensemble.py",
        },
    )
    figure.savefig(
        paths["png"],
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    figure.savefig(
        paths["svg"],
        format="svg",
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(figure)

    for path in paths.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"Figure output was not created correctly: {path}")
    with Image.open(paths["png"]) as image:
        dpi = image.info.get("dpi")
        if dpi is None or any(abs(float(value) - 600.0) > 1.0 for value in dpi[:2]):
            raise AssertionError(f"PNG does not report 600 dpi: {dpi}")
    return paths


def build_source_data(
    panel_a: pd.DataFrame,
    raw_detail: pd.DataFrame,
    raw_summary: pd.DataFrame,
    raw_detail_path: Path,
    raw_summary_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_method_a = "Common-seed comparison; mean and sample SD use seeds 2025-2027"
    for row in panel_a.itertuples(index=False):
        rows.append(
            {
                "panel": "A",
                "record_type": "seed",
                "model_or_ensemble_size": row.model,
                "seed": int(row.seed),
                "ensemble_size": math.nan,
                "subset_id": math.nan,
                "member_ids": str(int(row.seed)),
                "joint_ap": float(row.joint_ap),
                "joint_ap_sd": math.nan,
                "roc_auc": float(row.roc_auc),
                "roc_auc_sd": math.nan,
                "nll": float(row.nll),
                "nll_sd": math.nan,
                "brier": float(row.brier),
                "brier_sd": math.nan,
                "n_test": EXPECTED_TEST_COUNT,
                "calibrated": False,
                "aggregation_method": base_method_a,
                "source_file": row.metrics_path,
                "source_sha256": row.metrics_sha256,
            }
        )
    for model_name, _, _ in MODEL_SPECS:
        group = panel_a.loc[panel_a["model"] == model_name]
        rows.append(
            {
                "panel": "A",
                "record_type": "summary",
                "model_or_ensemble_size": model_name,
                "seed": "2025;2026;2027",
                "ensemble_size": math.nan,
                "subset_id": math.nan,
                "member_ids": "2025;2026;2027",
                "joint_ap": float(group["joint_ap"].mean()),
                "joint_ap_sd": float(group["joint_ap"].std(ddof=1)),
                "roc_auc": float(group["roc_auc"].mean()),
                "roc_auc_sd": float(group["roc_auc"].std(ddof=1)),
                "nll": float(group["nll"].mean()),
                "nll_sd": float(group["nll"].std(ddof=1)),
                "brier": float(group["brier"].mean()),
                "brier_sd": float(group["brier"].std(ddof=1)),
                "n_test": EXPECTED_TEST_COUNT,
                "calibrated": False,
                "aggregation_method": base_method_a,
                "source_file": ";".join(group["metrics_path"].tolist()),
                "source_sha256": ";".join(group["metrics_sha256"].tolist()),
            }
        )

    raw_detail_hash = sha256_file(raw_detail_path)
    for row in raw_detail.itertuples(index=False):
        rows.append(
            {
                "panel": "B",
                "record_type": "subset",
                "model_or_ensemble_size": str(int(row.ensemble_size)),
                "seed": math.nan,
                "ensemble_size": int(row.ensemble_size),
                "subset_id": int(row.subset_id),
                "member_ids": str(row.member_seeds),
                "joint_ap": float(row.joint_ap),
                "joint_ap_sd": math.nan,
                "roc_auc": float(row.roc_auc),
                "roc_auc_sd": math.nan,
                "nll": float(row.nll),
                "nll_sd": math.nan,
                "brier": float(row.brier),
                "brier_sd": math.nan,
                "n_test": int(row.n_test),
                "calibrated": False,
                "aggregation_method": row.aggregation_method,
                "source_file": str(raw_detail_path.resolve()),
                "source_sha256": raw_detail_hash,
            }
        )
    raw_summary_hash = sha256_file(raw_summary_path)
    for row in raw_summary.itertuples(index=False):
        rows.append(
            {
                "panel": "B",
                "record_type": "summary",
                "model_or_ensemble_size": str(int(row.ensemble_size)),
                "seed": math.nan,
                "ensemble_size": int(row.ensemble_size),
                "subset_id": math.nan,
                "member_ids": row.member_pool_seeds,
                "joint_ap": float(row.joint_ap_mean),
                "joint_ap_sd": float(row.joint_ap_sd)
                if pd.notna(row.joint_ap_sd)
                else math.nan,
                "roc_auc": float(row.roc_auc_mean),
                "roc_auc_sd": float(row.roc_auc_sd)
                if pd.notna(row.roc_auc_sd)
                else math.nan,
                "nll": float(row.nll_mean),
                "nll_sd": float(row.nll_sd) if pd.notna(row.nll_sd) else math.nan,
                "brier": float(row.brier_mean),
                "brier_sd": float(row.brier_sd)
                if pd.notna(row.brier_sd)
                else math.nan,
                "n_test": int(row.n_test),
                "calibrated": False,
                "aggregation_method": row.aggregation_method,
                "source_file": str(raw_summary_path.resolve()),
                "source_sha256": raw_summary_hash,
            }
        )
    columns = [
        "panel",
        "record_type",
        "model_or_ensemble_size",
        "seed",
        "ensemble_size",
        "subset_id",
        "member_ids",
        "joint_ap",
        "joint_ap_sd",
        "roc_auc",
        "roc_auc_sd",
        "nll",
        "nll_sd",
        "brier",
        "brier_sd",
        "n_test",
        "calibrated",
        "aggregation_method",
        "source_file",
        "source_sha256",
    ]
    source_data = pd.DataFrame(rows, columns=columns)
    if source_data["calibrated"].astype(bool).any():
        raise AssertionError("Fig3 source data unexpectedly contains calibrated=True")
    return source_data


def format_float(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.15f}"


def build_full_audit(
    project_root: Path,
    script_path: Path,
    start_local: datetime,
    start_utc: datetime,
    end_local: datetime,
    end_utc: datetime,
    elapsed_seconds: float,
    panel_a: pd.DataFrame,
    panel_a_audit: dict[str, object],
    member_audit: dict[str, object],
    raw_detail: pd.DataFrame,
    raw_summary: pd.DataFrame,
    raw_detail_path: Path,
    raw_summary_path: Path,
    source_data_path: Path,
    figure_paths: dict[str, Path],
    crosscheck: dict[str, object],
    calibrated_summary: pd.DataFrame,
    calibrated_hashes_before: dict[str, str],
    calibrated_hashes_after: dict[str, str],
) -> str:
    lines: list[str] = []
    append = lines.append
    append("FIG. 3 DATA AUDIT")
    append("=================")
    append("")
    append("Execution")
    append("---------")
    append(f"Script: {script_path.resolve()}")
    append(f"Script SHA256: {sha256_file(script_path)}")
    append(f"Project root: {project_root.resolve()}")
    append(f"Started (local): {start_local.isoformat()}")
    append(f"Started (UTC): {start_utc.isoformat()}")
    append(f"Finished (local): {end_local.isoformat()}")
    append(f"Finished (UTC): {end_utc.isoformat()}")
    append(f"Elapsed seconds: {elapsed_seconds:.3f}")
    append(f"Python: {platform.python_version()}")
    append(f"NumPy: {np.__version__}")
    append(f"pandas: {pd.__version__}")
    append(f"scikit-learn: {sklearn.__version__}")
    append(f"matplotlib: {mpl.__version__}")
    append("")
    append("Scope and exclusions")
    append("--------------------")
    append("No model training was performed.")
    append("Panel A reads completed per-seed metrics.json files.")
    append(
        "Panel B uses arithmetic means of frozen, uncalibrated per-member p_joint "
        "probabilities after material_id alignment."
    )
    append("No validation probabilities, temperatures, thresholds, or calibrated probabilities were used.")
    append("No threshold-based metric is used in Fig. 3.")
    append("")
    append("Panel A - common-seed task organization")
    append("-----------------------------------------")
    append(f"Common seeds: {list(COMMON_SEEDS)}")
    append(f"Split files declared by experiments: {panel_a_audit['split_names']}")
    append(f"n_test: {panel_a_audit['n_test']}")
    append(f"n_joint_positive: {panel_a_audit['n_joint_positive']}")
    append(f"Sorted material-ID SHA256: {panel_a_audit['sorted_material_id_sha256']}")
    append(
        "Sorted material-ID/label SHA256: "
        f"{panel_a_audit['sorted_material_id_label_sha256']}"
    )
    for model_name, _, _ in MODEL_SPECS:
        group = panel_a.loc[panel_a["model"] == model_name].sort_values("seed")
        append("")
        append(model_name)
        for row in group.itertuples(index=False):
            append(
                f"  seed {row.seed}: joint_ap={row.joint_ap:.15f}; "
                f"nll={row.nll:.15f}; brier={row.brier:.15f}; "
                f"roc_auc={row.roc_auc:.15f}"
            )
            append(f"    source: {row.metrics_path}")
            append(f"    source SHA256: {row.metrics_sha256}")
            append(f"    prediction source: {row.predictions_path}")
            append(f"    prediction SHA256: {row.predictions_sha256}")
        values = group["joint_ap"].to_numpy(float)
        append(f"  mean joint_ap: {np.mean(values):.15f}")
        append(f"  sample SD (numpy.std, ddof=1): {np.std(values, ddof=1):.15f}")
    append("")
    append("Panel A probability/logit and metric reproduction checks")
    for record in panel_a_audit["prediction_checks"]:
        append(
            f"  {record['model']}, seed {record['seed']}: "
            f"max_abs[p_joint-sigmoid(logit_joint)]="
            f"{record['max_abs_p_minus_sigmoid_logit']:.12g}; "
            f"recomputed_ap={record['recomputed_ap']:.15f}"
        )
    append("")
    append("Panel B - raw ensemble-size ablation")
    append("------------------------------------")
    append(f"Member seeds: {list(ENSEMBLE_SEEDS)}")
    append("Subset rule: all C(5,M) subsets for M in [1, 3, 5].")
    append(
        "Probability rule: p_joint_raw_ensemble(x) = arithmetic mean of member "
        "p_joint(x) in probability space."
    )
    append(f"n_test: {member_audit['n_test']}")
    append(f"n_joint_positive: {member_audit['n_joint_positive']}")
    append(f"Split manifest: {member_audit['split_path']}")
    append(f"Split manifest SHA256: {member_audit['split_sha256']}")
    append(f"Sorted material-ID SHA256: {member_audit['sorted_material_id_sha256']}")
    append(
        "Sorted material-ID/label SHA256: "
        f"{member_audit['sorted_material_id_label_sha256']}"
    )
    append("")
    append("Frozen member sources")
    for seed in ENSEMBLE_SEEDS:
        append(f"  seed {seed}: {member_audit['source_paths'][seed]}")
        append(f"    SHA256: {member_audit['source_sha256'][seed]}")
        append(f"    completion record: {member_audit['completed_paths'][seed]}")
        append(f"    completion SHA256: {member_audit['completed_sha256'][seed]}")
        append(
            "    max_abs[p_joint-sigmoid(logit_joint)]: "
            f"{member_audit['probability_logit_max_abs_diff'][seed]:.12g}"
        )
    append("")
    append("Raw subset metrics")
    for row in raw_detail.sort_values(["ensemble_size", "subset_id"]).itertuples(index=False):
        append(
            f"  M={row.ensemble_size}, subset={row.subset_id:02d}, "
            f"members=[{row.member_seeds}]: joint_ap={row.joint_ap:.15f}; "
            f"roc_auc={row.roc_auc:.15f}; nll={row.nll:.15f}; "
            f"brier={row.brier:.15f}; n_test={row.n_test}; calibrated=False"
        )
    append("")
    append("Raw size-level summary")
    for row in raw_summary.sort_values("ensemble_size").itertuples(index=False):
        append(
            f"  M={row.ensemble_size}, n_subsets={row.n_subsets}: "
            f"joint_ap_mean={row.joint_ap_mean:.15f}; "
            f"joint_ap_sd={format_float(row.joint_ap_sd)}; "
            f"nll_mean={row.nll_mean:.15f}; nll_sd={format_float(row.nll_sd)}; "
            f"brier_mean={row.brier_mean:.15f}; "
            f"brier_sd={format_float(row.brier_sd)}; calibrated=False"
        )
    append("")
    append("Raw M=5 cross-check against existing frozen raw metrics")
    append(f"  reference: {crosscheck['reference_path']}")
    for metric in ("joint_ap", "nll", "brier", "roc_auc"):
        append(
            f"  {metric}: recomputed={crosscheck['recomputed_values'][metric]:.15f}; "
            f"reference={crosscheck['reference_values'][metric]:.15f}; "
            f"difference={crosscheck['differences'][metric]:.12g}"
        )
    append(
        "  max_abs probability difference versus existing p_joint_mean: "
        f"{crosscheck['max_abs_probability_difference_vs_existing_p_joint_mean']:.12g}"
    )
    append("  status: PASS")
    append("")
    append("Audit-only comparison with the pre-existing calibrated ablation")
    append("---------------------------------------------------------------")
    for row in calibrated_summary.sort_values("ensemble_size").itertuples(index=False):
        raw = raw_summary.loc[raw_summary["ensemble_size"] == row.ensemble_size].iloc[0]
        append(
            f"  M={row.ensemble_size}: raw AP={raw['joint_ap_mean']:.15f}; "
            f"old calibrated AP={row.joint_ap_mean:.15f}; "
            f"raw-minus-calibrated={raw['joint_ap_mean'] - row.joint_ap_mean:.15f}"
        )
        append(
            f"       raw NLL={raw['nll_mean']:.15f}; "
            f"old calibrated NLL={row.nll_mean:.15f}; "
            f"raw-minus-calibrated={raw['nll_mean'] - row.nll_mean:.15f}"
        )
        append(
            f"       raw Brier={raw['brier_mean']:.15f}; "
            f"old calibrated Brier={row.brier_mean:.15f}; "
            f"raw-minus-calibrated={raw['brier_mean'] - row.brier_mean:.15f}"
        )
    append("These calibrated values were not included in Fig. 3 or Fig3_source_data.csv.")
    append("")
    append("Original calibrated-file immutability check")
    append("-------------------------------------------")
    for path, before_hash in calibrated_hashes_before.items():
        after_hash = calibrated_hashes_after[path]
        status = "PASS" if before_hash == after_hash else "FAIL"
        append(f"  {path}")
        append(f"    SHA256 before: {before_hash}")
        append(f"    SHA256 after:  {after_hash}")
        append(f"    unchanged: {status}")
    append("")
    append("New frozen raw sources and figure outputs")
    append("-----------------------------------------")
    for path in (
        raw_detail_path,
        raw_summary_path,
        source_data_path,
        figure_paths["pdf"],
        figure_paths["png"],
        figure_paths["svg"],
    ):
        append(f"  {path.resolve()}")
        append(f"    bytes: {path.stat().st_size}")
        append(f"    SHA256: {sha256_file(path)}")
    append("")
    append("Missing values, duplicates, and conflicts")
    append("----------------------------------------")
    append("Missing required Panel A seeds: none")
    append("Duplicate Panel A seeds used: none")
    append("Panel A seeds 2028/2029 mixed into comparison: no")
    append("Missing ensemble members: none")
    append("Duplicate material IDs within any member: none")
    append("Material-ID/label conflicts across members: none")
    append("Non-finite probabilities or metrics: none")
    append("Result conflicts after raw M=5 cross-check: none")
    append("M=5 SD: NA by definition (one subset)")
    append("")
    append("All required assertions passed.")
    return "\n".join(lines) + "\n"


def main() -> None:
    start_perf = time.perf_counter()
    start_local = datetime.now().astimezone()
    start_utc = datetime.now(timezone.utc)
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "paper_figures" / "fig3"
    )
    if not (project_root / "results" / "main").is_dir():
        raise AssertionError(f"Not a valid project root: {project_root}")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from train.metrics import binary_nll

    ensemble_dir = project_root / "results" / "main" / "M06_full_mtl" / "ensemble"
    calibrated_paths = [
        ensemble_dir / "ensemble_size_ablation.csv",
        ensemble_dir / "ensemble_size_ablation_detail.csv",
        ensemble_dir / "ensemble_metrics.json",
        ensemble_dir / "test_ensemble_predictions.csv",
        ensemble_dir / "temperatures.json",
    ]
    for path in calibrated_paths:
        if not path.is_file():
            raise AssertionError(f"Missing pre-existing ensemble audit file: {path}")
    calibrated_hashes_before = {
        str(path.resolve()): sha256_file(path) for path in calibrated_paths
    }

    panel_a, panel_a_audit = load_panel_a(project_root)
    member_frames, member_audit = load_ensemble_members(project_root)
    if panel_a_audit["sorted_material_id_label_sha256"] != member_audit[
        "sorted_material_id_label_sha256"
    ]:
        raise AssertionError("Panel A and Panel B do not share the same test IDs/labels")

    raw_detail, raw_summary, crosscheck = compute_raw_ablation(
        project_root, member_frames, binary_nll
    )
    print_terminal_audit(panel_a, raw_detail, raw_summary, crosscheck)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_detail_path = ensemble_dir / "ensemble_size_ablation_raw_detail.csv"
    raw_summary_path = ensemble_dir / "ensemble_size_ablation_raw.csv"
    raw_detail, raw_summary = write_raw_ablation_sources(
        raw_detail, raw_summary, raw_detail_path, raw_summary_path
    )

    figure_paths = create_figure(panel_a, raw_detail, raw_summary, output_dir)
    source_data = build_source_data(
        panel_a, raw_detail, raw_summary, raw_detail_path, raw_summary_path
    )
    source_data_path = output_dir / "Fig3_source_data.csv"
    atomic_to_csv(source_data, source_data_path)

    calibrated_source = pd.read_csv(ensemble_dir / "ensemble_size_ablation.csv")
    calibrated_source = calibrated_source.loc[
        calibrated_source["task"] == "joint",
        ["members", "pr_auc_mean", "nll_mean", "brier_mean"],
    ].rename(
        columns={
            "members": "ensemble_size",
            "pr_auc_mean": "joint_ap_mean",
        }
    )
    calibrated_source["ensemble_size"] = calibrated_source["ensemble_size"].astype(int)
    if tuple(sorted(calibrated_source["ensemble_size"].tolist())) != ENSEMBLE_SIZES:
        raise AssertionError("Old calibrated summary does not contain sizes 1, 3, 5")

    calibrated_hashes_after = {
        str(path.resolve()): sha256_file(path) for path in calibrated_paths
    }
    if calibrated_hashes_before != calibrated_hashes_after:
        raise AssertionError("A pre-existing calibrated ensemble file was modified")

    end_local = datetime.now().astimezone()
    end_utc = datetime.now(timezone.utc)
    elapsed_seconds = time.perf_counter() - start_perf
    audit_text = build_full_audit(
        project_root=project_root,
        script_path=Path(__file__).resolve(),
        start_local=start_local,
        start_utc=start_utc,
        end_local=end_local,
        end_utc=end_utc,
        elapsed_seconds=elapsed_seconds,
        panel_a=panel_a,
        panel_a_audit=panel_a_audit,
        member_audit=member_audit,
        raw_detail=raw_detail,
        raw_summary=raw_summary,
        raw_detail_path=raw_detail_path,
        raw_summary_path=raw_summary_path,
        source_data_path=source_data_path,
        figure_paths=figure_paths,
        crosscheck=crosscheck,
        calibrated_summary=calibrated_source,
        calibrated_hashes_before=calibrated_hashes_before,
        calibrated_hashes_after=calibrated_hashes_after,
    )
    audit_path = output_dir / "Fig3_data_audit.txt"
    atomic_write_text(audit_text, audit_path)

    print()
    print("Generated files:")
    for path in (
        raw_summary_path,
        raw_detail_path,
        figure_paths["pdf"],
        figure_paths["png"],
        figure_paths["svg"],
        source_data_path,
        audit_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
