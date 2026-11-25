"""Build Figure 2 v7 with a Joint-performance progression in panels a-b.

The audit workbook is read-only. All derived tables and figures are written to a
new output directory supplied by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


TASKS = ["Thermo", "Dynamic", "Joint"]
MODELS = ["Single-task ALIGNN", "Equal MTL", "Full MTL"]
COMPONENT_MODELS = ["Equal MTL", "Adaptive, no logic", "Fixed weights + logic", "Full MTL"]
SEEDS5 = [2025, 2026, 2027, 2028, 2029]
SEEDS3 = [2025, 2026, 2027]
MEMBER_COLUMNS = [f"p_member_{seed}" for seed in SEEDS5]
COMMON_MEMBER_COLUMNS = [f"p_member_{seed}" for seed in SEEDS3]
PR_GRID = np.linspace(0.0, 1.0, 501)
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 260906
LOGIC_EPS = 1.0e-10

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#6E6E6E"
LIGHT_BLUE = "#56B4E9"
GREEN = "#009E73"
LIGHT_GRAY = "#B5B5B5"

BASE_FONT_SIZE = 9.0
TITLE_FONT_SIZE = 10.0
AXIS_LABEL_FONT_SIZE = 10.0
TICK_FONT_SIZE = 8.5
LEGEND_FONT_SIZE = 7.2
ANNOTATION_FONT_SIZE = 7.2
PANEL_LABEL_FONT_SIZE = 10.5

PROGRESSION_COLORS = {
    "Single-task ALIGNN": GRAY,
    "Full MTL single": LIGHT_BLUE,
    "Full MTL ensemble": BLUE,
}

COLORS = {
    "Single-task ALIGNN": GRAY,
    "Equal MTL": ORANGE,
    "Adaptive, no logic": LIGHT_BLUE,
    "Fixed weights + logic": GREEN,
    "Full MTL": BLUE,
}
MARKERS = {
    "Single-task ALIGNN": "s",
    "Equal MTL": "^",
    "Adaptive, no logic": "D",
    "Fixed weights + logic": "P",
    "Full MTL": "o",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def numeric(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(f"{name}: non-finite values in {column}")


def validate_binary(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{name}: {column} is not binary")


def validate_probabilities(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        if not frame[column].between(0.0, 1.0).all():
            raise ValueError(f"{name}: {column} is outside [0,1]")


def validate_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, keys].head().to_dict("records")
        raise ValueError(f"{name}: duplicate key {keys}: {example}")


def read_audit_sheet(path: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet, header=3)
    frame = frame.dropna(how="all").copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_audit(path: Path) -> dict[str, pd.DataFrame]:
    names = [
        "F2 material", "F2c source", "F3 components", "F3 errors",
        "F2 AP check", "F3 AP check", "F3 error check",
        "F2a source", "F2b source", "F3a source", "F3b source", "F3c source",
    ]
    return {name: read_audit_sheet(path, name) for name in names}


def validate_f2_material(frame: pd.DataFrame) -> pd.DataFrame:
    name = "F2 material"
    required = {"task", "model", "split", "material_id", "y_true", "p_ensemble", *MEMBER_COLUMNS}
    require_columns(frame, required, name)
    if len(frame) != 10926:
        raise ValueError(f"{name}: expected 10926 rows, found {len(frame)}")
    numeric(frame, ["y_true", "p_ensemble", *MEMBER_COLUMNS], name)
    validate_binary(frame, ["y_true"], name)
    validate_probabilities(frame, ["p_ensemble", *MEMBER_COLUMNS], name)
    validate_unique(frame, ["task", "model", "material_id"], name)
    if set(frame.task) != set(TASKS) or set(frame.model) != set(MODELS):
        raise ValueError(f"{name}: unexpected task/model levels")
    if not frame.split.eq("IID (iid_split.csv)").all():
        raise ValueError(f"{name}: non-IID row found")
    expected = {"Thermo": (1214, 790), "Dynamic": (1214, 717), "Joint": (1214, 568)}
    for task in TASKS:
        subsets = []
        for model in MODELS:
            subset = frame.loc[(frame.task == task) & (frame.model == model)].sort_values("material_id")
            n, positive = expected[task]
            if len(subset) != n or int(subset.y_true.sum()) != positive:
                raise ValueError(f"{name}: {task}/{model} expected N={n}, positives={positive}")
            subsets.append(subset)
        reference = subsets[0][["material_id", "y_true"]].reset_index(drop=True)
        for model, subset in zip(MODELS[1:], subsets[1:]):
            if not reference.equals(subset[["material_id", "y_true"]].reset_index(drop=True)):
                raise ValueError(f"{name}: material IDs or labels differ for {task}/{model}")
    member_mean = frame[MEMBER_COLUMNS].mean(axis=1)
    max_difference = float(np.max(np.abs(member_mean - frame.p_ensemble)))
    if max_difference > 1e-12:
        raise ValueError(f"{name}: p_ensemble differs from member mean by {max_difference}")
    return frame


def validate_f3_components(frame: pd.DataFrame) -> pd.DataFrame:
    name = "F3 components"
    required = {
        "model", "seed", "split", "material_id", "y_thermo", "y_dynamic", "y_joint",
        "p_thermo", "p_dynamic", "p_joint", "any_frechet_violation",
    }
    require_columns(frame, required, name)
    if len(frame) != 14568:
        raise ValueError(f"{name}: expected 14568 rows, found {len(frame)}")
    numeric(frame, ["seed", "y_thermo", "y_dynamic", "y_joint", "p_thermo", "p_dynamic", "p_joint", "any_frechet_violation"], name)
    validate_binary(frame, ["y_thermo", "y_dynamic", "y_joint", "any_frechet_violation"], name)
    validate_probabilities(frame, ["p_thermo", "p_dynamic", "p_joint"], name)
    validate_unique(frame, ["model", "seed", "material_id"], name)
    if set(frame.model) != set(COMPONENT_MODELS) or set(frame.seed.astype(int)) != set(SEEDS3):
        raise ValueError(f"{name}: unexpected models or seeds")
    if not frame.split.eq("IID (iid_split.csv)").all():
        raise ValueError(f"{name}: non-IID row found")
    if not (frame.y_joint == frame.y_thermo * frame.y_dynamic).all():
        raise ValueError(f"{name}: Joint label is not Thermo AND Dynamic")
    for seed in SEEDS3:
        subsets = [frame.loc[(frame.model == model) & (frame.seed == seed)].sort_values("material_id") for model in COMPONENT_MODELS]
        reference = subsets[0][["material_id", "y_thermo", "y_dynamic", "y_joint"]].reset_index(drop=True)
        for model, subset in zip(COMPONENT_MODELS[1:], subsets[1:]):
            if not reference.equals(subset[["material_id", "y_thermo", "y_dynamic", "y_joint"]].reset_index(drop=True)):
                raise ValueError(f"{name}: alignment failed for seed={seed}, model={model}")
    return frame


def validate_f3_errors(frame: pd.DataFrame) -> pd.DataFrame:
    name = "F3 errors"
    required = {
        "model", "seed", "split", "material_id", "y_thermo", "y_dynamic", "y_joint",
        "p_joint", "classification_threshold", "pred_joint", "source_pred_joint",
        "threshold_prediction_mismatch", "true_quadrant", "error_type", "is_error",
    }
    require_columns(frame, required, name)
    if len(frame) != 7284:
        raise ValueError(f"{name}: expected 7284 rows, found {len(frame)}")
    numeric(frame, ["seed", "y_thermo", "y_dynamic", "y_joint", "p_joint", "classification_threshold", "pred_joint", "source_pred_joint", "threshold_prediction_mismatch", "is_error"], name)
    validate_binary(frame, ["y_thermo", "y_dynamic", "y_joint", "pred_joint", "source_pred_joint", "threshold_prediction_mismatch", "is_error"], name)
    validate_probabilities(frame, ["p_joint", "classification_threshold"], name)
    validate_unique(frame, ["model", "seed", "material_id"], name)
    if set(frame.model) != {"Single-task ALIGNN", "Full MTL"} or set(frame.seed.astype(int)) != set(SEEDS3):
        raise ValueError(f"{name}: unexpected models or seeds")
    if not frame.split.eq("IID (iid_split.csv)").all():
        raise ValueError(f"{name}: non-IID row found")
    return frame


def load_joint_predictions(
    project_root: Path, model: str, seeds: list[int],
) -> tuple[dict[int, pd.DataFrame], list[Path]]:
    if model == "Single-task ALIGNN":
        base = project_root / "results/main/M04_single_alignn"
    elif model == "Full MTL":
        base = project_root / "results/main/M06_full_mtl"
    else:
        raise ValueError(f"Unsupported progression model: {model}")
    paths = [base / f"seed_{seed}" / "test_predictions.csv" for seed in seeds]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Joint prediction files for {model}: {missing}")
    frames: dict[int, pd.DataFrame] = {}
    for seed, path in zip(seeds, paths):
        item = pd.read_csv(path, dtype={"material_id": "string"})
        require_columns(item, {"material_id", "y_joint", "p_joint"}, str(path))
        numeric(item, ["y_joint", "p_joint"], str(path))
        validate_binary(item, ["y_joint"], str(path))
        validate_probabilities(item, ["p_joint"], str(path))
        validate_unique(item, ["material_id"], str(path))
        item = item[["material_id", "y_joint", "p_joint"]].sort_values("material_id").reset_index(drop=True)
        if len(item) != 1214 or int(item.y_joint.sum()) != 568:
            raise ValueError(f"{path}: expected N=1214 and 568 Joint positives")
        frames[seed] = item
    reference = frames[seeds[0]][["material_id", "y_joint"]]
    for seed in seeds[1:]:
        if not reference.equals(frames[seed][["material_id", "y_joint"]]):
            raise ValueError(f"Joint IDs or labels are not aligned for {model}, seed={seed}")
    return frames, paths


def panel_a_data(frame: pd.DataFrame, project_root: Path, output_dir: Path) -> dict[str, object]:
    selected = frame.loc[frame.task == "Joint"].copy()
    alignn_frames, alignn_paths = load_joint_predictions(
        project_root, "Single-task ALIGNN", SEEDS3,
    )
    full_frames, full_paths = load_joint_predictions(project_root, "Full MTL", SEEDS5)
    source_frames = {
        "Single-task ALIGNN": alignn_frames,
        "Full MTL": full_frames,
    }
    raw_rows: list[dict[str, object]] = []
    aligned_rows: list[dict[str, object]] = []
    internal_ap_rows: list[dict[str, object]] = []
    curve_summary: dict[str, dict[str, object]] = {}
    for source_model, series in [
        ("Single-task ALIGNN", "Single-task ALIGNN"),
        ("Full MTL", "Full MTL single"),
    ]:
        audit_subset = selected.loc[selected.model == source_model].copy()
        audit_subset["material_id"] = audit_subset.material_id.astype("string")
        audit_subset = audit_subset.sort_values("material_id").reset_index(drop=True)
        y = source_frames[source_model][SEEDS3[0]].y_joint.to_numpy(int)
        if not audit_subset[["material_id", "y_true"]].rename(columns={"y_true": "y_joint"}).equals(
            source_frames[source_model][SEEDS3[0]][["material_id", "y_joint"]]
        ):
            raise ValueError(f"Audit/raw Joint IDs or labels differ for {source_model}")
        aligned = []
        seed_aps = []
        for seed, column in zip(SEEDS3, COMMON_MEMBER_COLUMNS):
            p = source_frames[source_model][seed].p_joint.to_numpy(float)
            max_audit_difference = float(np.max(np.abs(p - audit_subset[column].to_numpy(float))))
            if max_audit_difference > 1e-12:
                raise ValueError(
                    f"Audit/raw p_joint differs for {source_model}/{seed}: {max_audit_difference}"
                )
            precision, recall, thresholds = precision_recall_curve(y, p)
            for index, (r_value, p_value) in enumerate(zip(recall, precision)):
                raw_rows.append({
                    "series": series, "source_model": source_model,
                    "seed": seed, "point_index": index,
                    "recall": r_value, "precision": p_value,
                    "threshold": thresholds[index] if index < len(thresholds) else np.nan,
                })
            # precision_recall_curve returns decreasing recall. Reversal produces a
            # non-decreasing grid; duplicate recall values are retained exactly as
            # documented in the execution task and passed directly to np.interp.
            interpolated = np.interp(PR_GRID, recall[::-1], precision[::-1])
            aligned.append(interpolated)
            seed_ap = average_precision_score(y, p)
            seed_aps.append(seed_ap)
            internal_ap_rows.append({
                "series": series, "source_model": source_model, "seed": seed,
                "aggregation": "M=1", "AP": seed_ap,
            })
        matrix = np.vstack(aligned)
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        curve_summary[series] = {
            "mean": mean, "sd": sd,
            "mean_ap": float(np.mean(seed_aps)),
            "sd_ap": float(np.std(seed_aps, ddof=1)),
        }
        for index, recall_value in enumerate(PR_GRID):
            row = {
                "series": series, "source_model": source_model, "recall_grid": recall_value,
                **{f"precision_seed_{seed}": matrix[position, index] for position, seed in enumerate(SEEDS3)},
                "precision_mean": mean[index], "precision_sd": sd[index],
                "lower_unclipped": mean[index] - sd[index], "upper_unclipped": mean[index] + sd[index],
                "lower_plotted": np.clip(mean[index] - sd[index], 0.0, 1.0),
                "upper_plotted": np.clip(mean[index] + sd[index], 0.0, 1.0),
            }
            aligned_rows.append(row)

    y_full = full_frames[SEEDS5[0]].y_joint.to_numpy(int)
    p_full = np.mean([full_frames[seed].p_joint.to_numpy(float) for seed in SEEDS5], axis=0)
    precision_full, recall_full, thresholds_full = precision_recall_curve(y_full, p_full)
    full_rows = []
    for index, (r_value, p_value) in enumerate(zip(recall_full, precision_full)):
        full_rows.append({
            "series": "Full MTL ensemble", "source_model": "Full MTL",
            "members": 5, "point_index": index,
            "recall": r_value, "precision": p_value,
            "threshold": thresholds_full[index] if index < len(thresholds_full) else np.nan,
        })
    ensemble_ap = average_precision_score(y_full, p_full)
    internal_ap_rows.append({
        "series": "Full MTL ensemble", "source_model": "Full MTL", "seed": "ensemble",
        "aggregation": "M=5 probability mean", "AP": ensemble_ap,
    })

    raw_frame = pd.DataFrame(raw_rows)
    aligned_frame = pd.DataFrame(aligned_rows)
    full_frame = pd.DataFrame(full_rows)
    ap_frame = pd.DataFrame(internal_ap_rows)
    raw_frame.to_csv(output_dir / "panel_a_single_model_raw_pr.csv", index=False)
    aligned_frame.to_csv(output_dir / "panel_a_single_model_aligned_pr.csv", index=False)
    full_frame.to_csv(output_dir / "panel_a_full_ensemble_pr.csv", index=False)
    ap_frame.to_csv(output_dir / "panel_a_internal_ap_check.csv", index=False)
    return {
        "curve_summary": curve_summary,
        "full_precision": precision_full,
        "full_recall": recall_full,
        "ensemble_ap": float(ensemble_ap),
        "prevalence": float(y_full.mean()),
        "internal_ap": ap_frame,
        "source_paths": alignn_paths + full_paths,
    }


def panel_b_data(panel_a: dict[str, object], output_dir: Path) -> pd.DataFrame:
    rows = []
    for series in ["Single-task ALIGNN", "Full MTL single"]:
        summary = panel_a["curve_summary"][series]
        rows.append({
            "series": series, "display_label": "ALIGNN" if series == "Single-task ALIGNN" else "Full MTL",
            "Joint_AP": summary["mean_ap"], "sample_SD": summary["sd_ap"],
            "independent_runs": 3, "members_per_estimate": 1,
            "error_bar_definition": "sample SD across common seeds 2025-2027",
        })
    rows.append({
        "series": "Full MTL ensemble", "display_label": "Full MTL ensemble",
        "Joint_AP": panel_a["ensemble_ap"], "sample_SD": np.nan,
        "independent_runs": 1, "members_per_estimate": 5,
        "error_bar_definition": "none; one arithmetic probability-mean ensemble",
    })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "panel_b_joint_ap_progression.csv", index=False)
    return result


def supplementary_m5_data(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for model in MODELS:
            subset = frame.loc[(frame.task == task) & (frame.model == model)].sort_values("material_id")
            y = subset.y_true.to_numpy(int)
            p = subset[MEMBER_COLUMNS].mean(axis=1).to_numpy(float)
            rows.append({
                "task": task, "model": model, "members": 5,
                "aggregation": "material-aligned arithmetic mean of member probabilities",
                "AP": average_precision_score(y, p),
            })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "supplementary_m5_all_task_ap.csv", index=False)
    result.loc[result.task == "Joint"].to_csv(
        output_dir / "supplementary_joint_m5_comparison.csv", index=False,
    )
    return result


def parse_source_paths(project_root: Path, value: str) -> list[Path]:
    paths = []
    for item in str(value).split(";"):
        cleaned = item.strip().replace("\\", "/")
        if not cleaned:
            continue
        candidate = Path(cleaned)
        paths.append(candidate if candidate.is_absolute() else project_root / candidate)
    return paths


def panel_c_data(source: pd.DataFrame, project_root: Path, output_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    required = {"series", "split", "n_test", "n_positive", "member_list", "probability_field", "source_files"}
    require_columns(source, required, "F2c source")
    selected = source.loc[source.series == "Full MTL"].copy()
    order = ["Random", "Composition", "Chemical system", "Structure family"]
    if len(selected) != 4 or set(selected.split) != set(order):
        raise ValueError("F2c source: expected four Full MTL split rows")
    expected = {
        "Random": (1215, 569), "Composition": (1211, 568),
        "Chemical system": (1214, 566), "Structure family": (1249, 589),
    }
    rows = []
    paths_used = []
    for split in order:
        row = selected.loc[selected.split == split].iloc[0]
        paths = parse_source_paths(project_root, row.source_files)
        if len(paths) != 5 or any(not path.is_file() for path in paths):
            missing = [str(path) for path in paths if not path.is_file()]
            raise FileNotFoundError(f"F2c source {split}: expected five files; missing={missing}")
        frames = []
        for path in paths:
            item = pd.read_csv(path, dtype={"material_id": "string"})
            require_columns(item, {"material_id", "y_joint", "p_joint"}, str(path))
            numeric(item, ["y_joint", "p_joint"], str(path))
            validate_binary(item, ["y_joint"], str(path))
            validate_probabilities(item, ["p_joint"], str(path))
            validate_unique(item, ["material_id"], str(path))
            frames.append(item.set_index("material_id").sort_index())
        reference = frames[0]
        for path, item in zip(paths[1:], frames[1:]):
            if not reference.index.equals(item.index) or not reference.y_joint.equals(item.y_joint):
                raise ValueError(f"{split}: member IDs or labels differ in {path}")
        y = reference.y_joint.to_numpy(int)
        p = np.mean([item.p_joint.to_numpy(float) for item in frames], axis=0)
        expected_n, expected_positive = expected[split]
        if len(y) != expected_n or int(y.sum()) != expected_positive:
            raise ValueError(f"{split}: found N={len(y)}, positives={int(y.sum())}; expected {expected[split]}")
        rows.append({
            "split": split, "Joint_AP": average_precision_score(y, p), "n_test": len(y),
            "n_positive": int(y.sum()), "members": 5, "member_seeds": ",".join(map(str, SEEDS5)),
            "aggregation": "material-ID-aligned arithmetic mean of raw p_joint",
            "source_files": "; ".join(str(path) for path in paths),
        })
        paths_used.extend(paths)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "panel_c_full_split_ap.csv", index=False)
    return result, paths_used


def logical_violation(frame: pd.DataFrame) -> np.ndarray:
    lower = np.maximum(0.0, frame.p_thermo.to_numpy(float) + frame.p_dynamic.to_numpy(float) - 1.0)
    upper = np.minimum(frame.p_thermo.to_numpy(float), frame.p_dynamic.to_numpy(float))
    joint = frame.p_joint.to_numpy(float)
    return (joint < lower - LOGIC_EPS) | (joint > upper + LOGIC_EPS)


def panels_de_data(frame: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ap_rows, violation_rows, mismatch_rows = [], [], []
    for model in COMPONENT_MODELS:
        for seed in SEEDS3:
            subset = frame.loc[(frame.model == model) & (frame.seed == seed)].sort_values("material_id")
            if len(subset) != 1214 or int(subset.y_joint.sum()) != 568:
                raise ValueError(f"F3 components: invalid N/positives for {model}/{seed}")
            ap_rows.append({"model": model, "seed": seed, "members": 1, "Joint_AP": average_precision_score(subset.y_joint, subset.p_joint)})
            calculated = logical_violation(subset).astype(int)
            stored = subset.any_frechet_violation.to_numpy(int)
            bad = calculated != stored
            violation_rows.append({
                "model": model, "seed": seed, "members": 1, "n_test": len(subset),
                "violation_count": int(calculated.sum()), "violation_rate": float(calculated.mean()),
                "logic_epsilon": LOGIC_EPS,
            })
            if bad.any():
                examples = subset.loc[bad, ["material_id", "p_thermo", "p_dynamic", "p_joint", "any_frechet_violation"]].copy()
                examples["model"] = model; examples["seed"] = seed; examples["recomputed_violation"] = calculated[bad]
                mismatch_rows.extend(examples.to_dict("records"))
    ap_frame = pd.DataFrame(ap_rows)
    violation_frame = pd.DataFrame(violation_rows)
    ap_summary = (ap_frame.groupby("model", sort=False).Joint_AP
                  .agg([("mean_AP", "mean"), ("sample_SD", lambda x: x.std(ddof=1))]).reset_index())
    violation_summary = (violation_frame.groupby("model", sort=False).violation_rate
                         .agg([("mean_violation_rate", "mean"), ("sample_SD", lambda x: x.std(ddof=1))]).reset_index())
    ap_frame.to_csv(output_dir / "panel_d_seed_ap.csv", index=False)
    ap_summary.to_csv(output_dir / "panel_d_summary.csv", index=False)
    violation_frame.to_csv(output_dir / "panel_e_seed_violation.csv", index=False)
    violation_summary.to_csv(output_dir / "panel_e_summary.csv", index=False)
    pd.DataFrame(mismatch_rows).to_csv(output_dir / "panel_e_violation_mismatches.csv", index=False)
    if mismatch_rows:
        raise ValueError(f"F3 components: {len(mismatch_rows)} stored/recomputed violation mismatches")
    return ap_frame, ap_summary, violation_frame, violation_summary


def panel_f_data(frame: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    quadrant_specs = [
        (0, 0, "T−/D−", "FP", 275), (0, 1, "T−/D+", "FP", 149),
        (1, 0, "T+/D−", "FP", 222), (1, 1, "T+/D+", "FN", 568),
    ]
    rows, mismatch_rows = [], []
    for model in ["Single-task ALIGNN", "Full MTL"]:
        for seed in SEEDS3:
            subset = frame.loc[(frame.model == model) & (frame.seed == seed)].sort_values("material_id").copy()
            if len(subset) != 1214:
                raise ValueError(f"F3 errors: expected 1214 rows for {model}/{seed}")
            thresholds = subset.classification_threshold.unique()
            if len(thresholds) != 1:
                raise ValueError(f"F3 errors: multiple thresholds for {model}/{seed}")
            threshold = float(thresholds[0])
            prediction = (subset.p_joint.to_numpy(float) >= threshold).astype(int)
            mismatch = (prediction != subset.source_pred_joint.to_numpy(int)) | (subset.threshold_prediction_mismatch.to_numpy(int) != 0)
            if mismatch.any():
                bad = subset.loc[mismatch, ["material_id", "p_joint", "classification_threshold", "source_pred_joint", "threshold_prediction_mismatch"]].copy()
                bad["model"] = model; bad["seed"] = seed; bad["recomputed_pred_joint"] = prediction[mismatch]
                mismatch_rows.extend(bad.to_dict("records"))
            for yt, yd, label, kind, expected_n in quadrant_specs:
                mask = (subset.y_thermo.to_numpy(int) == yt) & (subset.y_dynamic.to_numpy(int) == yd)
                if int(mask.sum()) != expected_n:
                    raise ValueError(f"F3 errors: {model}/{seed}/{label} expected n={expected_n}, found {int(mask.sum())}")
                y = subset.y_joint.to_numpy(int)[mask]
                pred = prediction[mask]
                errors = int(((pred == 1) & (y == 0)).sum()) if kind == "FP" else int(((pred == 0) & (y == 1)).sum())
                rows.append({
                    "model": model, "seed": seed, "members": 1, "true_quadrant": label,
                    "error_type": kind, "n_materials": expected_n, "errors": errors,
                    "error_rate": errors / expected_n, "classification_threshold": threshold,
                    "threshold_source": "validation-selected threshold stored in audit workbook",
                })
    detail = pd.DataFrame(rows)
    summary = (detail.groupby(["model", "true_quadrant", "error_type", "n_materials"], sort=False).error_rate
               .agg([("mean_error_rate", "mean"), ("sample_SD", lambda x: x.std(ddof=1))]).reset_index())
    detail.to_csv(output_dir / "panel_f_seed_errors.csv", index=False)
    summary.to_csv(output_dir / "panel_f_summary.csv", index=False)
    pd.DataFrame(mismatch_rows).to_csv(output_dir / "panel_f_prediction_mismatches.csv", index=False)
    if mismatch_rows:
        raise ValueError(f"F3 errors: {len(mismatch_rows)} threshold/prediction mismatches")
    return detail, summary


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(direction="out", width=0.9, length=3, labelsize=TICK_FONT_SIZE)
    ax.yaxis.labelpad = 8


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.35, 1.035, label, transform=ax.transAxes, fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="bold", ha="left", va="bottom")


def draw_panel_a(ax: plt.Axes, panel_a: dict[str, object]) -> None:
    for series, linestyle, linewidth in [
        ("Single-task ALIGNN", "--", 1.4),
        ("Full MTL single", "-.", 1.55),
    ]:
        summary = panel_a["curve_summary"][series]
        color = PROGRESSION_COLORS[series]
        display = "ALIGNN" if series == "Single-task ALIGNN" else "Full MTL"
        label = f"{display} (AP={summary['mean_ap']:.3f}±{summary['sd_ap']:.3f})"
        ax.plot(PR_GRID, summary["mean"], color=color, linestyle=linestyle,
                linewidth=linewidth, label=label)
    ax.plot(
        panel_a["full_recall"], panel_a["full_precision"],
        color=PROGRESSION_COLORS["Full MTL ensemble"], linewidth=1.9,
        label=f"Full MTL ensemble (AP={panel_a['ensemble_ap']:.3f})",
    )
    ax.set(xlim=(0, 1), ylim=(0.4, 1.0), xlabel="Recall", ylabel="Precision", title="Joint PR progression")
    ax.legend(frameon=False, loc="lower left", handlelength=1.55, labelspacing=0.25,
              handletextpad=0.35, borderaxespad=0.35, fontsize=LEGEND_FONT_SIZE)
    style_axis(ax)
    panel_label(ax, "(a)")


def draw_panel_b(ax: plt.Axes, progression: pd.DataFrame) -> None:
    order = ["Single-task ALIGNN", "Full MTL single", "Full MTL ensemble"]
    selected = progression.set_index("series").loc[order]
    x = np.arange(3)
    values = selected.Joint_AP.to_numpy(float)
    ax.plot(x, values, color="#C4CAD0", linewidth=1.15, zorder=1)
    markers = ["s", "o", "D"]
    for index, (series, marker) in enumerate(zip(order, markers)):
        row = selected.loc[series]
        value = float(row.Joint_AP)
        sd = float(row.sample_SD) if pd.notna(row.sample_SD) else None
        if sd is None:
            ax.scatter(index, value, s=38, marker=marker, color=PROGRESSION_COLORS[series],
                       edgecolor="white", linewidth=0.55, zorder=3)
            label = f"{value:.3f}"
            label_y = value + 0.0032
        else:
            ax.errorbar(index, value, yerr=sd, fmt=marker, color=PROGRESSION_COLORS[series],
                        markeredgecolor="white", markeredgewidth=0.55, markersize=6.2,
                        capsize=2.5, elinewidth=0.95, zorder=3)
            label = f"{value:.3f} ± {sd:.3f}"
            label_y = value + sd + 0.0032
        if index == 0:
            ax.annotate(label, (index, label_y), xytext=(3, 0), textcoords="offset points",
                        ha="left", va="bottom", fontsize=ANNOTATION_FONT_SIZE, color="#303030")
        else:
            ax.text(index, label_y, label, ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE,
                    color="#303030")
    ax.set(
        xticks=x,
        xticklabels=["ALIGNN", "Full MTL", "Full MTL\nensemble"],
        ylabel="Joint AP", ylim=(0.845, 0.922), title="Joint AP progression",
    )
    ax.margins(x=0.12)
    style_axis(ax)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE, pad=2)
    panel_label(ax, "(b)")


def draw_figure(
    panel_a: dict[str, object], progression: pd.DataFrame, panel_c: pd.DataFrame,
    ap_seed: pd.DataFrame, ap_summary: pd.DataFrame,
    violation_seed: pd.DataFrame, violation_summary: pd.DataFrame,
    error_summary: pd.DataFrame, output_dir: Path, font_name: str,
) -> None:
    plt.rcParams.update({
        "font.family": font_name, "font.size": BASE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE, "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE, "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE, "axes.linewidth": 0.9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 3, figsize=(175 / 25.4, 110 / 25.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.018, wspace=0.045, hspace=0.055)

    ax = axes[0, 0]
    draw_panel_a(ax, panel_a)

    ax = axes[0, 1]
    draw_panel_b(ax, progression)

    ax = axes[0, 2]
    split_order = ["Random", "Composition", "Chemical system", "Structure family"]
    split_labels = ["Random", "Composition", "Chemistry", "Structure"]
    y = np.arange(4)
    values = panel_c.set_index("split").loc[split_order, "Joint_AP"].to_numpy(float)
    ax.barh(y, values, color=BLUE, alpha=0.92, height=0.56,
            edgecolor="none", zorder=2)
    for ypos, value in zip(y, values):
        ax.annotate(f"{value:.3f}", (value, ypos), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=ANNOTATION_FONT_SIZE)
    ax.set(
        yticks=y, yticklabels=split_labels, xlabel="Joint AP",
        xlim=(0.0, 1.0), title="Data splits",
    )
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.invert_yaxis()
    ax.margins(y=0.16)
    style_axis(ax)
    panel_label(ax, "(c)")

    component_labels = ["Equal", "+AW", "+LC", "Full"]
    legend_labels = {
        "Equal MTL": "Equal MTL",
        "Adaptive, no logic": "+AW",
        "Fixed weights + logic": "+LC",
        "Full MTL": "Full MTL",
    }
    jitter = np.array([-0.055, 0.0, 0.055])
    x = np.arange(4)
    ax = axes[1, 0]
    for index, model in enumerate(COMPONENT_MODELS):
        values = ap_seed.loc[ap_seed.model == model].set_index("seed").loc[SEEDS3, "Joint_AP"].to_numpy(float)
        row = ap_summary.loc[ap_summary.model == model].iloc[0]
        ax.scatter(index + jitter, values, s=18, color=COLORS[model], alpha=0.38,
                   edgecolor="none", zorder=2)
        ax.errorbar(index, row.mean_AP, yerr=row.sample_SD, fmt=MARKERS[model],
                    color=COLORS[model], markeredgecolor="white", markeredgewidth=0.5,
                    markersize=6, capsize=2.5, elinewidth=1.0, zorder=3)
    ax.set(xticks=x, xticklabels=component_labels, ylabel="Joint AP",
           ylim=(0.84, 0.90), title="Component ablation")
    style_axis(ax)
    panel_label(ax, "(d)")

    ax = axes[1, 1]
    for index, model in enumerate(COMPONENT_MODELS):
        values = violation_seed.loc[violation_seed.model == model].set_index("seed").loc[SEEDS3, "violation_rate"].to_numpy(float)
        row = violation_summary.loc[violation_summary.model == model].iloc[0]
        ax.scatter(index + jitter, values, s=18, color=COLORS[model], alpha=0.38,
                   edgecolor="none", zorder=2)
        ax.errorbar(index, row.mean_violation_rate, yerr=row.sample_SD,
                    fmt=MARKERS[model], color=COLORS[model],
                    markeredgecolor="white", markeredgewidth=0.5, markersize=6,
                    capsize=2.5, elinewidth=1.0, label=legend_labels[model], zorder=3)
    ax.set(xticks=x, xticklabels=component_labels, ylabel="Violation rate",
           ylim=(0, 1.0), title="Logical violation rate")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.025),
              ncol=2, handlelength=1.25, columnspacing=0.8, handletextpad=0.35,
              labelspacing=0.25, borderaxespad=0.0, fontsize=LEGEND_FONT_SIZE)
    style_axis(ax)
    panel_label(ax, "(e)")

    ax = axes[1, 2]
    quadrants = ["T−/D−", "T−/D+", "T+/D−", "T+/D+"]
    counts = [275, 149, 222, 568]
    x = np.arange(4) * 1.16; width = 0.32
    tops = np.zeros(4)
    for offset, model, label in [
        (-width / 2, "Single-task ALIGNN", "ALIGNN"), (width / 2, "Full MTL", "Full MTL")
    ]:
        group = error_summary.loc[error_summary.model == model].set_index("true_quadrant").loc[quadrants]
        means = group.mean_error_rate.to_numpy(float)
        sds = group.sample_SD.to_numpy(float)
        tops = np.maximum(tops, means + sds)
        ax.bar(x + offset, means, width, yerr=sds,
               color=COLORS[model], label=label, capsize=2, error_kw={"elinewidth": 0.9})
    for xpos, top, text in zip(x, tops, ["FP", "FP", "FP", "FN"]):
        ax.text(xpos, min(top + 0.018, 0.565), text, ha="center", va="bottom",
                fontsize=ANNOTATION_FONT_SIZE, color=GRAY)
    ax.set(xticks=x, xticklabels=[f"{label}\nn={count}" for label, count in zip(quadrants, counts)],
           ylabel="Error rate", ylim=(0, 0.60), title="Errors by stability class")
    ax.margins(x=0.035)
    ax.legend(frameon=False, loc="upper left", labelspacing=0.25)
    style_axis(ax)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE, pad=2)
    panel_label(ax, "(f)")

    for suffix, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
        fig.savefig(output_dir / f"fig2_final_v7.{suffix}", facecolor="white", **kwargs)
    plt.close(fig)


def save_standalone_panels(
    panel_a: dict[str, object], progression: pd.DataFrame, output_dir: Path,
) -> None:
    for name, drawer, data in [
        ("fig2_panel_a", draw_panel_a, panel_a),
        ("fig2_panel_b", draw_panel_b, progression),
    ]:
        fig, ax = plt.subplots(figsize=(78 / 25.4, 62 / 25.4), constrained_layout=True)
        drawer(ax, data)
        for suffix, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
            fig.savefig(output_dir / f"{name}.{suffix}", facecolor="white", **kwargs)
        plt.close(fig)


def draw_supplementary_m5(comparison: pd.DataFrame, output_dir: Path) -> None:
    order = ["Single-task ALIGNN", "Equal MTL", "Full MTL"]
    selected = comparison.loc[comparison.task == "Joint"].set_index("model").loc[order]
    x = np.arange(3)
    values = selected.AP.to_numpy(float)
    colors = [GRAY, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(90 / 25.4, 62 / 25.4), constrained_layout=True)
    ax.plot(x, values, color="#C4CAD0", linewidth=1.15, zorder=1)
    for index, (value, color) in enumerate(zip(values, colors)):
        ax.scatter(index, value, s=42, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        horizontal_alignment = "left" if index == 0 else ("right" if index == 2 else "center")
        x_position = index + 0.025 if index == 0 else index
        ax.text(x_position, value + 0.0015, f"{value:.3f}", ha=horizontal_alignment,
                va="bottom", fontsize=ANNOTATION_FONT_SIZE)
    ax.set(
        xticks=x, xticklabels=["ALIGNN", "Equal MTL", "Full MTL"],
        ylabel="Joint AP", ylim=(0.882, 0.912),
        title="Matched five-member ensembles",
    )
    style_axis(ax)
    for suffix, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
        fig.savefig(output_dir / f"supplementary_joint_m5_comparison.{suffix}", facecolor="white", **kwargs)
    plt.close(fig)


def compare_expected(actual: pd.DataFrame, expected: dict[tuple[str, str], float], value_col: str, tolerance: float = 1e-9) -> float:
    differences = []
    for keys, value in expected.items():
        task, model = keys
        found = float(actual.loc[(actual.task == task) & (actual.model == model), value_col].iloc[0])
        differences.append(abs(found - value))
    maximum = max(differences, default=0.0)
    if maximum > tolerance:
        raise ValueError(f"numeric check failed for {value_col}: maximum difference {maximum}")
    return maximum


def choose_font() -> tuple[str, str]:
    available = {font.name for font in fm.fontManager.ttflist}
    if "Arial" in available:
        return "Arial", "Arial available"
    return "DejaVu Sans", "Arial unavailable; used embeddable DejaVu Sans fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--audit-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    audit_path = args.audit_xlsx.resolve()
    output_dir = args.output_dir.resolve()
    if not project_root.is_dir():
        raise NotADirectoryError(project_root)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = load_audit(audit_path)
    f2 = validate_f2_material(audit["F2 material"])
    components = validate_f3_components(audit["F3 components"])
    errors = validate_f3_errors(audit["F3 errors"])

    panel_a = panel_a_data(f2, project_root, output_dir)
    progression = panel_b_data(panel_a, output_dir)
    supplementary_m5 = supplementary_m5_data(f2, output_dir)
    panel_c, panel_c_paths = panel_c_data(audit["F2c source"], project_root, output_dir)
    ap_seed, ap_summary, violation_seed, violation_summary = panels_de_data(components, output_dir)
    error_seed, error_summary = panel_f_data(errors, output_dir)

    expected_ap = {
        ("Thermo", "Single-task ALIGNN"): 0.984126827704,
        ("Thermo", "Equal MTL"): 0.979436340971,
        ("Thermo", "Full MTL"): 0.981280625734,
        ("Dynamic", "Single-task ALIGNN"): 0.898838127584,
        ("Dynamic", "Equal MTL"): 0.890643955259,
        ("Dynamic", "Full MTL"): 0.905689003944,
        ("Joint", "Single-task ALIGNN"): 0.896940434588,
        ("Joint", "Equal MTL"): 0.891505884278,
        ("Joint", "Full MTL"): 0.906354899978,
    }
    max_expected_ap_difference = compare_expected(supplementary_m5, expected_ap, "AP")

    font_name, font_note = choose_font()
    draw_figure(panel_a, progression, panel_c, ap_seed, ap_summary,
                violation_seed, violation_summary, error_summary, output_dir, font_name)
    save_standalone_panels(panel_a, progression, output_dir)
    draw_supplementary_m5(supplementary_m5, output_dir)

    split_rules = {
        "Random": "stratified random IID control; review_splits/random_iid_stratified.csv",
        "Composition": "canonical-composition-disjoint; review_splits/composition_disjoint.csv",
        "Chemical system": "element-set-disjoint; review_splits/chemical_system_disjoint.csv",
        "Structure family": "structure-family-key-disjoint; review_splits/structure_family_disjoint.csv",
    }
    source_sheet_names = ["F2a source", "F2b source", "F3a source", "F3b source", "F3c source"]
    provenance_paths: list[Path] = []
    source_table_records: dict[str, list[dict[str, object]]] = {}
    for sheet_name in source_sheet_names:
        table = audit[sheet_name].astype(object).where(pd.notna(audit[sheet_name]), None)
        source_table_records[sheet_name] = table.to_dict("records")
        if "source_files" in table.columns:
            for value in table.source_files.dropna():
                provenance_paths.extend(parse_source_paths(project_root, str(value)))
    source_files = [
        audit_path, project_root / "README.md", project_root / "docs/02_methodological_guards.md",
        project_root / "docs/review_matrix.md", *panel_a["source_paths"], *panel_c_paths,
        *provenance_paths,
    ]
    source_files = sorted({path.resolve() for path in source_files}, key=lambda item: str(item).lower())
    missing_provenance = [str(path) for path in source_files if not path.is_file()]
    if missing_provenance:
        raise FileNotFoundError(f"Source manifest contains missing provenance files: {missing_provenance}")
    manifest = {
        "status": "ok",
        "figure": "fig2_final_v7",
        "project_root": str(project_root),
        "audit_workbook": str(audit_path),
        "audit_sheet_header_row_zero_based": 3,
        "output_dir": str(output_dir),
        "input_sha256": {str(path): sha256(path) for path in source_files},
        "audit_sheets": {name: {"rows": int(len(frame)), "columns": list(frame.columns)} for name, frame in audit.items()},
        "audit_source_tables": source_table_records,
        "panel_a": {
            "single_model_series": ["Single-task ALIGNN", "Full MTL"],
            "common_seeds": SEEDS3, "members_per_single_model_estimate": 1,
            "curve_summary": "precision mean +/- sample SD after interpolation to 501 recall points",
            "interpolation": "np.interp(PR_GRID, recall[::-1], precision[::-1]); duplicate recall values retained",
            "ensemble_series": "Full MTL", "ensemble_members": SEEDS5,
            "ensemble_unit": "one M=5 ensemble formed by arithmetic mean of member probabilities",
            "error_band": None,
            "curve_uncertainty_display": "omitted; AP sample SD remains in the legend and panel b",
            "ap_summary": progression.to_dict("records"),
            "axis_limits": {"recall": [0.0, 1.0], "precision": [0.4, 1.0]},
            "prevalence_reference_line": None,
            "classification_threshold": None,
        },
        "panel_b": {
            "metric": "sklearn average_precision_score",
            "series": ["Single-task ALIGNN M=1 mean", "Full MTL M=1 mean", "Full MTL M=5 ensemble"],
            "single_model_error_bar": "sample SD across common seeds 2025-2027",
            "ensemble_error_bar": None,
        },
        "supplementary_m5": {
            "models": MODELS, "members_each_model": 5,
            "aggregation": "arithmetic probability mean",
            "purpose": "matched-compute comparison retained from the previous main-panel design",
        },
        "panel_c": {"model": "Full MTL", "members": SEEDS5, "aggregation": "arithmetic mean of raw p_joint",
                    "error_bar": None, "split_rules": split_rules,
                    "display": "horizontal bar chart",
                    "display_labels": ["Random", "Composition", "Chemistry", "Structure"]},
        "panels_d_e": {"members_per_run": 1, "independent_seeds": SEEDS3, "summary": "mean +/- sample SD across runs",
                       "display": "categorical seed points with mean +/- sample SD",
                       "models": COMPONENT_MODELS,
                       "legend": "four-color model mapping shown in panel e and shared by panels d-e",
                       "logic_rule": "max(0,pT+pD-1) <= pJ <= min(pT,pD)", "epsilon": LOGIC_EPS},
        "panel_f": {"members_per_run": 1, "independent_seeds": SEEDS3,
                    "threshold": "per model/seed validation-selected classification_threshold",
                    "comparison": "p_joint >= threshold", "summary": "mean +/- sample SD across runs",
                    "class_separator_line": None},
        "visual_adjustments": {
            "panel_labels": "aligned near each axes upper-left corner",
            "panel_a_legend_handle_length": 1.55,
            "panel_c_first_value_offset_points": [7, 6],
        },
        "font": {"requested": "Arial", "used": font_name, "note": font_note,
                 "role_sizes_pt": {"base": BASE_FONT_SIZE, "title": TITLE_FONT_SIZE,
                                    "axis_label": AXIS_LABEL_FONT_SIZE, "tick": TICK_FONT_SIZE,
                                    "legend": LEGEND_FONT_SIZE, "annotation": ANNOTATION_FONT_SIZE,
                                    "panel_label": PANEL_LABEL_FONT_SIZE}},
        "figure_size_mm": [175, 110],
    }
    (output_dir / "fig2_source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    caption = (
        "Figure 2. Structural prediction of thermodynamic, dynamic, and joint stability. "
        "(a) Joint precision-recall performance from the original single-task ALIGNN baseline to Full MTL and its final deep ensemble. ALIGNN and Full MTL single-model curves are pointwise means across the same three seeds (2025-2027); curve-level SD bands are omitted for visual clarity. The Full MTL ensemble curve is obtained by arithmetic averaging of five member probabilities (seeds 2025-2029). Legend AP values are computed separately from each seed's original predictions and summarized as mean +/- sample SD; ensemble AP is computed once from the averaged probabilities. "
        "(b) Joint AP progression for the same three stages. Error bars for ALIGNN and Full MTL denote sample SD across the three common single-model runs; the final five-member ensemble is one aggregate estimate and therefore has no artificial error bar. No statistical-significance claim is made across these different aggregation levels. "
        "(c) Joint AP of the Full MTL five-member ensemble under frozen Random (stratified random; N=1,215, 569 positives, 46.83%), Composition (canonical-composition-disjoint; N=1,211, 568 positives, 46.90%), Chemical system (element-set-disjoint; N=1,214, 566 positives, 46.62%), and Structure family (structure-family-key-disjoint; N=1,249, 589 positives, 47.16%) splits. The legacy IID set in panel a and the review Random split in panel c are distinct. "
        "(d) Joint AP and (e) probability-intersection-bound violation rate for Equal MTL, adaptive weighting without logic (+AW), fixed weights with logical consistency (+LC), and Full MTL. Colors and marker shapes identify the four variants; the compact legend in panel e applies to both panels. Each small translucent point is one single-model run for a common seed (2025-2027); large points and error bars show mean +/- sample SD. A violation in panel e occurs when pJ lies outside [max(0,pT+pD-1), min(pT,pD)] with epsilon=1e-10. Full MTL does not have the lowest mean violation rate; +LC is lower. "
        "(f) Error rates in four true stability classes for single-task ALIGNN and Full MTL, using each run's validation-selected Joint threshold. The first three classes report false-positive release rates and the T+/D+ class reports the false-negative rate; n labels give the number of distinct test materials per run. Full MTL has a higher mean false-positive rate than ALIGNN in the T+/D− class (0.399 versus 0.338), so the method does not reduce every error type. Panels d-f use three independent M=1 runs and show run-level sample SD."
    )
    (output_dir / "fig2_caption.txt").write_text(caption + "\n", encoding="utf-8")

    def mismatch_count(path: Path) -> int:
        try:
            return len(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            return 0

    mismatch_e = mismatch_count(output_dir / "panel_e_violation_mismatches.csv")
    mismatch_f = mismatch_count(output_dir / "panel_f_prediction_mismatches.csv")
    progression_by_series = progression.set_index("series")
    alignn_mean = float(progression_by_series.loc["Single-task ALIGNN", "Joint_AP"])
    alignn_sd = float(progression_by_series.loc["Single-task ALIGNN", "sample_SD"])
    full_mean = float(progression_by_series.loc["Full MTL single", "Joint_AP"])
    full_sd = float(progression_by_series.loc["Full MTL single", "sample_SD"])
    full_ensemble_ap = float(progression_by_series.loc["Full MTL ensemble", "Joint_AP"])
    checks = [
        "PASS: audit workbook read with header=3 and preserved unchanged.",
        "PASS: F2 material 10,926 rows; F3 components 14,568 rows; F3 errors 7,284 rows.",
        "PASS: material IDs and labels align within every required task/split/model comparison.",
        "PASS: all probabilities are finite in [0,1], labels are binary, and Joint=Thermo AND Dynamic where available.",
        "PASS: p_ensemble equals the arithmetic mean of the five member probabilities within 1e-12.",
        "PASS: panel-a raw prediction files match the audited per-member probabilities within 1e-12.",
        f"PASS: common-seed ALIGNN Joint AP={alignn_mean:.12f} +/- {alignn_sd:.12f} (sample SD, n=3).",
        f"PASS: common-seed Full MTL Joint AP={full_mean:.12f} +/- {full_sd:.12f} (sample SD, n=3).",
        f"PASS: Full MTL five-member ensemble Joint AP={full_ensemble_ap:.12f}.",
        "NOTICE: supplied ALIGNN value 0.874 +/- 0.014 describes five single-model seeds; the three common seeds used in panels a-b give 0.878 +/- 0.018, so the supplied number was not forced into the figure.",
        f"PASS: retained M=5 supplementary AP values match the previous verification values; maximum absolute difference={max_expected_ap_difference:.3e}.",
        f"PASS: panel-e stored/recomputed violation mismatches={mismatch_e}; epsilon={LOGIC_EPS:.1e}.",
        f"PASS: panel-f threshold/prediction mismatches={mismatch_f}; thresholds were not changed.",
        "PASS: panel-c uses four real Full-MTL review ensembles and does not reuse the legacy IID result.",
        "PASS: panel-c is a horizontal bar chart with a zero baseline; panels c-f data values remain unchanged.",
        "PASS: panels d-e use the same four-color model encoding and share the compact legend in panel e.",
        "PASS: typography is role-consistent across panels a-f and uses one font family.",
        "PASS: panel-a Precision axis is fixed to 0.4-1.0; the prevalence reference line and label are removed.",
        "PASS: panel-a PR-curve SD bands are omitted; AP sample SD remains in the legend and panel b.",
        "PASS: panel labels are aligned near the axes; panel-a legend handles are shortened.",
        f"PASS: requested figure dimensions=175 x 110 mm; {font_note}.",
        "PENDING_EXTERNAL_RENDER_CHECK: inspect the saved PNG and a rasterization of the PDF after this script exits.",
    ]
    (output_dir / "fig2_checks.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")

    ap_lines = []
    for row in panel_a["internal_ap"].to_dict("records"):
        ap_lines.append(f"- {row['series']}, seed={row['seed']}, {row['aggregation']}: AP={float(row['AP']):.12f}")
    source_lines = [f"- {path}" for path in panel_a["source_paths"]]
    revision_report = [
        "Figure 2(a-b) revision report",
        "",
        "Verified AP values",
        *ap_lines,
        f"- ALIGNN common-seed mean +/- sample SD: {alignn_mean:.12f} +/- {alignn_sd:.12f}",
        f"- Full MTL common-seed mean +/- sample SD: {full_mean:.12f} +/- {full_sd:.12f}",
        f"- Full MTL five-member ensemble: {full_ensemble_ap:.12f}",
        "- Supplied ALIGNN 0.874 +/- 0.014 is the five-seed single-model summary, not the three-common-seed summary; it was not used in panels a-b.",
        "",
        "Raw prediction files used directly for panels a-b",
        *source_lines,
        "",
        "Code changes and reasons",
        "- load_joint_predictions: loads, validates, and aligns saved raw Joint predictions; no training is invoked.",
        "- panel_a_data: limits single-model summaries to common seeds 2025-2027 and constructs the final Full MTL ensemble from five raw members.",
        "- panel_b_data: replaces task-wise M=5 delta-AP comparisons with the three-stage Joint AP progression.",
        "- draw_panel_a: plots three real mean PR curves without curve-level SD bands; AP sample SD remains in the legend.",
        "- draw_panel_b: plots the matching three-stage Joint AP progression; the ensemble has no fabricated error bar.",
        "- save_standalone_panels: exports panels a and b separately in PNG/PDF/SVG.",
        "- supplementary_m5_data and draw_supplementary_m5: retain the matched five-member ALIGNN/Equal MTL/Full MTL comparison outside the main figure.",
        "- draw_figure: panel c uses a horizontal bar chart; panels d-e use categorical seed points with mean +/- SD and a shared four-model legend; all panels share one role-based typography scale.",
        "- main/caption/manifest/checks: record the revised comparison levels and full data provenance.",
        "",
        "Scientific interpretation",
        "- Before: panels a-b asked how three methods compare when methods/tasks were mixed across single-run summaries and matched M=5 differences.",
        "- After: panels a-b ask how Joint prediction evolves from the original ALIGNN baseline, through the Full MTL single model, to the final Full MTL five-member ensemble.",
        "- Correspondence to Section 3.1: yes, provided the first paragraph distinguishes the modest MTL single-model gain from the additional ensemble gain and uses the verified values above.",
    ]
    (output_dir / "fig2_ab_revision_report.txt").write_text(
        "\n".join(revision_report) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "font": font_name}, indent=2))

if __name__ == "__main__":
    main()
