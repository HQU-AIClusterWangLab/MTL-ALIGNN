from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image
from sklearn.metrics import average_precision_score


SEEDS = (2025, 2026, 2027, 2028, 2029)
MODEL_NAME = "Full MTL ALIGNN"
PROBABILITY_LOGIT_TOLERANCE = 1e-6
METRIC_TOLERANCE = 1e-12
SANITY_TOLERANCE = 0.02

# These approximate manuscript values are validation references only. They are never
# used as plotted values or written as computed source data.
SPLIT_SPECS = (
    {
        "split": "random_stratified",
        "audit_key": "random_iid_stratified",
        "display": "Random",
        "tick": "Random",
        "experiment": "R23_full_mtl_alignn_random",
        "split_relative": "review_splits/random_iid_stratified.csv",
        "target_group": None,
        "sanity_reference": 0.876,
    },
    {
        "split": "composition_disjoint",
        "audit_key": "composition_disjoint",
        "display": "Composition-disjoint",
        "tick": "Composition",
        "experiment": "R27_full_mtl_alignn_composition",
        "split_relative": "review_splits/composition_disjoint.csv",
        "target_group": "canonical_reduced_composition_id",
        "sanity_reference": 0.895,
    },
    {
        "split": "chemical_system_disjoint",
        "audit_key": "chemical_system_disjoint",
        "display": "Chemical-system-disjoint",
        "tick": "Chemical\nsystem",
        "experiment": "R31_full_mtl_alignn_chemical",
        "split_relative": "review_splits/chemical_system_disjoint.csv",
        "target_group": "chemical_system_id",
        "sanity_reference": 0.886,
    },
    {
        "split": "structure_family_disjoint",
        "audit_key": "structure_family_disjoint",
        "display": "Structure-family-disjoint",
        "tick": "Structure\nfamily",
        "experiment": "R35_full_mtl_alignn_structure",
        "split_relative": "review_splits/structure_family_disjoint.csv",
        "target_group": "structure_family_id",
        "sanity_reference": 0.725,
    },
)

SOURCE_DATA_COLUMNS = [
    "record_type",
    "split",
    "split_display_name",
    "split_file",
    "split_relative_file",
    "split_sha256",
    "experiment",
    "config_file",
    "resolved_experiment_file",
    "seed",
    "model",
    "joint_ap_single",
    "single_member_mean",
    "single_member_sample_sd",
    "ensemble_size",
    "joint_ap_ensemble_raw",
    "delta_ap_vs_random",
    "n_test",
    "n_joint_positive",
    "joint_positive_rate",
    "source_prediction_file",
    "source_prediction_sha256",
    "member_seeds",
    "aggregation_method",
    "calibrated",
]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate Fig. 4 from frozen Full MTL strict-split member predictions "
            "using an uncalibrated five-member probability-space ensemble."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="Project root containing results/, data/, train/, and scripts/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <project-root>/paper_figures/fig4).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    nonnegative = values >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    output[~nonnegative] = exp_values / (1.0 + exp_values)
    return output


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_csv(temporary, index=False, na_rep="NA")
    os.replace(temporary, path)


def atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_savefig(
    figure: mpl.figure.Figure,
    path: Path,
    *,
    file_format: str,
    dpi: int | None = None,
) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    metadata = {
        "Creator": "plot_fig4_generalization.py",
        "Title": "Generalization across data splits",
    }
    save_kwargs: dict[str, object] = {
        "format": file_format,
        "facecolor": "white",
    }
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    if file_format in {"pdf", "svg"}:
        save_kwargs["metadata"] = metadata
    figure.savefig(temporary, **save_kwargs)
    os.replace(temporary, path)


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise AssertionError(f"Missing columns in {source}: {missing}")


def load_yaml(path: Path) -> dict[str, object]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise AssertionError(f"Expected mapping in YAML file: {path}")
    return parsed


def validate_experiment_config(
    project_root: Path, spec: dict[str, object]
) -> dict[str, object]:
    experiment = str(spec["experiment"])
    config_path = (
        project_root / "train" / "experiments" / "review" / f"{experiment}.yaml"
    )
    resolved_path = (
        project_root / "results" / "review" / experiment / "resolved_experiment.yaml"
    )
    for path in (config_path, resolved_path):
        if not path.is_file():
            raise AssertionError(f"Required experiment configuration is missing: {path}")

    expected_split = str(spec["split_relative"])
    if Path(expected_split).name == "iid_split.csv":
        raise AssertionError("Legacy iid_split.csv is forbidden for Fig. 4")

    for path in (config_path, resolved_path):
        config = load_yaml(path)
        if config.get("name") != experiment:
            raise AssertionError(f"Experiment name mismatch in {path}")
        if config.get("category") != "review":
            raise AssertionError(f"Expected category=review in {path}")
        if config.get("model_type") != "alignn":
            raise AssertionError(f"Expected model_type=alignn in {path}")
        if tuple(int(seed) for seed in config.get("seeds", [])) != SEEDS:
            raise AssertionError(f"Expected exactly seeds {SEEDS} in {path}")
        if str(config.get("split_file")) != expected_split:
            raise AssertionError(f"Split file mismatch in {path}")
        if tuple(config.get("heads", [])) != ("thermo", "dynamic", "joint"):
            raise AssertionError(f"Unexpected heads in {path}")
        if tuple(config.get("active_tasks", [])) != ("thermo", "dynamic", "joint"):
            raise AssertionError(f"Unexpected active tasks in {path}")
        if config.get("weighting") != "adaptive_auxiliary":
            raise AssertionError(f"Expected Full MTL adaptive weighting in {path}")
        if not math.isclose(float(config.get("logic_beta", math.nan)), 0.10):
            raise AssertionError(f"Expected logic_beta=0.10 in {path}")
        if config.get("logic_lower_bound") is not True:
            raise AssertionError(f"Expected logic_lower_bound=true in {path}")

    return {
        "config_path": config_path.resolve(),
        "config_sha256": sha256_file(config_path),
        "resolved_path": resolved_path.resolve(),
        "resolved_sha256": sha256_file(resolved_path),
    }


def validate_manifest(
    project_root: Path, spec: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest_path = (
        project_root / "data" / "processed" / str(spec["split_relative"])
    )
    hash_registry_path = (
        project_root / "data" / "processed" / "review_splits" / "manifest_sha256.json"
    )
    split_audit_path = (
        project_root / "data" / "processed" / "review_splits" / "audit.json"
    )
    for path in (manifest_path, hash_registry_path, split_audit_path):
        if not path.is_file():
            raise AssertionError(f"Required split audit source is missing: {path}")

    manifest_hash = sha256_file(manifest_path)
    registry = json.loads(hash_registry_path.read_text(encoding="utf-8"))
    registered_hash = registry["output_files"].get(manifest_path.name)
    if manifest_hash != registered_hash:
        raise AssertionError(
            f"Split manifest hash differs from frozen registry: {manifest_path}"
        )

    audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    split_audit_hash = sha256_file(split_audit_path)
    if registry["output_files"].get("audit.json") != split_audit_hash:
        raise AssertionError(f"Split audit hash differs from frozen registry: {split_audit_path}")
    if audit.get("frozen_manifest_set") is not True:
        raise AssertionError(f"Review split manifests are not marked frozen: {split_audit_path}")
    audit_entry = audit["manifests"].get(str(spec["audit_key"]))
    if not isinstance(audit_entry, dict):
        raise AssertionError(f"Missing split entry in {split_audit_path}: {spec['audit_key']}")
    if audit_entry.get("manifest_sha256") != manifest_hash:
        raise AssertionError(f"Split audit hash conflict for {manifest_path}")
    expected_target = spec["target_group"]
    if audit_entry.get("target_group_column") != expected_target:
        raise AssertionError(f"Unexpected target group definition for {manifest_path}")
    if expected_target is None:
        if audit_entry.get("target_group_disjoint_enforced") is not False:
            raise AssertionError("Random stratified split is incorrectly marked disjoint")
    else:
        if audit_entry.get("target_group_disjoint_enforced") is not True:
            raise AssertionError(f"Disjoint grouping is not enforced for {manifest_path}")
        if int(audit_entry.get("target_group_overlap_unique_count", -1)) != 0:
            raise AssertionError(f"Target groups overlap across {manifest_path}")

    manifest = pd.read_csv(manifest_path, dtype={"material_id": str})
    require_columns(
        manifest,
        {"material_id", "row_id", "split", "joint_label"},
        manifest_path,
    )
    if manifest["material_id"].isna().any() or manifest["material_id"].duplicated().any():
        raise AssertionError(f"Manifest IDs are missing or duplicated: {manifest_path}")
    split_values = set(manifest["split"].astype(str).str.lower().unique())
    if split_values != {"train", "val", "test"}:
        raise AssertionError(f"Unexpected split labels in {manifest_path}: {split_values}")

    test = manifest.loc[
        manifest["split"].astype(str).str.lower().eq("test"),
        ["material_id", "row_id", "joint_label"],
    ].copy()
    test["material_id"] = test["material_id"].astype(str)
    test["row_id"] = pd.to_numeric(test["row_id"], errors="raise").astype(int)
    test["joint_label"] = pd.to_numeric(
        test["joint_label"], errors="raise"
    ).astype(int)
    if not set(test["joint_label"].unique()).issubset({0, 1}):
        raise AssertionError(f"Non-binary Joint labels in {manifest_path}")
    test = test.sort_values("material_id", kind="mergesort").reset_index(drop=True)
    n_test = int(len(test))
    n_positive = int(test["joint_label"].sum())
    positive_rate = float(test["joint_label"].mean())

    legacy_exclusion: dict[str, object] | None = None
    if str(spec["audit_key"]) == "random_iid_stratified":
        legacy_path = project_root / "data" / "processed" / "iid_split.csv"
        legacy_hash = sha256_file(legacy_path)
        registered_legacy_hash = None
        for path_text, path_hash in registry.get("source_files", {}).items():
            if Path(path_text).resolve() == legacy_path.resolve():
                registered_legacy_hash = path_hash
                break
        if registered_legacy_hash != legacy_hash:
            raise AssertionError("Legacy IID hash is inconsistent with the frozen split audit")
        legacy = pd.read_csv(legacy_path, dtype={"material_id": str})
        require_columns(legacy, {"material_id", "split"}, legacy_path)
        legacy_test = legacy.loc[
            legacy["split"].astype(str).str.lower().eq("test"), "material_id"
        ].astype(str)
        if set(legacy_test) == set(test["material_id"]):
            raise AssertionError("Strict Random test IDs unexpectedly equal legacy IID test IDs")
        legacy_exclusion = {
            "path": legacy_path.resolve(),
            "sha256": legacy_hash,
            "n_test": int(len(legacy_test)),
            "strict_random_n_test": n_test,
            "test_id_sets_identical": False,
            "legacy_only_id_count": int(
                len(set(legacy_test) - set(test["material_id"]))
            ),
            "strict_only_id_count": int(
                len(set(test["material_id"]) - set(legacy_test))
            ),
        }

    audit_summary = audit_entry["split_summary"]["test"]
    if int(audit_summary["n_rows"]) != n_test:
        raise AssertionError(f"Manifest/audit test-count conflict for {manifest_path}")
    if int(audit_summary["label_positive_counts"]["joint_label"]) != n_positive:
        raise AssertionError(f"Manifest/audit positive-count conflict for {manifest_path}")
    if not math.isclose(
        float(audit_summary["label_rates"]["joint_label"]),
        positive_rate,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise AssertionError(f"Manifest/audit positive-rate conflict for {manifest_path}")

    return test, {
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": manifest_hash,
        "manifest_registry_path": hash_registry_path.resolve(),
        "manifest_registry_sha256": sha256_file(hash_registry_path),
        "split_audit_path": split_audit_path.resolve(),
        "split_audit_sha256": split_audit_hash,
        "split_algorithm_version": audit.get("algorithm_version"),
        "frozen_manifest_set": True,
        "assignment_seed": audit_entry.get("seed"),
        "assignment_sha256": audit_entry.get("assignment_sha256"),
        "target_group_column": expected_target,
        "target_group_disjoint_enforced": bool(
            audit_entry.get("target_group_disjoint_enforced")
        ),
        "target_group_overlap_unique_count": audit_entry.get(
            "target_group_overlap_unique_count"
        ),
        "n_test": n_test,
        "n_joint_positive": n_positive,
        "joint_positive_rate": positive_rate,
        "sorted_material_id_sha256": sha256_lines(test["material_id"].tolist()),
        "sorted_material_id_label_sha256": sha256_lines(
            [
                f"{material_id}\t{label}"
                for material_id, label in zip(
                    test["material_id"], test["joint_label"]
                )
            ]
        ),
        "legacy_exclusion": legacy_exclusion,
    }


def validate_structure_family_definition(project_root: Path) -> dict[str, object]:
    prepare_path = project_root / "data" / "prepare_data.py"
    review_split_path = project_root / "data" / "make_review_splits.py"
    utils_path = project_root / "data" / "utils.py"
    prepare_text = prepare_path.read_text(encoding="utf-8")
    split_text = review_split_path.read_text(encoding="utf-8")
    utils_text = utils_path.read_text(encoding="utf-8")
    expected_expression = '"structure_family": f"{anon}|{layergroup}"'
    if expected_expression not in prepare_text:
        raise AssertionError(
            "Could not verify structure_family = anonymous_formula|layergroup"
        )
    if 'result["structure_family_id"]' not in split_text or (
        'result["structure_family"]' not in split_text
    ):
        raise AssertionError("Could not verify structure_family_id construction")
    if "Composition(formula).anonymized_formula" not in utils_text:
        raise AssertionError("Could not verify anonymous_formula construction")
    return {
        "definition": "anonymous_formula(formula)|layergroup",
        "prepare_data_path": prepare_path.resolve(),
        "prepare_data_sha256": sha256_file(prepare_path),
        "make_review_splits_path": review_split_path.resolve(),
        "make_review_splits_sha256": sha256_file(review_split_path),
        "anonymous_formula_path": utils_path.resolve(),
        "anonymous_formula_sha256": sha256_file(utils_path),
        "anonymous_formula_method": "pymatgen Composition(formula).anonymized_formula",
    }


def load_member_prediction(
    prediction_path: Path,
    manifest_test: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    frame = pd.read_csv(prediction_path, dtype={"material_id": str})
    require_columns(
        frame,
        {"material_id", "row_id", "y_joint", "p_joint", "logit_joint"},
        prediction_path,
    )
    calibrated_columns = [
        column
        for column in frame
        if "calibrat" in column.lower() or "temperature" in column.lower()
    ]
    if calibrated_columns:
        raise AssertionError(
            f"Calibrated fields are forbidden in raw source {prediction_path}: "
            f"{calibrated_columns}"
        )
    if frame["material_id"].isna().any() or frame["material_id"].duplicated().any():
        raise AssertionError(f"Missing/duplicate material IDs in {prediction_path}")
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise AssertionError(f"Missing/duplicate row IDs in {prediction_path}")

    compact = frame.loc[
        :, ["material_id", "row_id", "y_joint", "p_joint", "logit_joint"]
    ].copy()
    compact["material_id"] = compact["material_id"].astype(str)
    compact["row_id"] = pd.to_numeric(compact["row_id"], errors="raise").astype(int)
    compact["y_joint"] = pd.to_numeric(
        compact["y_joint"], errors="raise"
    ).astype(int)
    compact["p_joint"] = pd.to_numeric(
        compact["p_joint"], errors="raise"
    ).astype(float)
    compact["logit_joint"] = pd.to_numeric(
        compact["logit_joint"], errors="raise"
    ).astype(float)
    if not set(compact["y_joint"].unique()).issubset({0, 1}):
        raise AssertionError(f"Non-binary Joint labels in {prediction_path}")
    if not np.isfinite(compact[["p_joint", "logit_joint"]].to_numpy(float)).all():
        raise AssertionError(f"Non-finite raw probabilities/logits in {prediction_path}")
    if not compact["p_joint"].between(0.0, 1.0, inclusive="both").all():
        raise AssertionError(f"Raw probabilities outside [0,1] in {prediction_path}")

    max_sigmoid_difference = float(
        np.max(
            np.abs(
                compact["p_joint"].to_numpy(float)
                - stable_sigmoid(compact["logit_joint"].to_numpy(float))
            )
        )
    )
    if max_sigmoid_difference > PROBABILITY_LOGIT_TOLERANCE:
        raise AssertionError(
            f"Stored p_joint is inconsistent with raw logit_joint in {prediction_path}: "
            f"{max_sigmoid_difference:.12g}"
        )

    compact = compact.sort_values("material_id", kind="mergesort").reset_index(drop=True)
    if not compact["material_id"].equals(manifest_test["material_id"]):
        raise AssertionError(f"Prediction IDs do not match split manifest: {prediction_path}")
    if not compact["row_id"].equals(manifest_test["row_id"]):
        raise AssertionError(f"Prediction row IDs do not match manifest: {prediction_path}")
    if not np.array_equal(
        compact["y_joint"].to_numpy(int),
        manifest_test["joint_label"].to_numpy(int),
    ):
        raise AssertionError(f"Prediction labels do not match manifest: {prediction_path}")
    return compact, max_sigmoid_difference


def validate_single_member_metric(
    experiment_dir: Path,
    experiment: str,
    seed: int,
    recomputed_ap: float,
) -> dict[str, object]:
    metrics_path = experiment_dir / f"seed_{seed}" / "metrics.json"
    completed_path = experiment_dir / f"seed_{seed}" / "COMPLETED.json"
    root_resolved_path = experiment_dir / "resolved_experiment.yaml"
    seed_resolved_path = experiment_dir / f"seed_{seed}" / "resolved_experiment.yaml"
    for path in (metrics_path, completed_path, root_resolved_path, seed_resolved_path):
        if not path.is_file():
            raise AssertionError(f"Required frozen run record is missing: {path}")
    root_resolved_hash = sha256_file(root_resolved_path)
    seed_resolved_hash = sha256_file(seed_resolved_path)
    if seed_resolved_hash != root_resolved_hash:
        raise AssertionError(
            f"Seed-specific resolved config differs from experiment config: "
            f"{seed_resolved_path}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if int(metrics.get("seed", -1)) != seed:
        raise AssertionError(f"Seed mismatch in {metrics_path}")
    if completed.get("status") != "completed":
        raise AssertionError(f"Run is not frozen-completed: {completed_path}")
    if completed.get("name") != experiment or int(completed.get("seed", -1)) != seed:
        raise AssertionError(f"Completion identity mismatch in {completed_path}")
    frozen_ap = float(metrics["test_joint"]["pr_auc"])
    completed_ap = float(completed["test_joint_pr_auc"])
    if not math.isclose(frozen_ap, completed_ap, rel_tol=0.0, abs_tol=METRIC_TOLERANCE):
        raise AssertionError(f"metrics/COMPLETED AP conflict for {experiment}, seed {seed}")
    if not math.isclose(
        recomputed_ap, frozen_ap, rel_tol=0.0, abs_tol=METRIC_TOLERANCE
    ):
        raise AssertionError(
            f"Frozen prediction AP does not reproduce metrics.json for {experiment}, "
            f"seed {seed}: {recomputed_ap} vs {frozen_ap}"
        )
    return {
        "metrics_path": metrics_path.resolve(),
        "metrics_sha256": sha256_file(metrics_path),
        "completed_path": completed_path.resolve(),
        "completed_sha256": sha256_file(completed_path),
        "seed_resolved_path": seed_resolved_path.resolve(),
        "seed_resolved_sha256": seed_resolved_hash,
    }


def validate_seed_aggregate(
    project_root: Path,
    experiment_dir: Path,
    experiment: str,
    member_mean: float,
    member_sd: float,
    ensemble_ap: float,
    n_test: int,
    n_positive: int,
    source_paths: dict[int, Path],
    source_hashes: dict[int, str],
    sorted_material_id_sha256: str,
) -> dict[str, object]:
    local_summary_path = experiment_dir / "metrics_mean_std.csv"
    global_summary_path = (
        project_root
        / "results"
        / "review"
        / "strict_gnn_summary_all"
        / "experiment_summary.csv"
    )
    paired_summary_path = global_summary_path.with_name("paired_comparisons.csv")
    strict_status_path = global_summary_path.with_name("status.json")
    strict_summary_json_path = global_summary_path.with_name("summary.json")
    pairing_audit_path = global_summary_path.with_name("pairing_audit.csv")
    summarizer_path = project_root / "train" / "summarize_review_gnn.py"
    local = pd.read_csv(local_summary_path)
    local_row = local.loc[local["metric"].eq("test_joint.pr_auc")]
    if len(local_row) != 1:
        raise AssertionError(f"Missing unique test_joint.pr_auc in {local_summary_path}")
    local_mean = float(local_row.iloc[0]["mean"])
    local_sd = float(local_row.iloc[0]["std"])
    if not math.isclose(local_mean, member_mean, rel_tol=0.0, abs_tol=METRIC_TOLERANCE):
        raise AssertionError(f"Local seed-mean conflict for {experiment}")
    if not math.isclose(local_sd, member_sd, rel_tol=0.0, abs_tol=METRIC_TOLERANCE):
        raise AssertionError(f"Local seed-SD conflict for {experiment}")

    global_summary = pd.read_csv(global_summary_path)
    global_row = global_summary.loc[
        global_summary["experiment"].eq(experiment)
        & global_summary["subset"].eq("all_test")
        & global_summary["task"].eq("joint")
    ]
    if len(global_row) != 1:
        raise AssertionError(f"Missing unique strict summary row for {experiment}")
    global_mean = float(global_row.iloc[0]["pr_auc_mean"])
    global_sd = float(global_row.iloc[0]["pr_auc_std"])
    if not math.isclose(global_mean, member_mean, rel_tol=0.0, abs_tol=METRIC_TOLERANCE):
        raise AssertionError(f"Global strict seed-mean conflict for {experiment}")
    if not math.isclose(global_sd, member_sd, rel_tol=0.0, abs_tol=METRIC_TOLERANCE):
        raise AssertionError(f"Global strict seed-SD conflict for {experiment}")

    paired = pd.read_csv(paired_summary_path)
    paired_rows = paired.loc[
        paired["candidate_experiment"].eq(experiment)
        & paired["subset"].eq("all_test")
        & paired["task"].eq("joint")
        & paired["metric"].eq("pr_auc")
    ]
    if paired_rows.empty:
        raise AssertionError(f"Missing formal raw ensemble cross-check for {experiment}")
    unique_ensemble_values = paired_rows["candidate_ensemble_metric"].drop_duplicates()
    if len(unique_ensemble_values) != 1:
        raise AssertionError(f"Conflicting strict ensemble values for {experiment}")
    formal_ensemble_ap = float(unique_ensemble_values.iloc[0])
    if not math.isclose(
        formal_ensemble_ap, ensemble_ap, rel_tol=0.0, abs_tol=METRIC_TOLERANCE
    ):
        raise AssertionError(f"Formal strict raw ensemble conflict for {experiment}")
    if set(paired_rows["n_seeds"].astype(int)) != {len(SEEDS)}:
        raise AssertionError(f"Formal strict ensemble member-count conflict for {experiment}")
    if set(paired_rows["n_samples"].astype(int)) != {n_test}:
        raise AssertionError(f"Formal strict ensemble test-count conflict for {experiment}")
    if set(paired_rows["n_positive"].astype(int)) != {n_positive}:
        raise AssertionError(f"Formal strict ensemble positive-count conflict for {experiment}")
    pairing = pd.read_csv(pairing_audit_path)
    pairing_rows = pairing.loc[
        pairing["candidate_experiment"].eq(experiment)
        & pairing["subset"].eq("all_test")
        & pairing["task"].eq("joint")
    ]
    if pairing_rows.empty or set(pairing_rows["test_material_ids_sha256"]) != {
        sorted_material_id_sha256
    }:
        raise AssertionError(f"Formal strict material-ID hash conflict for {experiment}")
    if set(pairing_rows["paired_across_variants_and_seeds"].astype(bool)) != {True}:
        raise AssertionError(f"Formal strict pairing audit failed for {experiment}")
    strict_status = json.loads(strict_status_path.read_text(encoding="utf-8"))
    if strict_status.get("state") != "completed" or (
        strict_status.get("summary_artifacts_valid") is not True
    ):
        raise AssertionError(f"Strict review summary is not completed: {strict_status_path}")
    if tuple(int(seed) for seed in strict_status.get("required_seeds", [])) != SEEDS:
        raise AssertionError(f"Strict review summary uses unexpected seeds: {strict_status_path}")
    strict_summary = json.loads(strict_summary_json_path.read_text(encoding="utf-8"))
    if strict_summary.get("state") != "completed" or (
        strict_summary.get("default_review_five_seed_set") is not True
    ):
        raise AssertionError(f"Strict review frozen summary is invalid: {strict_summary_json_path}")
    if tuple(int(seed) for seed in strict_summary.get("seeds", [])) != SEEDS:
        raise AssertionError(f"Frozen strict summary uses unexpected seeds: {strict_summary_json_path}")
    if Path(str(strict_summary.get("review_results_dir"))).resolve() != (
        project_root / "results" / "review"
    ).resolve():
        raise AssertionError("Frozen strict summary points to a non-canonical review directory")
    frozen_manifest = {
        str(Path(str(record["path"])).resolve()).casefold(): record
        for record in strict_summary.get("input_manifest", [])
    }
    for seed in SEEDS:
        key = str(source_paths[seed].resolve()).casefold()
        if key not in frozen_manifest:
            raise AssertionError(
                f"Frozen strict input manifest omits {source_paths[seed]}"
            )
        frozen_record = frozen_manifest[key]
        if frozen_record.get("sha256") != source_hashes[seed]:
            raise AssertionError(
                f"Frozen strict input hash conflict for {source_paths[seed]}"
            )
        if int(frozen_record.get("bytes", -1)) != source_paths[seed].stat().st_size:
            raise AssertionError(
                f"Frozen strict input byte-count conflict for {source_paths[seed]}"
            )
    summarizer_text = summarizer_path.read_text(encoding="utf-8")
    raw_probability_expression = (
        'loaded[(spec.experiment, int(seed))][0][f"p_{spec.task}"].to_numpy(float)'
    )
    if (
        "test_probability = np.mean(" not in summarizer_text
        or raw_probability_expression not in summarizer_text
    ):
        raise AssertionError("Could not verify raw probability-space strict ensemble method")
    return {
        "local_summary_path": local_summary_path.resolve(),
        "local_summary_sha256": sha256_file(local_summary_path),
        "global_summary_path": global_summary_path.resolve(),
        "global_summary_sha256": sha256_file(global_summary_path),
        "local_mean": local_mean,
        "local_sd": local_sd,
        "global_mean": global_mean,
        "global_sd": global_sd,
        "paired_summary_path": paired_summary_path.resolve(),
        "paired_summary_sha256": sha256_file(paired_summary_path),
        "formal_raw_ensemble_ap": formal_ensemble_ap,
        "strict_status_path": strict_status_path.resolve(),
        "strict_status_sha256": sha256_file(strict_status_path),
        "strict_summary_json_path": strict_summary_json_path.resolve(),
        "strict_summary_json_sha256": sha256_file(strict_summary_json_path),
        "frozen_member_manifest_matches": len(SEEDS),
        "pairing_audit_path": pairing_audit_path.resolve(),
        "pairing_audit_sha256": sha256_file(pairing_audit_path),
        "formal_test_material_ids_sha256": sorted_material_id_sha256,
        "summarizer_path": summarizer_path.resolve(),
        "summarizer_sha256": sha256_file(summarizer_path),
    }


def audit_duplicate_review_copy(
    project_root: Path,
    experiment: str,
    source_paths: dict[int, Path],
    source_hashes: dict[int, str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    results_root = project_root / "results"
    for candidate_root in results_root.iterdir():
        if not candidate_root.is_dir() or candidate_root.name == "review":
            continue
        if not candidate_root.name.startswith("review"):
            continue
        for seed in SEEDS:
            candidate = (
                candidate_root / experiment / f"seed_{seed}" / "test_predictions.csv"
            )
            if candidate.is_file():
                candidate_hash = sha256_file(candidate)
                identical = candidate_hash == source_hashes[seed]
                if not identical:
                    raise AssertionError(
                        f"Conflicting duplicate prediction file: {candidate} vs "
                        f"{source_paths[seed]}"
                    )
                records.append(
                    {
                        "source": source_paths[seed],
                        "duplicate": candidate.resolve(),
                        "sha256": candidate_hash,
                        "identical": True,
                    }
                )
    return records


def load_all_data(project_root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if tuple(SEEDS) != (2025, 2026, 2027, 2028, 2029):
        raise AssertionError("Fig. 4 requires exactly seeds 2025--2029")
    structure_definition = validate_structure_family_definition(project_root)
    source_rows: list[dict[str, object]] = []
    split_results: list[dict[str, object]] = []

    for spec in SPLIT_SPECS:
        config_audit = validate_experiment_config(project_root, spec)
        manifest_test, manifest_audit = validate_manifest(project_root, spec)
        experiment = str(spec["experiment"])
        experiment_dir = project_root / "results" / "review" / experiment
        member_frames: dict[int, pd.DataFrame] = {}
        member_aps: dict[int, float] = {}
        member_audits: dict[int, dict[str, object]] = {}
        source_paths: dict[int, Path] = {}
        source_hashes: dict[int, str] = {}

        reference_ids: pd.Series | None = None
        reference_labels: np.ndarray | None = None
        for seed in SEEDS:
            prediction_path = experiment_dir / f"seed_{seed}" / "test_predictions.csv"
            if not prediction_path.is_file():
                raise AssertionError(f"Missing frozen member prediction: {prediction_path}")
            member, max_sigmoid_difference = load_member_prediction(
                prediction_path, manifest_test
            )
            if reference_ids is None:
                reference_ids = member["material_id"].copy()
                reference_labels = member["y_joint"].to_numpy(int)
            else:
                if not member["material_id"].equals(reference_ids):
                    raise AssertionError(f"Member ID alignment conflict for {experiment}")
                if not np.array_equal(
                    member["y_joint"].to_numpy(int), reference_labels
                ):
                    raise AssertionError(f"Member label conflict for {experiment}")

            member_ap = float(
                average_precision_score(
                    member["y_joint"].to_numpy(int),
                    member["p_joint"].to_numpy(float),
                )
            )
            if not 0.0 <= member_ap <= 1.0:
                raise AssertionError(f"Member AP outside [0,1]: {prediction_path}")
            frozen_audit = validate_single_member_metric(
                experiment_dir, experiment, seed, member_ap
            )
            prediction_hash = sha256_file(prediction_path)
            source_paths[seed] = prediction_path.resolve()
            source_hashes[seed] = prediction_hash
            member_frames[seed] = member
            member_aps[seed] = member_ap
            member_audits[seed] = {
                "prediction_path": prediction_path.resolve(),
                "prediction_sha256": prediction_hash,
                "prediction_bytes": prediction_path.stat().st_size,
                "columns": list(pd.read_csv(prediction_path, nrows=0).columns),
                "max_abs_p_joint_minus_sigmoid_logit_joint": max_sigmoid_difference,
                "joint_ap": member_ap,
                **frozen_audit,
            }

        if tuple(sorted(member_frames)) != SEEDS:
            raise AssertionError(f"Unexpected member seed set for {experiment}")
        probability_matrix = np.column_stack(
            [member_frames[seed]["p_joint"].to_numpy(float) for seed in SEEDS]
        )
        raw_ensemble_probability = probability_matrix.mean(axis=1, dtype=np.float64)
        if not np.isfinite(raw_ensemble_probability).all() or not (
            (raw_ensemble_probability >= 0.0) & (raw_ensemble_probability <= 1.0)
        ).all():
            raise AssertionError(f"Invalid raw ensemble probability for {experiment}")
        ensemble_ap = float(
            average_precision_score(
                manifest_test["joint_label"].to_numpy(int), raw_ensemble_probability
            )
        )
        member_values = np.asarray([member_aps[seed] for seed in SEEDS], dtype=float)
        member_mean = float(np.mean(member_values))
        member_sd = float(np.std(member_values, ddof=1))
        summary_crosscheck = validate_seed_aggregate(
            project_root,
            experiment_dir,
            experiment,
            member_mean,
            member_sd,
            ensemble_ap,
            int(manifest_audit["n_test"]),
            int(manifest_audit["n_joint_positive"]),
            source_paths,
            source_hashes,
            str(manifest_audit["sorted_material_id_sha256"]),
        )
        sanity_difference = ensemble_ap - float(spec["sanity_reference"])
        if abs(sanity_difference) > SANITY_TOLERANCE:
            raise AssertionError(
                f"Raw ensemble AP for {spec['display']} differs clearly from the "
                f"manuscript sanity value: recomputed={ensemble_ap:.12f}, "
                f"reference={float(spec['sanity_reference']):.12f}, "
                f"difference={sanity_difference:.12f}"
            )
        duplicate_records = audit_duplicate_review_copy(
            project_root, experiment, source_paths, source_hashes
        )

        for seed in SEEDS:
            source_rows.append(
                {
                    "record_type": "single_member",
                    "split": spec["split"],
                    "split_display_name": spec["display"],
                    "split_file": str(manifest_audit["manifest_path"]),
                    "split_relative_file": spec["split_relative"],
                    "split_sha256": manifest_audit["manifest_sha256"],
                    "experiment": experiment,
                    "config_file": str(config_audit["config_path"]),
                    "resolved_experiment_file": str(config_audit["resolved_path"]),
                    "seed": seed,
                    "model": MODEL_NAME,
                    "joint_ap_single": member_aps[seed],
                    "single_member_mean": member_mean,
                    "single_member_sample_sd": member_sd,
                    "ensemble_size": 1,
                    "joint_ap_ensemble_raw": math.nan,
                    "n_test": manifest_audit["n_test"],
                    "n_joint_positive": manifest_audit["n_joint_positive"],
                    "joint_positive_rate": manifest_audit["joint_positive_rate"],
                    "source_prediction_file": str(source_paths[seed]),
                    "source_prediction_sha256": source_hashes[seed],
                    "member_seeds": str(seed),
                    "aggregation_method": "none; one frozen raw member p_joint",
                    "calibrated": False,
                }
            )

        aggregation_method = (
            "material_id-aligned arithmetic mean of stored raw p_joint in probability "
            "space across seeds 2025,2026,2027,2028,2029"
        )
        source_rows.append(
            {
                "record_type": "raw_ensemble",
                "split": spec["split"],
                "split_display_name": spec["display"],
                "split_file": str(manifest_audit["manifest_path"]),
                "split_relative_file": spec["split_relative"],
                "split_sha256": manifest_audit["manifest_sha256"],
                "experiment": experiment,
                "config_file": str(config_audit["config_path"]),
                "resolved_experiment_file": str(config_audit["resolved_path"]),
                "seed": pd.NA,
                "model": MODEL_NAME,
                "joint_ap_single": math.nan,
                "single_member_mean": member_mean,
                "single_member_sample_sd": member_sd,
                "ensemble_size": 5,
                "joint_ap_ensemble_raw": ensemble_ap,
                "n_test": manifest_audit["n_test"],
                "n_joint_positive": manifest_audit["n_joint_positive"],
                "joint_positive_rate": manifest_audit["joint_positive_rate"],
                "source_prediction_file": ";".join(
                    str(source_paths[seed]) for seed in SEEDS
                ),
                "source_prediction_sha256": ";".join(
                    f"{seed}:{source_hashes[seed]}" for seed in SEEDS
                ),
                "member_seeds": ",".join(str(seed) for seed in SEEDS),
                "aggregation_method": aggregation_method,
                "calibrated": False,
            }
        )
        split_results.append(
            {
                "spec": spec,
                "config_audit": config_audit,
                "manifest_audit": manifest_audit,
                "member_aps": member_aps,
                "member_mean": member_mean,
                "member_sd": member_sd,
                "member_audits": member_audits,
                "ensemble_ap": ensemble_ap,
                "sanity_difference": sanity_difference,
                "summary_crosscheck": summary_crosscheck,
                "duplicate_records": duplicate_records,
                "aggregation_method": aggregation_method,
                "structure_definition": structure_definition,
            }
        )

    random_ensemble_ap = float(split_results[0]["ensemble_ap"])
    for result in split_results:
        result["delta_vs_random"] = float(result["ensemble_ap"]) - random_ensemble_ap

    source_data = pd.DataFrame(source_rows, columns=SOURCE_DATA_COLUMNS)
    source_data["seed"] = pd.array(source_data["seed"], dtype="Int64")
    ensemble_mask = source_data["record_type"].eq("raw_ensemble")
    source_data.loc[ensemble_mask, "delta_ap_vs_random"] = (
        source_data.loc[ensemble_mask, "joint_ap_ensemble_raw"] - random_ensemble_ap
    )
    if len(source_data) != len(SPLIT_SPECS) * (len(SEEDS) + 1):
        raise AssertionError("Unexpected Fig. 4 source-data row count")
    if source_data["calibrated"].astype(bool).any():
        raise AssertionError("Source data unexpectedly includes calibrated=True")
    for spec in SPLIT_SPECS:
        group = source_data.loc[source_data["split"].eq(spec["split"])]
        member_group = group.loc[group["record_type"].eq("single_member")]
        ensemble_group = group.loc[group["record_type"].eq("raw_ensemble")]
        if tuple(member_group["seed"].astype(int).tolist()) != SEEDS:
            raise AssertionError(f"Source CSV seed contract failed for {spec['display']}")
        if len(ensemble_group) != 1 or int(ensemble_group.iloc[0]["ensemble_size"]) != 5:
            raise AssertionError(f"Source CSV ensemble contract failed for {spec['display']}")
    return source_data, split_results


def print_dumbbell_summary(split_results: list[dict[str, object]]) -> None:
    print("[FIG4 DUMBBELL SUMMARY]")
    for index, result in enumerate(split_results):
        spec = result["spec"]
        manifest = result["manifest_audit"]
        label = str(spec["tick"]).replace("\n", " ")
        print(f"{label}:")
        print(f"  AP = {result['ensemble_ap']:.15f}")
        if index > 0:
            print(f"  delta_vs_random = {result['delta_vs_random']:+.15f}")
        sources = ";".join(
            str(result["member_audits"][seed]["prediction_path"])
            for seed in SEEDS
        )
        print(f"  source = {sources}")
        print(f"  n_test = {manifest['n_test']}")
        print(f"  positive_rate = {manifest['joint_positive_rate']:.15f}")
        print()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.8,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.3,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.3,
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


def automatic_x_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    lower = float(np.min(values)) - 0.025
    upper = float(np.max(values)) + 0.025
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    if lower <= 0.0:
        raise AssertionError("Fig. 4 x-axis must not start at zero")
    return lower, upper


def create_figure(
    split_results: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(4.8, 3.4))
    figure.subplots_adjust(left=0.29, right=0.98, bottom=0.20, top=0.91)

    y = np.arange(len(split_results) - 1, -1, -1, dtype=float)
    ensemble_values = np.asarray(
        [result["ensemble_ap"] for result in split_results], dtype=float
    )
    reference_ap = float(ensemble_values[0])
    color_main = "#24577A"
    color_reference = "#B9BEC3"
    color_segment = "#8E979E"
    color_guide = "#ECEFF1"
    color_text = "#26333B"
    color_delta = "#5F686F"

    axis.axvline(
        reference_ap,
        color=color_reference,
        linewidth=0.95,
        linestyle=(0, (3.0, 3.0)),
        zorder=1,
    )
    for row_y, value in zip(y[1:], ensemble_values[1:]):
        axis.hlines(
            row_y,
            min(reference_ap, float(value)),
            max(reference_ap, float(value)),
            color=color_segment,
            linewidth=1.15,
            zorder=2,
        )
    axis.scatter(
        ensemble_values,
        y,
        s=62,
        color=color_main,
        edgecolor="none",
        linewidth=0.0,
        zorder=5,
    )
    x_lower, x_upper = automatic_x_limits(ensemble_values)
    value_offset = 0.012 * (x_upper - x_lower)
    for row_y, value in zip(y, ensemble_values):
        value_is_left_of_reference = float(value) < reference_ap
        axis.text(
            float(value) - value_offset if value_is_left_of_reference else float(value) + value_offset,
            row_y,
            f"{value:.3f}",
            ha="right" if value_is_left_of_reference else "left",
            va="center",
            fontsize=8.2,
            color=color_text,
            zorder=6,
        )
    for row_y, result in zip(y[1:], split_results[1:]):
        value = float(result["ensemble_ap"])
        midpoint = 0.5 * (reference_ap + value)
        axis.text(
            midpoint,
            row_y + 0.18,
            f"ΔAP {float(result['delta_vs_random']):+.3f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=color_delta,
            zorder=6,
        )
    axis.text(
        reference_ap - value_offset,
        y[0] + 0.36,
        "Random reference",
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=color_delta,
        zorder=6,
    )

    axis.set_yticks(y, [str(result["spec"]["tick"]).replace("\n", " ") for result in split_results])
    axis.set_ylim(-0.48, 3.58)
    axis.set_xlim(x_lower, x_upper)
    axis.set_xlabel("Joint average precision")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.xaxis.grid(True, color=color_guide, linewidth=0.50, alpha=0.78)
    axis.yaxis.grid(False)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#4F565C")
    axis.spines["bottom"].set_color("#4F565C")
    axis.tick_params(colors="#30363B")

    pdf_path = output_dir / "Fig4_generalization_dumbbell.pdf"
    png_path = output_dir / "Fig4_generalization_dumbbell.png"
    svg_path = output_dir / "Fig4_generalization_dumbbell.svg"
    atomic_savefig(figure, pdf_path, file_format="pdf")
    atomic_savefig(figure, png_path, file_format="png", dpi=600)
    atomic_savefig(figure, svg_path, file_format="svg")
    plt.close(figure)
    return {"pdf": pdf_path, "png": png_path, "svg": svg_path}


def validate_written_outputs(
    source_data_path: Path,
    figure_paths: dict[str, Path],
) -> dict[str, object]:
    readback = pd.read_csv(source_data_path)
    require_columns(readback, set(SOURCE_DATA_COLUMNS), source_data_path)
    if len(readback) != 24:
        raise AssertionError(f"Unexpected source-data row count: {len(readback)}")
    calibrated_values = readback["calibrated"].astype(str).str.lower().unique().tolist()
    if calibrated_values != ["false"]:
        raise AssertionError(f"CSV calibration contract failed: {calibrated_values}")
    if set(readback.loc[readback["record_type"].eq("raw_ensemble"), "ensemble_size"]) != {5}:
        raise AssertionError("CSV raw ensemble rows must all use ensemble_size=5")
    random_rows = readback.loc[
        readback["split"].eq("random_stratified")
        & readback["record_type"].eq("raw_ensemble")
    ]
    if len(random_rows) != 1:
        raise AssertionError("CSV must contain one Random raw-ensemble reference row")
    random_ap = float(random_rows.iloc[0]["joint_ap_ensemble_raw"])
    for spec in SPLIT_SPECS:
        group = readback.loc[readback["split"].eq(spec["split"])]
        members = group.loc[group["record_type"].eq("single_member")]
        ensembles = group.loc[group["record_type"].eq("raw_ensemble")]
        if tuple(members["seed"].astype(int).tolist()) != SEEDS:
            raise AssertionError(f"CSV readback seed contract failed for {spec['display']}")
        if members["joint_ap_single"].isna().any():
            raise AssertionError(f"CSV readback member AP is missing for {spec['display']}")
        if len(ensembles) != 1 or not pd.isna(ensembles.iloc[0]["seed"]):
            raise AssertionError(f"CSV readback ensemble seed must be NA for {spec['display']}")
        if pd.isna(ensembles.iloc[0]["joint_ap_ensemble_raw"]):
            raise AssertionError(f"CSV readback ensemble AP is missing for {spec['display']}")
        if ensembles.iloc[0]["member_seeds"] != "2025,2026,2027,2028,2029":
            raise AssertionError(f"CSV readback member list failed for {spec['display']}")
        expected_delta = float(ensembles.iloc[0]["joint_ap_ensemble_raw"]) - random_ap
        observed_delta = float(ensembles.iloc[0]["delta_ap_vs_random"])
        if not math.isclose(
            observed_delta, expected_delta, rel_tol=0.0, abs_tol=METRIC_TOLERANCE
        ):
            raise AssertionError(f"CSV delta-vs-Random contract failed for {spec['display']}")
        if members["delta_ap_vs_random"].notna().any():
            raise AssertionError(f"CSV member rows must not carry split deltas for {spec['display']}")

    for path in figure_paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing/empty figure output: {path}")
    if not figure_paths["pdf"].read_bytes().startswith(b"%PDF"):
        raise AssertionError("PDF output does not have a PDF header")
    svg_text = figure_paths["svg"].read_text(encoding="utf-8")
    if "<image" in svg_text.lower():
        raise AssertionError("SVG unexpectedly contains a raster image")
    forbidden_svg_terms = (
        "temperature",
        "calibrated",
        "reliability",
        "uncertainty",
        "error auroc",
    )
    if any(term in svg_text.lower() for term in forbidden_svg_terms):
        raise AssertionError("SVG contains forbidden calibration/uncertainty text")
    ensemble_rows = readback.loc[readback["record_type"].eq("raw_ensemble")]
    dynamic_delta_labels = [
        f"ΔAP {float(delta):+.3f}"
        for delta in ensemble_rows.loc[
            ~ensemble_rows["split"].eq("random_stratified"),
            "delta_ap_vs_random",
        ]
    ]
    for required_text in (
        "Random reference",
        "Joint average precision",
        *dynamic_delta_labels,
    ):
        if required_text not in svg_text:
            raise AssertionError(f"SVG is missing required dumbbell label: {required_text}")

    with Image.open(figure_paths["png"]) as image:
        width, height = image.size
        dpi = image.info.get("dpi", (math.nan, math.nan))
    if width < 2500 or height < 1800:
        raise AssertionError(f"PNG is unexpectedly small: {width}x{height}")
    if not (599.0 <= float(dpi[0]) <= 601.0 and 599.0 <= float(dpi[1]) <= 601.0):
        raise AssertionError(f"PNG does not report 600 dpi: {dpi}")
    return {
        "png_width_px": width,
        "png_height_px": height,
        "png_dpi": dpi,
        "figure_sha256": {
            name: sha256_file(path) for name, path in figure_paths.items()
        },
        "source_data_sha256": sha256_file(source_data_path),
    }


def verify_sources_unchanged(
    split_results: list[dict[str, object]],
) -> None:
    for result in split_results:
        if sha256_file(Path(result["manifest_audit"]["manifest_path"])) != result[
            "manifest_audit"
        ]["manifest_sha256"]:
            raise AssertionError("Split manifest changed during Fig. 4 generation")
        for seed in SEEDS:
            member = result["member_audits"][seed]
            if sha256_file(Path(member["prediction_path"])) != member[
                "prediction_sha256"
            ]:
                raise AssertionError("Frozen prediction changed during Fig. 4 generation")
            if sha256_file(Path(member["metrics_path"])) != member["metrics_sha256"]:
                raise AssertionError("Frozen metrics changed during Fig. 4 generation")
            if sha256_file(Path(member["completed_path"])) != member[
                "completed_sha256"
            ]:
                raise AssertionError("Completion record changed during Fig. 4 generation")
            if sha256_file(Path(member["seed_resolved_path"])) != member[
                "seed_resolved_sha256"
            ]:
                raise AssertionError("Seed resolved config changed during Fig. 4 generation")


def build_audit_text(
    project_root: Path,
    script_path: Path,
    source_data_path: Path,
    figure_paths: dict[str, Path],
    output_checks: dict[str, object],
    split_results: list[dict[str, object]],
    started_at: datetime,
    finished_at: datetime,
    elapsed_seconds: float,
) -> str:
    lines: list[str] = []
    append = lines.append
    append("FIG. 4 DUMBBELL DATA AUDIT")
    append("==========================")
    append("")
    append("Purpose: Results & Discussion 3.3 - Generalization and applicability domain")
    append(f"Project root: {project_root.resolve()}")
    append(f"Plotting script: {script_path.resolve()}")
    append(f"Plotting script SHA256: {sha256_file(script_path)}")
    append(f"Execution started: {started_at.astimezone().isoformat()}")
    append(f"Execution finished: {finished_at.astimezone().isoformat()}")
    append(f"Execution time (seconds): {elapsed_seconds:.6f}")
    append(f"Python: {platform.python_version()} ({sys.executable})")
    append(f"Platform: {platform.platform()}")
    append(f"numpy: {np.__version__}")
    append(f"pandas: {pd.__version__}")
    append(f"scikit-learn: {sklearn.__version__}")
    append(f"matplotlib: {mpl.__version__}")
    append(f"PyYAML: {yaml.__version__}")
    append("")
    append("GLOBAL METHOD CONTRACT")
    append("----------------------")
    append(f"Model: {MODEL_NAME}")
    append(f"Member seeds: {list(SEEDS)} (exactly five independently trained members)")
    append("Raw probability source: stored per-seed p_joint from frozen test_predictions.csv")
    append("Alignment key: material_id (sorted before probability aggregation)")
    append(
        "Ensemble aggregation: arithmetic mean of five stored raw p_joint values in "
        "probability space, followed by sklearn.metrics.average_precision_score"
    )
    append("Temperature fitting performed by this script: False")
    append("Calibrated probability used: False")
    append("Legacy data/processed/iid_split.csv used: False")
    append("Model retraining performed: False")
    append("Five-seed single-model mean used as ensemble AP: False")
    append("Reference split: Random strict (R23), not legacy iid_split.csv")
    append("Delta definition: AP_split - AP_random")
    append("Main figure: horizontal reference/dumbbell plot")
    append("Main figure displays: four deterministic raw five-member ensemble AP endpoints")
    append("Random reference line: vertical dashed line at the recomputed Random AP")
    append("Non-random segments: horizontal lines from Random AP to each split AP")
    append("Single-member points displayed: False")
    append("Categorical trend line displayed: False")
    append("Main figure error bars: none")
    append(
        "Manuscript approximate values are sanity references only; they are not plot inputs."
    )
    append(f"Sanity-stop tolerance: absolute AP difference > {SANITY_TOLERANCE:.3f}")
    append("")

    structure = split_results[0]["structure_definition"]
    append("STRUCTURE-FAMILY DEFINITION")
    append("---------------------------")
    append(f"Definition: {structure['definition']}")
    append(f"Construction source: {structure['prepare_data_path']}")
    append(f"Construction source SHA256: {structure['prepare_data_sha256']}")
    append(f"Strict-split source: {structure['make_review_splits_path']}")
    append(f"Strict-split source SHA256: {structure['make_review_splits_sha256']}")
    append(f"Anonymous-formula source: {structure['anonymous_formula_path']}")
    append(f"Anonymous-formula source SHA256: {structure['anonymous_formula_sha256']}")
    append(f"Anonymous-formula method: {structure['anonymous_formula_method']}")
    append("")

    positive_rates: list[float] = []
    for result in split_results:
        spec = result["spec"]
        config = result["config_audit"]
        manifest = result["manifest_audit"]
        positive_rates.append(float(manifest["joint_positive_rate"]))
        append(f"[{spec['display']}]")
        append(f"Experiment: {spec['experiment']}")
        append(f"Config: {config['config_path']}")
        append(f"Config SHA256: {config['config_sha256']}")
        append(f"Resolved config: {config['resolved_path']}")
        append(f"Resolved config SHA256: {config['resolved_sha256']}")
        append(f"Split file: {manifest['manifest_path']}")
        append(f"Split relative path declared by config: {spec['split_relative']}")
        append(f"Split SHA256: {manifest['manifest_sha256']}")
        append(f"Frozen hash registry: {manifest['manifest_registry_path']}")
        append(f"Frozen split audit: {manifest['split_audit_path']}")
        append(f"Frozen split audit SHA256: {manifest['split_audit_sha256']}")
        append(f"Frozen manifest set: {manifest['frozen_manifest_set']}")
        append(f"Split algorithm: {manifest['split_algorithm_version']}")
        append(f"Split assignment seed: {manifest['assignment_seed']}")
        append(f"Split assignment SHA256: {manifest['assignment_sha256']}")
        append(f"Target group column: {manifest['target_group_column']}")
        append(
            "Target-group disjoint enforced: "
            f"{manifest['target_group_disjoint_enforced']}"
        )
        append(
            "Target-group cross-split overlap count: "
            f"{manifest['target_group_overlap_unique_count']}"
        )
        append(f"n_test: {manifest['n_test']}")
        append(f"n_joint_positive: {manifest['n_joint_positive']}")
        append(f"joint_positive_rate: {manifest['joint_positive_rate']:.15f}")
        append(f"Sorted material-ID SHA256: {manifest['sorted_material_id_sha256']}")
        append(
            "Sorted material-ID + Joint-label SHA256: "
            f"{manifest['sorted_material_id_label_sha256']}"
        )
        append("Frozen raw member predictions and Joint AP:")
        for seed in SEEDS:
            member = result["member_audits"][seed]
            append(f"  seed {seed}: {result['member_aps'][seed]:.15f}")
            append(f"    prediction: {member['prediction_path']}")
            append(f"    prediction SHA256: {member['prediction_sha256']}")
            append(f"    prediction bytes: {member['prediction_bytes']}")
            append(f"    metrics: {member['metrics_path']}")
            append(f"    metrics SHA256: {member['metrics_sha256']}")
            append(f"    completion record: {member['completed_path']}")
            append(f"    completion SHA256: {member['completed_sha256']}")
            append(f"    seed resolved config: {member['seed_resolved_path']}")
            append(f"    seed resolved config SHA256: {member['seed_resolved_sha256']}")
            append("    seed/root resolved config byte-identical: True")
            append(
                "    max |stored p_joint - sigmoid(logit_joint)|: "
                f"{member['max_abs_p_joint_minus_sigmoid_logit_joint']:.12g}"
            )
            append("    calibrated-like columns present: False")
        append(f"Single-member mean: {result['member_mean']:.15f}")
        append(
            "Single-member sample SD (numpy.std, ddof=1): "
            f"{result['member_sd']:.15f}"
        )
        append(f"Raw five-member ensemble Joint AP: {result['ensemble_ap']:.15f}")
        if spec["split"] == "random_stratified":
            append("Delta AP vs Random: reference (0.000000000000000)")
        else:
            append(f"Delta AP vs Random: {result['delta_vs_random']:+.15f}")
        append(f"Aggregation method: {result['aggregation_method']}")
        append(f"Sanity reference AP: {float(spec['sanity_reference']):.15f}")
        append(f"Recomputed minus sanity reference: {result['sanity_difference']:+.15f}")
        append("Sanity check status: PASS")
        summary = result["summary_crosscheck"]
        append(f"Per-experiment seed summary: {summary['local_summary_path']}")
        append(f"Global strict seed summary: {summary['global_summary_path']}")
        append(f"Formal raw ensemble summary: {summary['paired_summary_path']}")
        append(
            "Formal raw five-member ensemble AP: "
            f"{summary['formal_raw_ensemble_ap']:.15f}"
        )
        append("Recomputed/formal raw ensemble AP cross-check: exact within 1e-12")
        append(f"Strict summary completion record: {summary['strict_status_path']}")
        append(f"Frozen strict summary JSON: {summary['strict_summary_json_path']}")
        append(
            "Frozen summary input-manifest member hashes matched: "
            f"{summary['frozen_member_manifest_matches']}/{len(SEEDS)}"
        )
        append(f"Raw ensemble summarizer source: {summary['summarizer_path']}")
        append(f"Formal pairing audit: {summary['pairing_audit_path']}")
        append(
            "Formal/recomputed sorted material-ID SHA256: "
            f"{summary['formal_test_material_ids_sha256']}"
        )
        append("Formal cross-variant/seed pairing audit: PASS")
        append("Per-seed metrics/COMPLETED/prediction recomputation conflict: none")
        append("Existing formal raw strict-split five-member ensemble summary: found and matched")
        if result["duplicate_records"]:
            append("Duplicate review prediction copies:")
            for duplicate in result["duplicate_records"]:
                append(f"  {duplicate['duplicate']}")
                append(f"    SHA256: {duplicate['sha256']}")
                append("    byte-identical to primary source: True")
        else:
            append("Duplicate review prediction copies found: none")
        append("Missing values in member IDs/labels/probabilities: none")
        append("Duplicate material IDs: none")
        append("Member ID/label alignment status: PASS")
        append("")

    rates = np.asarray(positive_rates, dtype=float)
    append("POSITIVE-RATE COMPARABILITY")
    append("---------------------------")
    append(f"Minimum Joint-positive rate: {rates.min():.15f}")
    append(f"Maximum Joint-positive rate: {rates.max():.15f}")
    append(f"Max-minus-min rate: {np.ptp(rates):.15f}")
    append(
        "All four rates are reported because AP depends on prevalence; no test set was "
        "resampled or altered."
    )
    legacy = split_results[0]["manifest_audit"]["legacy_exclusion"]
    append("")
    append("LEGACY IID EXCLUSION")
    append("--------------------")
    append(f"Explicitly excluded source: {legacy['path']}")
    append(f"Excluded source SHA256: {legacy['sha256']}")
    append(f"Legacy test n: {legacy['n_test']}")
    append(f"Strict Random test n: {legacy['strict_random_n_test']}")
    append(f"Legacy/strict test ID sets identical: {legacy['test_id_sets_identical']}")
    append(f"Legacy-only test IDs: {legacy['legacy_only_id_count']}")
    append(f"Strict-only test IDs: {legacy['strict_only_id_count']}")
    append("Legacy file used for model metrics or plotting: False")
    append("")
    append("OUTPUTS")
    append("-------")
    append(f"Source data CSV: {source_data_path.resolve()}")
    append(f"Source data CSV SHA256: {output_checks['source_data_sha256']}")
    for name in ("pdf", "png", "svg"):
        path = figure_paths[name]
        append(f"{name.upper()}: {path.resolve()}")
        append(f"{name.upper()} SHA256: {output_checks['figure_sha256'][name]}")
    append(
        f"PNG raster: {output_checks['png_width_px']} x "
        f"{output_checks['png_height_px']} px"
    )
    append(f"PNG embedded dpi: {output_checks['png_dpi']}")
    append("PDF/SVG saved directly by matplotlib as vector output")
    append("SVG raster <image> elements: none")
    append("")
    append("FINAL CONFLICT CHECK")
    append("--------------------")
    append("Missing required split/member source: none")
    append("Unexpected seed: none")
    append("Raw/calibrated mixing: none")
    append("Legacy fixed IID benchmark mixing: none")
    append("Split-manifest/hash conflict: none")
    append("Result-value conflict: none")
    append("Source files changed during execution: none")
    append("Overall status: PASS")
    return "\n".join(lines) + "\n"


def main() -> None:
    started_clock = time.perf_counter()
    started_at = datetime.now().astimezone()
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "paper_figures" / "fig4"
    )
    if not (project_root / "results" / "review").is_dir():
        raise AssertionError(f"Invalid project root: {project_root}")

    source_data, split_results = load_all_data(project_root)
    print_dumbbell_summary(split_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_data_path = output_dir / "Fig4_source_data.csv"
    atomic_to_csv(source_data, source_data_path)
    figure_paths = create_figure(split_results, output_dir)
    output_checks = validate_written_outputs(source_data_path, figure_paths)
    verify_sources_unchanged(split_results)

    finished_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - started_clock
    audit_text = build_audit_text(
        project_root=project_root,
        script_path=Path(__file__),
        source_data_path=source_data_path,
        figure_paths=figure_paths,
        output_checks=output_checks,
        split_results=split_results,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
    )
    audit_path = output_dir / "Fig4_data_audit.txt"
    atomic_write_text(audit_text, audit_path)

    print("[OUTPUTS]")
    for path in (
        figure_paths["pdf"],
        figure_paths["png"],
        figure_paths["svg"],
        source_data_path,
        audit_path,
    ):
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
