"""Generate publication-ready figures from frozen review-supplement CSV tables.

The plotting stage is intentionally read-only with respect to model predictions. It
validates every required input table before writing any figure, and it does not infer
or synthesize unavailable embedding-distance results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "review_supplements"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


REQUIRED_SCHEMAS: Mapping[str, set[str]] = {
    "bootstrap_summary.csv": {
        "status",
        "comparator",
        "scope",
        "seed",
        "ensemble_members",
        "metric",
        "difference_full_minus_comparator",
        "ci95_low",
        "ci95_high",
        "bootstrap_replicates_valid",
    },
    "logic_violations.csv": {
        "status",
        "model",
        "probability_scale",
        "violation",
        "n",
        "violation_count",
        "violation_rate",
    },
    "calibration_shift.csv": {
        "status",
        "evaluation",
        "probability_scale",
        "temperature_fit_domain",
        "threshold_fit_domain",
        "nll",
        "brier",
        "ece",
    },
    "threshold_sensitivity.csv": {
        "status",
        "record_type",
        "ehull_threshold_eV_per_atom",
        "unknown_full_pool_count",
        "unknown_candidate_count",
        "unknown_predicted_joint_positive_count",
        "p_Dplus_given_Tplus",
    },
    "pool_shift.csv": {
        "status",
        "record_type",
        "comparison",
        "feature",
        "standardized_mean_difference_target_minus_known",
        "total_variation_distance",
        "fold",
        "cv_auroc",
        "cv_auroc_std",
        "missing_feature_count",
    },
}


def require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    """Raise with a precise message when a source table is incompatible."""

    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")


def require_ok_rows(frame: pd.DataFrame, source: str) -> None:
    statuses = frame["status"].dropna().astype(str)
    bad = sorted(set(statuses) - {"ok"})
    if bad or len(statuses) != len(frame):
        raise ValueError(f"{source}: all rows must have status='ok'; found {bad or ['missing']}")


def require_finite(frame: pd.DataFrame, columns: Sequence[str], source: str) -> None:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(values.to_numpy(dtype=float))
    if bad.any():
        positions = np.argwhere(bad)
        examples = [
            f"row={frame.index[int(i)]}, column={columns[int(j)]}"
            for i, j in positions[:5]
        ]
        raise ValueError(f"{source}: non-finite required values at {examples}")


def require_unique(frame: pd.DataFrame, columns: Sequence[str], source: str) -> None:
    duplicated = frame.duplicated(list(columns), keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, list(columns)].head(5).to_dict("records")
        raise ValueError(f"{source}: duplicate keys for {list(columns)}: {example}")


def load_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate all source tables before any output directory is created."""

    tables: dict[str, pd.DataFrame] = {}
    for name, schema in REQUIRED_SCHEMAS.items():
        path = input_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required review-supplement table not found: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"{path}: table is empty")
        require_columns(frame, schema, str(path))
        require_ok_rows(frame, str(path))
        tables[name] = frame
    return tables


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_pair(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure was not written correctly: {path}")
    return paths


def _friendly_comparator(value: str) -> str:
    return {
        "M04_single_alignn": "Single-task ALIGNN",
        "M05_equal_mtl": "Equal-weight MTL",
        "A03_no_logic": "No logic regularization",
    }.get(value, value)


def plot_bootstrap_forest(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    source = "bootstrap_summary.csv"
    selected = frame.loc[
        frame["scope"].eq("common_seed_probability_mean_ensemble")
    ].copy()
    expected_comparators = {"M04_single_alignn", "M05_equal_mtl", "A03_no_logic"}
    expected_metrics = {"pr_auc", "roc_auc"}
    if set(selected["comparator"]) != expected_comparators:
        raise ValueError(
            f"{source}: ensemble forest plot requires comparators {sorted(expected_comparators)}"
        )
    if set(selected["metric"]) != expected_metrics:
        raise ValueError(f"{source}: ensemble forest plot requires PR-AUC and ROC-AUC")
    require_unique(selected, ["comparator", "metric"], source)
    numeric = [
        "difference_full_minus_comparator",
        "ci95_low",
        "ci95_high",
        "bootstrap_replicates_valid",
    ]
    require_finite(selected, numeric, source)
    if not (
        (selected["ci95_low"] <= selected["difference_full_minus_comparator"])
        & (selected["difference_full_minus_comparator"] <= selected["ci95_high"])
    ).all():
        raise ValueError(f"{source}: a point estimate lies outside its reported CI")
    if (selected["bootstrap_replicates_valid"] <= 0).any():
        raise ValueError(f"{source}: bootstrap replicate count must be positive")

    comparator_order = ["M04_single_alignn", "M05_equal_mtl", "A03_no_logic"]
    metric_order = ["pr_auc", "roc_auc"]
    records = []
    for comparator in comparator_order:
        for metric in metric_order:
            row = selected.loc[
                selected["comparator"].eq(comparator) & selected["metric"].eq(metric)
            ].iloc[0]
            records.append(row)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    y = np.arange(len(records))[::-1]
    for position, row in zip(y, records):
        estimate = float(row["difference_full_minus_comparator"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        color = BLUE if row["metric"] == "pr_auc" else ORANGE
        marker = "o" if row["metric"] == "pr_auc" else "s"
        ax.errorbar(
            estimate,
            position,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=6,
            capsize=3,
            elinewidth=1.5,
        )
        ax.annotate(
            f"{estimate:+.3f} [{low:+.3f}, {high:+.3f}]",
            (high, position),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    labels = [
        f"{_friendly_comparator(str(row['comparator']))}  |  "
        f"{'PR-AUC' if row['metric'] == 'pr_auc' else 'ROC-AUC'}"
        for row in records
    ]
    ax.set_yticks(y, labels)
    ax.axvline(0.0, color=GRAY, linewidth=1.0, linestyle="--")
    lows = selected["ci95_low"].to_numpy(float)
    highs = selected["ci95_high"].to_numpy(float)
    span = max(float(highs.max() - lows.min()), 0.02)
    ax.set_xlim(float(lows.min() - 0.12 * span), float(highs.max() + 0.62 * span))
    ax.set_xlabel("Paired metric difference (Full MTL minus comparator)")
    ax.set_title("Paired bootstrap comparison of common-seed ensembles")
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    return _save_pair(fig, output_dir, "fig01_paired_bootstrap_forest")


def plot_logic_violations(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    source = "logic_violations.csv"
    models = ["M06_full_mtl", "A03_no_logic"]
    scales = ["raw", "calibrated"]
    violations = [
        "V1_pJ_gt_pT",
        "V2_pJ_gt_pD",
        "V3_pJ_below_Frechet",
        "ANY_max_magnitude",
    ]
    selected = frame.loc[
        frame["model"].isin(models)
        & frame["probability_scale"].isin(scales)
        & frame["violation"].isin(violations)
    ].copy()
    require_unique(selected, ["model", "probability_scale", "violation"], source)
    expected_count = len(models) * len(scales) * len(violations)
    if len(selected) != expected_count:
        raise ValueError(f"{source}: expected {expected_count} model/scale/violation rows")
    require_finite(selected, ["n", "violation_count", "violation_rate"], source)
    if not selected["violation_rate"].between(0.0, 1.0).all():
        raise ValueError(f"{source}: violation rates must lie in [0, 1]")

    display_labels = ["pJ > pT", "pJ > pD", "pJ < pT+pD-1", "Any violation"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    x = np.arange(len(violations))
    width = 0.36
    for ax, scale in zip(axes, scales):
        for offset, model, color, label in [
            (-width / 2, models[0], BLUE, "Full MTL"),
            (width / 2, models[1], VERMILLION, "No logic regularization"),
        ]:
            subset = selected.loc[
                selected["model"].eq(model) & selected["probability_scale"].eq(scale)
            ].set_index("violation")
            values = np.array([float(subset.loc[v, "violation_rate"]) * 100 for v in violations])
            bars = ax.bar(x + offset, values, width, color=color, label=label, alpha=0.92)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5)
        ax.set_xticks(x, display_labels, rotation=18, ha="right")
        ax.set_title("Raw probabilities" if scale == "raw" else "IID-temperature calibrated")
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Violation rate (%)")
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle("Logical probability-constraint violations on the IID test set", y=1.02)
    fig.tight_layout()
    return _save_pair(fig, output_dir, "fig02_logic_violation_rates")


def plot_calibration_shift(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    source = "calibration_shift.csv"
    evaluations = ["IID", "chemical_OOD", "structure_OOD"]
    scales = ["raw", "calibrated_with_scenario_validation_temperature"]
    selected = frame.loc[
        frame["evaluation"].isin(evaluations)
        & frame["probability_scale"].isin(scales)
    ].copy()
    require_unique(selected, ["evaluation", "probability_scale"], source)
    if len(selected) != len(evaluations) * len(scales):
        raise ValueError(
            f"{source}: expected raw and scenario-calibrated rows for all domains"
        )
    require_finite(selected, ["nll", "brier", "ece"], source)
    calibrated = selected["probability_scale"].eq(scales[1])
    if not selected.loc[calibrated, "temperature_fit_domain"].eq(
        "scenario_development_validation_only"
    ).all():
        raise ValueError(
            f"{source}: calibrated rows were not frozen from matching scenario validation"
        )
    if not selected["threshold_fit_domain"].eq(
        "scenario_development_validation_only"
    ).all():
        raise ValueError(
            f"{source}: decision thresholds were not frozen from matching scenario validation"
        )

    display_domains = ["IID", "Chemical OOD", "Structure OOD"]
    metric_specs = [("nll", "NLL"), ("brier", "Brier score"), ("ece", "ECE")]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    x = np.arange(len(evaluations))
    width = 0.36
    for ax, (column, title) in zip(axes, metric_specs):
        for offset, scale, color, label in [
            (-width / 2, scales[0], GRAY, "Raw"),
            (width / 2, scales[1], GREEN, "Scenario-val calibrated"),
        ]:
            subset = selected.loc[selected["probability_scale"].eq(scale)].set_index(
                "evaluation"
            )
            values = np.array([float(subset.loc[item, column]) for item in evaluations])
            bars = ax.bar(x + offset, values, width, color=color, label=label, alpha=0.92)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.2, rotation=90)
        ax.set_xticks(x, display_domains, rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, max(float(selected[column].max()) * 1.25, 0.08))
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Scenario validation calibration frozen to the matching held-out test",
        y=1.03,
    )
    fig.tight_layout()
    return _save_pair(fig, output_dir, "fig03_scenario_validation_calibration")


def plot_threshold_sensitivity(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    source = "threshold_sensitivity.csv"
    selected = frame.loc[frame["record_type"].eq("ehull_threshold_statistics")].copy()
    selected = selected.sort_values("ehull_threshold_eV_per_atom")
    require_unique(selected, ["ehull_threshold_eV_per_atom"], source)
    if len(selected) < 3:
        raise ValueError(f"{source}: at least three E_hull thresholds are required")
    numeric = [
        "ehull_threshold_eV_per_atom",
        "unknown_full_pool_count",
        "unknown_candidate_count",
        "unknown_predicted_joint_positive_count",
        "p_Dplus_given_Tplus",
    ]
    require_finite(selected, numeric, source)
    if selected["unknown_full_pool_count"].nunique() != 1:
        raise ValueError(f"{source}: full Unknown pool count must be constant")
    if not selected["unknown_candidate_count"].is_monotonic_increasing:
        raise ValueError(f"{source}: candidate counts must increase with E_hull threshold")
    if not selected["p_Dplus_given_Tplus"].between(0.0, 1.0).all():
        raise ValueError(f"{source}: conditional probabilities must lie in [0, 1]")

    threshold = selected["ehull_threshold_eV_per_atom"].to_numpy(float)
    candidates = selected["unknown_candidate_count"].to_numpy(float)
    positives = selected["unknown_predicted_joint_positive_count"].to_numpy(float)
    conditional = selected["p_Dplus_given_Tplus"].to_numpy(float) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].plot(threshold, candidates, "o-", color=BLUE, linewidth=2, label="Unknown candidates")
    axes[0].plot(
        threshold,
        positives,
        "s-",
        color=ORANGE,
        linewidth=2,
        label="Predicted joint-positive",
    )
    for x_value, y_value in zip(threshold, candidates):
        axes[0].annotate(f"{int(y_value):,}", (x_value, y_value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5)
    for x_value, y_value in zip(threshold, positives):
        axes[0].annotate(f"{int(y_value):,}", (x_value, y_value), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7.5)
    axes[0].set_ylabel("Number of Unknown materials")
    axes[0].legend(frameon=False)
    axes[0].grid(color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)

    axes[1].plot(threshold, conditional, "o-", color=GREEN, linewidth=2)
    for x_value, y_value in zip(threshold, conditional):
        axes[1].annotate(f"{y_value:.1f}%", (x_value, y_value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    axes[1].set_ylabel("P(dynamic + | thermo +) in labeled data (%)")
    axes[1].set_ylim(max(0.0, float(conditional.min()) - 7), min(100.0, float(conditional.max()) + 7))
    axes[1].grid(color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    for ax in axes:
        ax.set_xlabel("E_hull threshold (eV/atom)")
        ax.set_xticks(threshold, [f"{value:.3f}" for value in threshold])
    full_count = int(selected["unknown_full_pool_count"].iloc[0])
    fig.suptitle(f"E_hull threshold sensitivity (direct pJ ranking retains all {full_count:,} Unknown samples)", y=1.02)
    fig.tight_layout()
    return _save_pair(fig, output_dir, "fig04_ehull_threshold_sensitivity")


def top_shift_rows(frame: pd.DataFrame, comparison: str, n: int = 10) -> pd.DataFrame:
    """Return the largest absolute standardized mean shifts with deterministic ties."""

    selected = frame.loc[
        frame["record_type"].eq("continuous_shift")
        & frame["comparison"].eq(comparison)
    ].copy()
    selected["absolute_effect"] = selected[
        "standardized_mean_difference_target_minus_known"
    ].abs()
    return selected.sort_values(
        ["absolute_effect", "feature"], ascending=[False, True], kind="mergesort"
    ).head(n)


def _friendly_feature(value: str) -> str:
    labels = {
        "ehull": "E_hull",
        "n_elements": "Number of elements",
        "natoms": "Number of atoms",
        "z_min": "Minimum atomic number",
        "z_mean": "Mean atomic number",
        "mass_mean": "Mean atomic mass",
        "radius_mean": "Mean atomic radius",
        "radius_std": "Atomic-radius SD",
        "layergroup_number": "Layer-group number",
        "spacegroup_number": "Space-group number",
        "cell_c": "Cell length c",
        "cell_volume": "Cell volume",
        "source_label": "Source label",
        "anonymous_formula": "Anonymous formula",
        "chemical_system": "Chemical system",
        "structure_family": "Structure family",
        "layergroup": "Layer group",
        "bravais_type": "Bravais type",
    }
    return labels.get(value, value.replace("_", " ").title())


def plot_pool_shift(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    source = "pool_shift.csv"
    comparison = "labeled_vs_unknown_all"
    continuous = frame.loc[
        frame["record_type"].eq("continuous_shift")
        & frame["comparison"].eq(comparison)
    ].copy()
    categorical = frame.loc[
        frame["record_type"].eq("categorical_summary")
        & frame["comparison"].eq(comparison)
    ].copy()
    diagnostic = frame.loc[
        frame["record_type"].eq("diagnostic_known_vs_unknown_classifier_summary")
        & frame["comparison"].eq(comparison)
    ].copy()
    if len(continuous) < 10 or categorical.empty or len(diagnostic) != 1:
        raise ValueError(
            f"{source}: incomplete full-pool continuous, categorical, or diagnostic shift rows"
        )
    require_unique(continuous, ["feature"], source)
    require_unique(categorical, ["feature"], source)
    require_finite(
        continuous,
        ["standardized_mean_difference_target_minus_known"],
        source,
    )
    require_finite(categorical, ["total_variation_distance"], source)
    require_finite(
        diagnostic,
        ["cv_auroc", "cv_auroc_std", "missing_feature_count"],
        source,
    )
    if float(diagnostic["missing_feature_count"].iloc[0]) != 0.0:
        raise ValueError(f"{source}: diagnostic classifier reports missing features")

    top = top_shift_rows(frame, comparison=comparison, n=10).sort_values(
        "standardized_mean_difference_target_minus_known"
    )
    categorical = categorical.sort_values("total_variation_distance")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.1))

    effects = top["standardized_mean_difference_target_minus_known"].to_numpy(float)
    colors = [VERMILLION if value < 0 else BLUE for value in effects]
    axes[0].barh(np.arange(len(top)), effects, color=colors, alpha=0.9)
    axes[0].set_yticks(np.arange(len(top)), [_friendly_feature(str(x)) for x in top["feature"]])
    axes[0].axvline(0.0, color=GRAY, linewidth=0.9)
    axes[0].set_xlabel("Standardized mean difference\n(Unknown minus labeled)")
    axes[0].set_title("Largest continuous-feature shifts")
    axes[0].grid(axis="x", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)

    tvd = categorical["total_variation_distance"].to_numpy(float)
    axes[1].barh(np.arange(len(categorical)), tvd, color=PURPLE, alpha=0.9)
    axes[1].set_yticks(
        np.arange(len(categorical)),
        [_friendly_feature(str(x)) for x in categorical["feature"]],
    )
    axes[1].set_xlabel("Total variation distance")
    axes[1].set_xlim(0, max(1.0, float(tvd.max()) * 1.12))
    axes[1].set_title("Categorical distribution shifts")
    axes[1].grid(axis="x", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    auc = float(diagnostic["cv_auroc"].iloc[0])
    auc_std = float(diagnostic["cv_auroc_std"].iloc[0])
    axes[1].text(
        0.98,
        0.04,
        f"Known-vs-Unknown diagnostic\n5-fold CV AUROC = {auc:.3f} +/- {auc_std:.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "boxstyle": "round,pad=0.35"},
    )
    fig.suptitle("Feature distribution shift: 8,803 Unknown vs 8,102 labeled materials", y=1.02)
    fig.tight_layout()
    return _save_pair(fig, output_dir, "fig05_unknown_pool_shift_top_effects")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_figures(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Validate all inputs, then generate the complete figure set."""

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    tables = load_inputs(input_dir)
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written.extend(plot_bootstrap_forest(tables["bootstrap_summary.csv"], output_dir))
    written.extend(plot_logic_violations(tables["logic_violations.csv"], output_dir))
    written.extend(plot_calibration_shift(tables["calibration_shift.csv"], output_dir))
    written.extend(plot_threshold_sensitivity(tables["threshold_sensitivity.csv"], output_dir))
    written.extend(plot_pool_shift(tables["pool_shift.csv"], output_dir))

    # Remove only the two superseded generated artifacts whose filename implies
    # the rejected practice of transferring one model's calibration to another.
    for suffix in (".png", ".pdf"):
        obsolete = output_dir / f"fig03_iid_to_ood_frozen_calibration{suffix}"
        if obsolete.parent != output_dir:
            raise RuntimeError(f"Refusing out-of-scope artifact cleanup: {obsolete}")
        obsolete.unlink(missing_ok=True)

    manifest = {
        "status": "ok",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "embedding_figure_generated": False,
        "embedding_figure_reason": "embedding distances are unavailable in the source analysis",
        "sources": {
            name: {"sha256": _sha256(input_dir / name), "rows": int(len(frame))}
            for name, frame in tables.items()
        },
        "figures": [
            {
                "path": str(path),
                "format": path.suffix.lstrip("."),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in written
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing review-supplement CSV tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure destination (default: <input-dir>/figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.input_dir / "figures")
    manifest = generate_figures(args.input_dir, output_dir)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": manifest["output_dir"],
                "figure_files": len(manifest["figures"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
