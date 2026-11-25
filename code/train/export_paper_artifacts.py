from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from train.common import load_yaml


MAIN_EXPERIMENTS = [
    "M01_random_forest",
    "M02_xgboost",
    "M03_cgcnn",
    "M04_single_alignn",
    "M05_equal_mtl",
    "M06_full_mtl",
]
ABLATION_EXPERIMENTS = [
    "A01_joint_thermo",
    "A02_joint_dynamic",
    "A03_no_logic",
    "A04_fixed_weight",
    "A05_shuffle_auxiliary",
    "A06_adaptive_all",
]


def load_seed_metrics(experiment_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(experiment_dir.glob("seed_*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "seed": data.get("seed"),
            "joint_pr_auc": data["test_joint"]["pr_auc"],
            "joint_roc_auc": data["test_joint"]["roc_auc"],
            "joint_mcc": data["test_joint"]["mcc"],
            "joint_f1": data["test_joint"]["f1"],
            "joint_balanced_accuracy": data["test_joint"]["balanced_accuracy"],
            "joint_nll": data["test_joint"]["nll"],
            "joint_brier": data["test_joint"]["brier"],
            "joint_ece": data["test_joint"]["ece"],
            "runtime_seconds": data.get("runtime_seconds", np.nan),
        }
        for task in ("thermo", "dynamic"):
            if f"test_{task}" in data:
                row[f"{task}_pr_auc"] = data[f"test_{task}"]["pr_auc"]
                row[f"{task}_mcc"] = data[f"test_{task}"]["mcc"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_experiments(results_root: Path, category: str, names):
    rows = []
    seed_rows = []
    for name in names:
        directory = results_root / category / name
        frame = load_seed_metrics(directory)
        if frame.empty:
            continue
        frame.insert(0, "experiment", name)
        seed_rows.append(frame)
        row = {"experiment": name, "n_seeds": len(frame)}
        for column in frame.select_dtypes(include=[np.number]).columns:
            if column == "seed":
                continue
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_std"] = float(frame[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows), (
        pd.concat(seed_rows, ignore_index=True)
        if seed_rows else pd.DataFrame()
    )


def save_figure(fig, base: Path):
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(results_root: Path, paper_dir: Path):
    sources = {
        "Single ALIGNN": results_root
        / "main"
        / "M04_single_alignn"
        / "seed_2025"
        / "test_predictions.csv",
        "Equal MTL": results_root
        / "main"
        / "M05_equal_mtl"
        / "seed_2025"
        / "test_predictions.csv",
        "Full Ensemble": results_root
        / "main"
        / "M06_full_mtl"
        / "ensemble"
        / "test_ensemble_predictions.csv",
    }
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    found = False
    for label, path in sources.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        probability_column = (
            "p_joint_calibrated"
            if "p_joint_calibrated" in frame
            else "p_joint"
        )
        precision, recall, _ = precision_recall_curve(
            frame["y_joint"], frame[probability_column]
        )
        axis.plot(recall, precision, label=label)
        found = True
    if found:
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Joint-stability precision-recall curves")
        axis.legend()
        save_figure(fig, paper_dir / "figure_joint_pr")
    else:
        plt.close(fig)


def plot_calibration(results_root: Path, paper_dir: Path):
    path = (
        results_root
        / "main"
        / "M06_full_mtl"
        / "ensemble"
        / "test_ensemble_predictions.csv"
    )
    if not path.exists():
        return
    frame = pd.read_csv(path)
    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    for column, label in [
        ("p_joint_mean", "Uncalibrated"),
        ("p_joint_calibrated", "Temperature-scaled"),
    ]:
        fraction, mean_probability = calibration_curve(
            frame["y_joint"],
            frame[column],
            n_bins=15,
            strategy="uniform",
        )
        axis.plot(mean_probability, fraction, marker="o", label=label)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Empirical positive rate")
    axis.set_title("Joint probability calibration")
    axis.legend()
    save_figure(fig, paper_dir / "figure_calibration")


def plot_risk_coverage(results_root: Path, paper_dir: Path):
    directory = (
        results_root
        / "main"
        / "M06_full_mtl"
        / "uncertainty_evaluation"
    )
    paths = sorted(directory.glob("risk_coverage_*.csv"))
    if not paths:
        return
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    for path in paths:
        frame = pd.read_csv(path)
        label = f"{frame['method'].iloc[0]} / {frame['uncertainty_score'].iloc[0]}"
        axis.plot(frame["coverage"], frame["risk"], label=label)
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective risk")
    axis.set_title("Risk-coverage curves")
    axis.legend(fontsize=8)
    save_figure(fig, paper_dir / "figure_risk_coverage")


def plot_ranking(results_root: Path, paper_dir: Path):
    path = (
        results_root
        / "main"
        / "M06_full_mtl"
        / "ranking"
        / "ranking_comparison.csv"
    )
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame = frame[frame["scope"] == "thermo_positive_test"].copy()
    columns = [column for column in frame if column.startswith("precision_at_")]
    if not columns:
        return
    long = frame.melt(
        id_vars=["method"],
        value_vars=columns,
        var_name="K",
        value_name="Precision",
    )
    pivot = long.pivot(index="K", columns="method", values="Precision")
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    pivot.plot(kind="bar", ax=axis)
    axis.set_ylabel("Precision@K")
    axis.set_title("Retrospective screening on Thermo-positive test materials")
    axis.legend(fontsize=7)
    save_figure(fig, paper_dir / "figure_ranking_precision")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    results_root = Path(cfg["paths"]["results_dir"])
    paper_dir = results_root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    main_summary, main_seeds = summarize_experiments(
        results_root, "main", MAIN_EXPERIMENTS
    )
    ablation_summary, ablation_seeds = summarize_experiments(
        results_root, "ablation", ABLATION_EXPERIMENTS
    )
    main_summary.to_csv(paper_dir / "table_main_comparison.csv", index=False)
    ablation_summary.to_csv(
        paper_dir / "table_ablation_comparison.csv", index=False
    )

    sheets = {
        "Main comparison": main_summary,
        "Main seeds": main_seeds,
        "Ablation comparison": ablation_summary,
        "Ablation seeds": ablation_seeds,
    }
    optional_tables = {
        "Data quadrants": Path(cfg["paths"]["processed_dir"])
        / "thermo_dynamic_quadrants.csv",
        "Uncertainty methods": results_root
        / "main"
        / "M06_full_mtl"
        / "uncertainty_evaluation"
        / "classification_and_calibration.csv",
        "Uncertainty quality": results_root
        / "main"
        / "M06_full_mtl"
        / "uncertainty_evaluation"
        / "uncertainty_quality.csv",
        "Joint fusion": results_root
        / "main"
        / "M06_full_mtl"
        / "joint_fusion"
        / "joint_fusion_comparison.csv",
        "Ranking": results_root
        / "main"
        / "M06_full_mtl"
        / "ranking"
        / "ranking_comparison.csv",
        "Lambda selection": results_root
        / "main"
        / "M06_full_mtl"
        / "ranking"
        / "lambda_selection.csv",
        "Ranking paired bootstrap": results_root
        / "main"
        / "M06_full_mtl"
        / "ranking"
        / "paired_risk_vs_probability_bootstrap.csv",
        "Ensemble size": results_root
        / "main"
        / "M06_full_mtl"
        / "ensemble"
        / "ensemble_size_ablation.csv",
        "MC sample count": results_root
        / "main"
        / "M06_full_mtl"
        / "uncertainty_evaluation"
        / "mc_sample_count_ablation.csv",
        "Task gradients equal MTL": results_root
        / "main"
        / "M05_equal_mtl"
        / "task_gradient_analysis"
        / "gradient_cosines_by_batch.csv",
        "Task gradients full MTL": results_root
        / "main"
        / "M06_full_mtl"
        / "task_gradient_analysis"
        / "gradient_cosines_by_batch.csv",
        "Probability Top100": results_root
        / "main"
        / "M10_unknown_screening"
        / "probability_top100.csv",
        "Risk Top100": results_root
        / "main"
        / "M10_unknown_screening"
        / "risk_top100.csv",
    }
    for name, path in optional_tables.items():
        if path.exists():
            sheets[name] = pd.read_csv(path)

    with pd.ExcelWriter(
        paper_dir / "paper_tables.xlsx",
        engine="openpyxl",
    ) as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)

    plot_pr_curves(results_root, paper_dir)
    plot_calibration(results_root, paper_dir)
    plot_risk_coverage(results_root, paper_dir)
    plot_ranking(results_root, paper_dir)
    print(f"Paper artifacts written to {paper_dir}")


if __name__ == "__main__":
    main()
