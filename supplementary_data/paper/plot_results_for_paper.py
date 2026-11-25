from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def save_figure(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def find_seed_files(model_dir: Path, filename: str = "test_predictions.csv") -> dict[int, Path]:
    result = {}
    for path in sorted(model_dir.glob(f"seed_*/{filename}")):
        try:
            seed = int(path.parent.name.split("_")[-1])
        except ValueError:
            continue
        result[seed] = path
    return result


def model_seed_ap(model_dir: Path) -> dict[int, float]:
    out = {}
    for seed, path in find_seed_files(model_dir).items():
        df = pd.read_csv(path)
        out[seed] = average_precision_score(df["y_joint"], df["p_joint"])
    return out


def mean_pr_across_seeds(model_dir: Path, seeds: list[int], recall_grid: np.ndarray):
    curves = []
    aps = []
    for seed in seeds:
        path = model_dir / f"seed_{seed}" / "test_predictions.csv"
        df = pd.read_csv(path)
        p, r, _ = precision_recall_curve(df["y_joint"], df["p_joint"])
        # sklearn returns recall in descending order. Reverse for np.interp.
        r = r[::-1]
        p = p[::-1]
        r_unique, idx = np.unique(r, return_index=True)
        p_unique = p[idx]
        curves.append(np.interp(recall_grid, r_unique, p_unique))
        aps.append(average_precision_score(df["y_joint"], df["p_joint"]))
    curves = np.asarray(curves)
    return curves.mean(axis=0), curves.std(axis=0, ddof=1), np.asarray(aps)


def figure_2_labels(root: Path, out: Path) -> None:
    supervised = pd.read_csv(root / "1. Labels and Quadrants" / "supervised.csv")
    q = pd.crosstab(supervised["thermo_label"], supervised["dynamic_label"])
    q = q.reindex(index=[1, 0], columns=[0, 1], fill_value=0)
    total = q.to_numpy().sum()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    ax = axes[0]
    im = ax.imshow(q.to_numpy(), aspect="auto")
    ax.set_xticks([0, 1], ["Dynamic−", "Dynamic+"])
    ax.set_yticks([0, 1], ["Thermo+", "Thermo−"])
    ax.set_title("A  Thermo–Dynamic label quadrants", loc="left")
    for i in range(2):
        for j in range(2):
            n = int(q.iloc[i, j])
            ax.text(j, i, f"{n:,}\n({100*n/total:.2f}%)", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    ax = axes[1]
    d_yes = supervised.loc[supervised["dynamic_label"] == 1, "ehull"].to_numpy(float)
    d_no = supervised.loc[supervised["dynamic_label"] == 0, "ehull"].to_numpy(float)
    bins = np.linspace(0, 1.0, 51)
    ax.hist(d_yes, bins=bins, density=True, histtype="step", linewidth=1.8, label="Dynamic+")
    ax.hist(d_no, bins=bins, density=True, histtype="step", linewidth=1.8, label="Dynamic−")
    ax.axvline(0.10, linestyle="--", linewidth=1.4, label=r"$E_{hull}=0.10$ eV/atom")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"$E_{hull}$ (eV/atom)")
    ax.set_ylabel("Density")
    ax.set_title("B  Energy-above-hull distributions", loc="left")
    ax.legend(frameon=False)

    save_figure(fig, out, "Fig2_label_structure")


def figure_3_main_models(root: Path, out: Path) -> None:
    base = root / "2. Single Equal Full Main Models"
    models = {
        "Single ALIGNN": base / "M04_single_alignn",
        "Equal MTL": base / "M05_equal_mtl",
        "Full MTL": base / "M06_full_mtl",
    }
    seed_sets = [set(find_seed_files(p)) for p in models.values()]
    common = sorted(set.intersection(*seed_sets))
    if not common:
        raise RuntimeError("No common seeds found for Single/Equal/Full.")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # Panel A: paired AP across common seeds.
    ax = axes[0]
    x = np.arange(len(models))
    per_model = {name: model_seed_ap(path) for name, path in models.items()}
    for seed in common:
        ys = [per_model[name][seed] for name in models]
        ax.plot(x, ys, marker="o", linewidth=1.0, alpha=0.65)
    means = [np.mean([per_model[name][s] for s in common]) for name in models]
    stds = [np.std([per_model[name][s] for s in common], ddof=1) for name in models]
    ax.errorbar(x, means, yerr=stds, fmt="D", capsize=4, linewidth=1.8, label="Mean ± SD")
    ax.set_xticks(x, list(models.keys()), rotation=15, ha="right")
    ax.set_ylabel("Joint average precision")
    ax.set_title(f"A  Common-seed comparison (n={len(common)} seeds)", loc="left")
    ax.legend(frameon=False)

    # Panel B: mean PR curves across exactly the same common seeds.
    ax = axes[1]
    grid = np.linspace(0, 1, 301)
    for name, path in models.items():
        mean_p, std_p, aps = mean_pr_across_seeds(path, common, grid)
        line = ax.plot(grid, mean_p, label=f"{name} (AP={aps.mean():.3f})")[0]
        ax.fill_between(grid, np.clip(mean_p - std_p, 0, 1), np.clip(mean_p + std_p, 0, 1),
                        alpha=0.12, color=line.get_color())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("B  Mean precision–recall curves", loc="left")
    ax.legend(frameon=False, fontsize=9)

    save_figure(fig, out, "Fig3_main_model_comparison")


def figure_4_ablation_gradient(root: Path, out: Path) -> None:
    ab_base = root / "3. Ablation Experiments"
    full_dir = root / "2. Single Equal Full Main Models" / "M06_full_mtl"
    variants = {
        "Joint+Thermo": ab_base / "A01_joint_thermo",
        "Joint+Dynamic": ab_base / "A02_joint_dynamic",
        "No logic": ab_base / "A03_no_logic",
        "Fixed weight": ab_base / "A04_fixed_weight",
        "Shuffled auxiliary": ab_base / "A05_shuffle_auxiliary",
        "Adaptive all": ab_base / "A06_adaptive_all",
    }
    full_ap = model_seed_ap(full_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))

    ax = axes[0]
    labels = []
    means = []
    stds = []
    all_deltas = []
    for label, path in variants.items():
        ap = model_seed_ap(path)
        seeds = sorted(set(ap) & set(full_ap))
        delta = np.asarray([ap[s] - full_ap[s] for s in seeds], dtype=float)
        labels.append(label)
        means.append(delta.mean())
        stds.append(delta.std(ddof=1) if len(delta) > 1 else 0.0)
        all_deltas.append(delta)
    y = np.arange(len(labels))
    ax.axvline(0, linestyle="--", linewidth=1.0)
    for yi, vals in zip(y, all_deltas):
        ax.scatter(vals, np.full_like(vals, yi, dtype=float), s=28, alpha=0.7)
    ax.errorbar(means, y, xerr=stds, fmt="D", capsize=4, linewidth=1.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ Joint AP (variant − Full, paired seed)")
    ax.set_title("A  Ablation relative to Full MTL", loc="left")

    grad_base = root / "4. Gradient Conflicts"
    equal = pd.read_csv(grad_base / "M05_equal_mtl" / "task_gradient_analysis" / "gradient_cosines_by_batch.csv")
    full = pd.read_csv(grad_base / "M06_full_mtl" / "task_gradient_analysis" / "gradient_cosines_by_batch.csv")
    cols = ["cosine_joint_thermo", "cosine_joint_dynamic", "cosine_thermo_dynamic"]
    pair_labels = ["Joint–Thermo", "Joint–Dynamic", "Thermo–Dynamic"]

    ax = axes[1]
    positions_equal = np.arange(1, 4) - 0.14
    positions_full = np.arange(1, 4) + 0.14
    v1 = ax.violinplot([equal[c].dropna().to_numpy() for c in cols], positions=positions_equal,
                       widths=0.24, showmeans=True, showextrema=False)
    v2 = ax.violinplot([full[c].dropna().to_numpy() for c in cols], positions=positions_full,
                       widths=0.24, showmeans=True, showextrema=False)
    # Use the first body to get legend handles without prescribing colors.
    if v1["bodies"] and v2["bodies"]:
        ax.legend([v1["bodies"][0], v2["bodies"][0]], ["Equal MTL", "Full MTL"], frameon=False)
    ax.axhline(0, linestyle="--", linewidth=1.0)
    ax.set_xticks([1, 2, 3], pair_labels, rotation=15, ha="right")
    ax.set_ylabel("Gradient cosine similarity")
    ax.set_title("B  Task-gradient interactions", loc="left")

    save_figure(fig, out, "Fig4_ablation_and_gradients")


def figure_5_ensemble_calibration(root: Path, out: Path) -> None:
    ens_dir = root / "2. Single Equal Full Main Models" / "M06_full_mtl" / "ensemble"
    size = pd.read_csv(ens_dir / "ensemble_size_ablation.csv")
    size = size[size["task"] == "joint"].sort_values("members")
    pred = pd.read_csv(ens_dir / "test_ensemble_predictions.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    ax = axes[0]
    ax.errorbar(size["members"], size["pr_auc_mean"], yerr=size["pr_auc_std"], marker="o", capsize=4)
    ax.set_xticks(size["members"])
    ax.set_xlabel("Ensemble members")
    ax.set_ylabel("Joint average precision")
    ax.set_title("A  Ranking performance", loc="left")

    ax = axes[1]
    ax.errorbar(size["members"], size["nll_mean"], yerr=size["nll_std"], marker="o", capsize=4, label="NLL")
    ax.set_xticks(size["members"])
    ax.set_xlabel("Ensemble members")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("B  Probability quality", loc="left")

    ax = axes[2]
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    for col, label in [("p_joint_mean", "Uncalibrated"), ("p_joint_calibrated", "Temperature-scaled")]:
        frac, mean_p = calibration_curve(pred["y_joint"], pred[col], n_bins=15, strategy="uniform")
        ax.plot(mean_p, frac, marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("C  Reliability diagram", loc="left")
    ax.legend(frameon=False, fontsize=9)

    save_figure(fig, out, "Fig5_ensemble_and_calibration")


def figure_6_uncertainty(root: Path, out: Path) -> None:
    ens_dir = root / "2. Single Equal Full Main Models" / "M06_full_mtl" / "ensemble"
    pred = pd.read_csv(ens_dir / "test_ensemble_predictions.csv")
    thresholds = json.loads((ens_dir / "thresholds.json").read_text(encoding="utf-8"))
    threshold = float(thresholds["joint"])
    pred_label = (pred["p_joint_calibrated"].to_numpy() >= threshold).astype(int)
    y = pred["y_joint"].to_numpy(int)
    is_error = pred_label != y
    u = pred["p_joint_std"].to_numpy(float)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    groups = [u[~is_error], u[is_error]]
    ax.violinplot(groups, positions=[1, 2], showmeans=False, showmedians=True, showextrema=False)
    ax.boxplot(groups, positions=[1, 2], widths=0.16, showfliers=False)
    ax.set_xticks([1, 2], ["Correct", "Error"])
    ax.set_ylabel("Ensemble predictive SD")
    ax.set_title("A  Uncertainty by prediction outcome", loc="left")

    ax = axes[1]
    fpr, tpr, _ = roc_curve(is_error.astype(int), u)
    error_auc = roc_auc_score(is_error.astype(int), u)
    ax.plot(fpr, tpr, label=f"Predictive SD (AUROC={error_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("B  Error detection", loc="left")
    ax.legend(frameon=False, fontsize=9)

    rc = pd.read_csv(root / "6. Uncertainty" / "uncertainty_evaluation" / "risk_coverage_deep_ensemble_predictive_std.csv")
    ax = axes[2]
    ax.plot(rc["coverage"], rc["risk"])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("C  Selective prediction", loc="left")

    save_figure(fig, out, "Fig6_uncertainty_quality")


def figure_7_ood(root: Path, out: Path) -> None:
    ood_dir = root / "7. OOD" / "OOD_comparison"
    summary = pd.read_csv(ood_dir / "ood_ensemble_summary_common_members.csv")
    quality = pd.read_csv(ood_dir / "ood_uncertainty_quality.csv")
    labels = {
        "IID": "IID",
        "O01_chemical_system_ood": "Chemical-system OOD",
        "O02_structure_family_ood": "Structure-family OOD",
    }
    summary["label"] = summary["evaluation"].map(labels)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    ax = axes[0]
    x = np.arange(len(summary))
    width = 0.36
    ax.bar(x - width/2, summary["pr_auc"], width=width, label="AP")
    ax.bar(x + width/2, summary["mcc"], width=width, label="MCC")
    ax.set_xticks(x, summary["label"], rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("A  Predictive performance", loc="left")
    ax.legend(frameon=False)

    files = [
        ("IID", ood_dir / "IID_ensemble_test_predictions.csv"),
        ("Chemical-system OOD", ood_dir / "O01_chemical_system_ood_ensemble_test_predictions.csv"),
        ("Structure-family OOD", ood_dir / "O02_structure_family_ood_ensemble_test_predictions.csv"),
    ]
    dist = [pd.read_csv(p)["predictive_std"].to_numpy(float) for _, p in files]
    ax = axes[1]
    ax.violinplot(dist, positions=[1, 2, 3], showmedians=True, showextrema=False)
    ax.set_xticks([1, 2, 3], [x[0] for x in files], rotation=18, ha="right")
    ax.set_ylabel("Ensemble predictive SD")
    ax.set_title("B  Uncertainty under shift", loc="left")

    primary = quality[quality["uncertainty_score"] == "predictive_std"].copy()
    primary["label"] = primary["evaluation"].map(labels)
    primary = primary.set_index("evaluation").loc[summary["evaluation"]].reset_index()
    ax = axes[2]
    ax.bar(np.arange(len(primary)), primary["error_auroc"])
    ax.set_xticks(np.arange(len(primary)), primary["label"], rotation=18, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Error-detection AUROC")
    ax.set_title("C  Error localization", loc="left")

    save_figure(fig, out, "Fig7_ood_generalization")


def figure_8_unknown(root: Path, out: Path) -> None:
    path = root / "8. Unknown 8803" / "M10_unknown_screening" / "unknown_all_predictions.csv"
    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    eligible = df["thermo_eligible"].astype(bool).to_numpy()
    ax.scatter(df.loc[~eligible, "ehull"], df.loc[~eligible, "p_joint_calibrated"], s=7, alpha=0.20,
               label="Thermo-ineligible", rasterized=True)
    ax.scatter(df.loc[eligible, "ehull"], df.loc[eligible, "p_joint_calibrated"], s=11, alpha=0.55,
               label="Thermo-eligible", rasterized=True)
    ax.axvline(0.10, linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$E_{hull}$ (eV/atom)")
    ax.set_ylabel("Calibrated Joint probability")
    ax.set_title("A  Unknown candidate space", loc="left")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    sc = ax.scatter(df["p_joint_calibrated"], df["p_joint_std"], s=7, alpha=0.25,
                    c=df["ehull"], rasterized=True)
    ax.set_xlabel("Calibrated Joint probability")
    ax.set_ylabel("Ensemble predictive SD")
    ax.set_title("B  Probability–uncertainty landscape", loc="left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$E_{hull}$ (eV/atom)")

    save_figure(fig, out, "Fig8_unknown_screening")


def main() -> None:
    # Use the script directory as the data root by default.
    # The historical bundle expected this script in D:/ck/two/paper.
    # Passing --root manually is therefore optional.
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate publication-ready Results & Discussion figures from the curated result bundle."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir,
        help="Data root. Default: the folder containing this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/figures_revised",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.output or (root / "figures_revised")).resolve()

    # Check required input paths first and report missing paths together.
    required_paths = [
        root / "1. Labels and Quadrants" / "supervised.csv",
        root / "2. Single Equal Full Main Models" / "M04_single_alignn",
        root / "2. Single Equal Full Main Models" / "M05_equal_mtl",
        root / "2. Single Equal Full Main Models" / "M06_full_mtl",
        root / "3. Ablation Experiments",
        root / "4. Gradient Conflicts",
        root / "6. Uncertainty" / "uncertainty_evaluation",
        root / "7. OOD" / "OOD_comparison",
        root / "8. Unknown 8803" / "M10_unknown_screening" / "unknown_all_predictions.csv",
    ]
    missing = [p for p in required_paths if not p.exists()]
    if missing:
        print("\n[ERROR] The data root or directory layout does not match. Current root:")
        print(f"  {root}")
        print("\nThe following paths are missing:")
        for p in missing:
            print(f"  - {p}")
        print("\nVerify that the required data directories are present under the selected paper bundle root.")
        raise SystemExit(2)

    print(f"Data root : {root}")
    print(f"Output dir: {out}")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure_2_labels(root, out)
    figure_3_main_models(root, out)
    figure_4_ablation_gradient(root, out)
    figure_5_ensemble_calibration(root, out)
    figure_6_uncertainty(root, out)
    figure_7_ood(root, out)
    figure_8_unknown(root, out)
    print(f"Saved figures to: {out}")


if __name__ == "__main__":
    main()