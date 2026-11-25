from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.db import connect
from ase.io import write
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.uncertainty import aggregate_binary_probabilities
from train.common import build_model, get_device, load_yaml, save_json
from train.dataset import CrystalGraphDataset, collate_graphs
from train.metrics import sigmoid


TASKS = ["thermo", "dynamic", "joint"]


def discover_member_dirs(full_dir: Path, experiment: dict) -> list[Path]:
    """Return only valid ensemble seed directories containing checkpoints.

    Avoids matching files such as ``seed_metrics_flat.csv`` when using the
    ``seed_*`` naming convention.
    """
    configured_seeds = experiment.get("seeds")
    if not configured_seeds:
        configured_seeds = (
            experiment.get("training", {}).get("ensemble_seeds")
        )

    if configured_seeds:
        member_dirs = [full_dir / f"seed_{int(seed)}" for seed in configured_seeds]
        missing = [
            str(path / "best_checkpoint.pt")
            for path in member_dirs
            if not (path / "best_checkpoint.pt").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Configured Deep Ensemble checkpoint(s) are missing:\n  - "
                + "\n  - ".join(missing)
            )
        return member_dirs

    # Fallback for older experiment directories without an explicit seed list.
    member_dirs = []
    for path in full_dir.glob("seed_*"):
        if not path.is_dir():
            continue
        if not re.fullmatch(r"seed_\d+", path.name):
            continue
        if not (path / "best_checkpoint.pt").is_file():
            continue
        member_dirs.append(path)

    member_dirs.sort(key=lambda p: int(p.name.split("_", 1)[1]))
    if len(member_dirs) < 2:
        raise RuntimeError(
            "Deep Ensemble members are missing. Expected directories like "
            "'seed_2025' containing 'best_checkpoint.pt'."
        )
    return member_dirs



def move_batch(batch, device):
    (graph, line_graph, lattice), labels, metadata = batch
    return (
        graph.to(device),
        line_graph.to(device),
        lattice.to(device),
    ), labels.to(device), metadata


@torch.no_grad()
def predict_member(model, loader, device):
    model.eval()
    base_rows = []
    probabilities = {task: [] for task in TASKS}
    logits_by_task = {task: [] for task in TASKS}
    for batch in tqdm(loader, desc="Unknown inference", leave=False):
        graph_batch, labels, metadata = move_batch(batch, device)
        outputs = model(graph_batch)
        for task in TASKS:
            task_logits = outputs[task].detach().cpu().numpy()
            logits_by_task[task].extend(task_logits.tolist())
            probabilities[task].extend(
                torch.sigmoid(outputs[task]).detach().cpu().numpy().tolist()
            )
        for index, item in enumerate(metadata):
            row = dict(item)
            row["thermo_label"] = int(labels[index, 0].detach().cpu())
            base_rows.append(row)
    return pd.DataFrame(base_rows), probabilities, logits_by_task


def robust_normalize(values, lower: float, upper: float):
    return np.clip(
        (np.asarray(values, dtype=float) - lower)
        / max(upper - lower, 1e-12),
        0.0,
        1.0,
    )


def confidence_tier(uncertainty_normalized: pd.Series) -> pd.Series:
    """Risk strata based on validation-derived robust uncertainty scaling.

    0--1 uncertainty is obtained using the validation q05/q95 range already
    frozen by evaluate_ranking.py. The thirds are descriptive risk strata,
    not calibrated coverage guarantees.
    """
    values = uncertainty_normalized.to_numpy(float)
    return pd.Series(
        np.select(
            [values <= 1.0 / 3.0, values <= 2.0 / 3.0],
            ["A_low_uncertainty", "B_medium_uncertainty"],
            default="C_high_uncertainty",
        ),
        index=uncertainty_normalized.index,
    )


def add_pj_aliases_and_logic_diagnostics(
    frame: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Expose paper-facing pJ aliases and calibrated logical-consistency diagnostics."""
    required = {
        "p_joint_mean",
        "p_joint_calibrated",
        "p_thermo_calibrated",
        "p_dynamic_calibrated",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing probability columns for pJ diagnostics: {missing}")

    result = frame.copy()
    for column in sorted(required):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains missing or non-finite values.")
        if np.any((values < -tolerance) | (values > 1.0 + tolerance)):
            raise ValueError(f"{column} contains values outside [0, 1].")

    # pJ is the uncalibrated five-member ensemble mean; pJ_calibrated is the
    # validation-temperature-scaled probability used for full-pool ranking.
    result["pJ"] = result["p_joint_mean"].astype(float)
    result["pJ_calibrated"] = result["p_joint_calibrated"].astype(float)

    thermo = result["p_thermo_calibrated"].to_numpy(float)
    dynamic = result["p_dynamic_calibrated"].to_numpy(float)
    joint = result["pJ_calibrated"].to_numpy(float)
    frechet_lower = np.maximum(0.0, thermo + dynamic - 1.0)
    upper_bound = np.minimum(thermo, dynamic)

    result["pJ_frechet_lower_bound_calibrated"] = frechet_lower
    result["pJ_upper_bound_calibrated"] = upper_bound
    result["pJ_gt_pT_violation_magnitude"] = np.maximum(joint - thermo, 0.0)
    result["pJ_gt_pD_violation_magnitude"] = np.maximum(joint - dynamic, 0.0)
    result["pJ_below_frechet_violation_magnitude"] = np.maximum(
        frechet_lower - joint,
        0.0,
    )
    result["pJ_gt_pT_violation"] = (
        result["pJ_gt_pT_violation_magnitude"] > tolerance
    )
    result["pJ_gt_pD_violation"] = (
        result["pJ_gt_pD_violation_magnitude"] > tolerance
    )
    result["pJ_below_frechet_violation"] = (
        result["pJ_below_frechet_violation_magnitude"] > tolerance
    )
    result["pJ_any_logic_violation"] = result[
        [
            "pJ_gt_pT_violation",
            "pJ_gt_pD_violation",
            "pJ_below_frechet_violation",
        ]
    ].any(axis=1)
    return result


def stable_rank(
    frame: pd.DataFrame,
    *,
    by: list[str],
    ascending: list[bool],
) -> pd.Series:
    """Return a deterministic, unique, continuous 1..N rank."""
    if len(by) != len(ascending):
        raise ValueError("Rank sort columns and directions must have equal length.")
    missing = [column for column in by if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing rank sort columns: {missing}")
    if not frame.index.is_unique:
        raise ValueError("Stable ranking requires a unique DataFrame index.")

    ordered_index = frame.sort_values(
        by,
        ascending=ascending,
        kind="mergesort",
        na_position="last",
    ).index
    ranks = pd.Series(
        np.arange(1, len(frame) + 1, dtype=np.int64),
        index=ordered_index,
    ).reindex(frame.index)
    expected = np.arange(1, len(frame) + 1, dtype=np.int64)
    if not np.array_equal(np.sort(ranks.to_numpy(dtype=np.int64)), expected):
        raise RuntimeError("Ranking is not a unique, continuous 1..N permutation.")
    return ranks.astype(np.int64)


def rank_all_unknown(
    frame: pd.DataFrame,
    *,
    uncertainty_column: str,
    uncertainty_lower: float,
    uncertainty_upper: float,
    selected_lambda: float,
    top_k: int,
) -> pd.DataFrame:
    """Rank every Unknown row by calibrated pJ; Thermo eligibility is diagnostic only."""
    if frame.empty:
        raise ValueError("The Unknown ranking pool is empty.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if "material_id" not in frame or "thermo_label" not in frame:
        raise KeyError("Unknown ranking requires material_id and thermo_label.")
    if frame["material_id"].isna().any():
        raise ValueError("material_id contains missing values.")

    result = frame.reset_index(drop=True).copy()
    result["material_id"] = result["material_id"].astype(str)
    if (result["material_id"].str.len() == 0).any():
        raise ValueError("material_id contains empty values.")
    if result["material_id"].duplicated().any():
        raise ValueError("material_id must be unique in the Unknown ranking pool.")
    if uncertainty_column not in result.columns:
        raise KeyError(
            f"Selected uncertainty column {uncertainty_column!r} is not present "
            "in Unknown predictions."
        )
    uncertainty = pd.to_numeric(
        result[uncertainty_column], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(uncertainty).all():
        raise ValueError(f"{uncertainty_column} contains missing or non-finite values.")

    result = add_pj_aliases_and_logic_diagnostics(result)
    result["uncertainty_normalized"] = robust_normalize(
        uncertainty,
        float(uncertainty_lower),
        float(uncertainty_upper),
    )
    result["confidence_tier"] = confidence_tier(result["uncertainty_normalized"])
    result["ranking_scope"] = "all_unknown"

    result["score_joint_probability"] = result["pJ_calibrated"]
    result["score_dynamic_probability"] = result["p_dynamic_calibrated"]
    result["score_product_fusion"] = (
        result["p_thermo_calibrated"] * result["p_dynamic_calibrated"]
    )
    result["score_minimum_fusion"] = np.minimum(
        result["p_thermo_calibrated"], result["p_dynamic_calibrated"]
    )
    result["score_risk_aware"] = (
        result["pJ_calibrated"]
        - float(selected_lambda) * result["uncertainty_normalized"]
    )
    result["thermo_eligible"] = result["thermo_label"].astype(int) == 1

    # material_id is the final tie-breaker, making ranks stable across input order.
    result["probability_rank"] = stable_rank(
        result,
        by=["score_joint_probability", "uncertainty_normalized", "material_id"],
        ascending=[False, True, True],
    )
    result["risk_rank"] = stable_rank(
        result,
        by=[
            "score_risk_aware",
            "pJ_calibrated",
            "uncertainty_normalized",
            "material_id",
        ],
        ascending=[False, False, True, True],
    )
    result["uncertainty_rank"] = stable_rank(
        result,
        by=["uncertainty_normalized", "pJ_calibrated", "material_id"],
        ascending=[True, False, True],
    )
    result["uncertainty_percentile"] = result["uncertainty_rank"] / len(result)

    result["screening_tier"] = "Outside_probability_TopK"
    top_mask = result["probability_rank"] <= min(top_k, len(result))
    result.loc[
        top_mask & (result["confidence_tier"] == "A_low_uncertainty"),
        "screening_tier",
    ] = "A_priority"
    result.loc[
        top_mask & (result["confidence_tier"] == "B_medium_uncertainty"),
        "screening_tier",
    ] = "B_review"
    result.loc[
        top_mask & (result["confidence_tier"] == "C_high_uncertainty"),
        "screening_tier",
    ] = "C_exploratory"
    return result


def select_top_k_by_rank(
    frame: pd.DataFrame,
    rank_column: str,
    top_k: int,
) -> pd.DataFrame:
    """Select Top-K using the already-frozen deterministic full-pool rank."""
    if rank_column not in frame:
        raise KeyError(rank_column)
    return (
        frame.sort_values(
            [rank_column, "material_id"],
            ascending=[True, True],
            kind="mergesort",
        )
        .head(min(int(top_k), len(frame)))
        .copy()
    )


def screening_population_summary(frame: pd.DataFrame) -> dict:
    eligible_count = int(frame["thermo_eligible"].astype(bool).sum())
    return {
        "unknown_count": int(len(frame)),
        "ranking_pool_count": int(len(frame)),
        "ranking_scope": "all_unknown",
        "thermo_eligible_count": eligible_count,
        "thermo_ineligible_count": int(len(frame) - eligible_count),
    }


def joint_logic_violation_summary(frame: pd.DataFrame) -> dict:
    count = int(len(frame))

    def magnitude_summary(column: str) -> dict:
        values = frame[column].to_numpy(float)
        return {
            "mean": float(values.mean()) if count else 0.0,
            "max": float(values.max()) if count else 0.0,
        }

    return {
        "evaluated_count": count,
        "pJ_gt_pT_count": int(frame["pJ_gt_pT_violation"].sum()),
        "pJ_gt_pD_count": int(frame["pJ_gt_pD_violation"].sum()),
        "pJ_below_frechet_count": int(
            frame["pJ_below_frechet_violation"].sum()
        ),
        "any_violation_count": int(frame["pJ_any_logic_violation"].sum()),
        "any_violation_fraction": (
            float(frame["pJ_any_logic_violation"].mean()) if count else 0.0
        ),
        "pJ_gt_pT_magnitude": magnitude_summary(
            "pJ_gt_pT_violation_magnitude"
        ),
        "pJ_gt_pD_magnitude": magnitude_summary(
            "pJ_gt_pD_violation_magnitude"
        ),
        "pJ_below_frechet_magnitude": magnitude_summary(
            "pJ_below_frechet_violation_magnitude"
        ),
    }


def topk_overlap_table(frame: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    rows = []
    for k in sorted(set(int(k) for k in k_values if int(k) > 0)):
        k_eff = min(k, len(frame))
        if k_eff == 0:
            continue
        prob_ids = set(
            frame.nsmallest(k_eff, "probability_rank")["material_id"].astype(str)
        )
        risk_ids = set(
            frame.nsmallest(k_eff, "risk_rank")["material_id"].astype(str)
        )
        overlap = len(prob_ids & risk_ids)
        union = len(prob_ids | risk_ids)
        rows.append(
            {
                "K": k,
                "effective_K": k_eff,
                "overlap_count": overlap,
                "overlap_fraction": overlap / k_eff,
                "jaccard": overlap / union if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def export_cifs(frame: pd.DataFrame, db, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    for rank, row in enumerate(frame.itertuples(index=False), start=1):
        atoms = db.get(id=int(row.row_id)).toatoms()
        filename = output_dir / f"{rank:03d}_{str(row.material_id).replace('/', '_')}.cif"
        write(str(filename), atoms, format="cif")
        expected_names.add(filename.name)

    # A forced rerun can replace most of the previous Top-100. Remove only
    # stale generated CIF files after every new CIF was written successfully.
    for existing in output_dir.glob("*.cif"):
        if existing.name not in expected_names:
            existing.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-experiment-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    full_dir = Path(args.full_experiment_dir)
    experiment = load_yaml(full_dir / "resolved_experiment.yaml")
    output = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(experiment["paths"]["results_dir"]) / "main" / "M10_unknown_screening"
    )
    completed = output / "COMPLETED.json"
    if completed.exists() and not args.force:
        print(f"SKIP completed: {completed}")
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.force and completed.exists():
        # A failed forced rerun must not leave the previous success marker in
        # place, otherwise the next normal invocation would skip stale output.
        completed.unlink()

    device = get_device(experiment["training"].get("device", "auto"))
    processed = Path(experiment["paths"]["processed_dir"])
    dataset = CrystalGraphDataset(processed / "unknown_pool.csv", require_labels=False)

    member_dirs = discover_member_dirs(full_dir, experiment)
    print(
        "Deep Ensemble members:",
        ", ".join(path.name for path in member_dirs),
    )

    first_checkpoint = torch.load(member_dirs[0] / "best_checkpoint.pt", map_location="cpu")
    batch_size = int(first_checkpoint.get("batch_size", experiment["training"]["batch_size"]))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graphs,
    )

    base = None
    member_values = {task: [] for task in TASKS}
    member_logits = {task: [] for task in TASKS}
    for member_dir in member_dirs:
        model = build_model(experiment).to(device)
        checkpoint = torch.load(member_dir / "best_checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        current, probabilities, logits_by_task = predict_member(model, loader, device)
        if base is None:
            base = current
        elif current["material_id"].tolist() != base["material_id"].tolist():
            raise RuntimeError("Unknown prediction order changed across members.")
        for task in TASKS:
            member_values[task].append(np.asarray(probabilities[task], dtype=float))
            member_logits[task].append(np.asarray(logits_by_task[task], dtype=float))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if base is None:
        raise RuntimeError("No ensemble predictions were generated.")

    result = base.copy()
    temperatures = json.loads(
        (full_dir / "ensemble" / "temperatures.json").read_text(encoding="utf-8")
    )

    for task in TASKS:
        matrix = np.vstack(member_values[task])
        stats = aggregate_binary_probabilities(matrix)
        result[f"p_{task}_mean"] = stats["mean"]
        result[f"p_{task}_std"] = stats["std"]
        result[f"{task}_predictive_entropy"] = stats["predictive_entropy"]
        result[f"{task}_expected_entropy"] = stats["expected_entropy"]
        result[f"{task}_mutual_information"] = stats["mutual_information"]

        logits_matrix = np.vstack(member_logits[task])
        result[f"p_{task}_calibrated"] = sigmoid(
            logits_matrix / float(temperatures[task])
        ).mean(axis=0)

        for index, (values, logits_values) in enumerate(zip(matrix, logits_matrix), start=1):
            result[f"p_{task}_member_{index}"] = values
            result[f"logit_{task}_member_{index}"] = logits_values

    ranking_configuration = json.loads(
        (full_dir / "ranking" / "ranking_configuration.json").read_text(encoding="utf-8")
    )
    uncertainty_column = ranking_configuration["uncertainty_column"]
    selected_lambda = float(ranking_configuration["lambda"])
    top_k = int(experiment["ranking"]["top_k_unknown"])
    result = rank_all_unknown(
        result,
        uncertainty_column=uncertainty_column,
        uncertainty_lower=float(ranking_configuration["uncertainty_q05"]),
        uncertainty_upper=float(ranking_configuration["uncertainty_q95"]),
        selected_lambda=selected_lambda,
        top_k=top_k,
    )
    # The Thermo-positive subset is retained only as a transparent diagnostic;
    # it no longer controls pJ ranking, Top-K selection, or CIF export.
    eligible = result[result["thermo_eligible"]].copy()

    result.to_csv(output / "unknown_all_predictions.csv", index=False)
    eligible.to_csv(output / "thermo_eligible_predictions.csv", index=False)

    probability_top = select_top_k_by_rank(result, "probability_rank", top_k)
    risk_top = select_top_k_by_rank(result, "risk_rank", top_k)

    # Rank values are defined over all 8,803 Unknown rows. Display rank is local
    # to each exported Top-K table.
    probability_top["topk_display_rank"] = np.arange(1, len(probability_top) + 1)
    risk_top["topk_display_rank"] = np.arange(1, len(risk_top) + 1)

    probability_top.to_csv(output / "probability_top100.csv", index=False)
    risk_top.to_csv(output / "risk_top100.csv", index=False)

    report_k = list(experiment["ranking"].get("report_k", [10, 20, 50, 100]))
    diagnostics_k = sorted(set(report_k + [top_k]))
    overlap_table = topk_overlap_table(result, diagnostics_k)
    overlap_table.to_csv(output / "probability_vs_risk_overlap.csv", index=False)

    spearman = spearmanr(
        result["probability_rank"], result["risk_rank"], nan_policy="omit"
    )
    kendall = kendalltau(
        result["probability_rank"], result["risk_rank"], nan_policy="omit"
    )
    population_summary = screening_population_summary(result)
    logic_summary = joint_logic_violation_summary(result)
    ranking_diagnostics = {
        **population_summary,
        "lambda": selected_lambda,
        "lambda_is_zero": bool(abs(selected_lambda) < 1e-12),
        "uncertainty_method": ranking_configuration.get("uncertainty_method"),
        "uncertainty_score": ranking_configuration.get("uncertainty_score"),
        "uncertainty_column": uncertainty_column,
        "pJ_alias": "p_joint_mean",
        "pJ_calibrated_alias": "p_joint_calibrated",
        "logical_consistency_calibrated": logic_summary,
        "spearman_probability_rank_vs_risk_rank": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "kendall_tau_probability_rank_vs_risk_rank": float(kendall.statistic),
        "kendall_pvalue": float(kendall.pvalue),
        "top_k_overlap": overlap_table.to_dict(orient="records"),
        "confidence_tier_counts_probability_topk": {
            str(k): int(v)
            for k, v in probability_top["confidence_tier"].value_counts().to_dict().items()
        },
        "screening_tier_counts_probability_topk": {
            str(k): int(v)
            for k, v in probability_top["screening_tier"].value_counts().to_dict().items()
        },
    }
    save_json(ranking_diagnostics, output / "candidate_ranking_diagnostics.json")

    with pd.ExcelWriter(output / "unknown_screening_top100.xlsx", engine="openpyxl") as writer:
        probability_top.to_excel(writer, sheet_name="Probability Top100", index=False)
        risk_top.to_excel(writer, sheet_name="Risk Top100", index=False)
        overlap_table.to_excel(writer, sheet_name="Ranking Diagnostics", index=False)
        pd.DataFrame(
            [
                {
                    "metric": "lambda",
                    "value": selected_lambda,
                },
                {
                    "metric": "spearman probability-rank vs risk-rank",
                    "value": float(spearman.statistic),
                },
                {
                    "metric": "kendall tau probability-rank vs risk-rank",
                    "value": float(kendall.statistic),
                },
            ]
        ).to_excel(writer, sheet_name="Rank Correlation", index=False)

    probability_top.to_excel(output / "probability_top100.xlsx", index=False)
    risk_top.to_excel(output / "risk_top100.xlsx", index=False)

    db = connect(experiment["paths"]["db_path"])
    export_cifs(probability_top, db, output / "probability_top100_cif")
    export_cifs(risk_top, db, output / "risk_top100_cif")

    overlap_count = len(
        set(probability_top["material_id"]).intersection(set(risk_top["material_id"]))
    )
    summary = {
        "status": "completed",
        **population_summary,
        "top_k": top_k,
        "probability_risk_overlap": overlap_count,
        "probability_risk_overlap_fraction": overlap_count / max(len(probability_top), 1),
        "lambda": selected_lambda,
        "lambda_is_zero": bool(abs(selected_lambda) < 1e-12),
        "pJ_alias": "p_joint_mean",
        "pJ_calibrated_alias": "p_joint_calibrated",
        "logical_consistency_calibrated": logic_summary,
        "ranking_configuration": ranking_configuration,
        "ranking_diagnostics_file": "candidate_ranking_diagnostics.json",
        "interpretation_note": (
            "All Unknown materials are ranked directly by calibrated pJ; the "
            "Thermo-positive subset is retained only as a diagnostic. "
            "Risk-aware ranking is retained as a secondary decision analysis. "
            "Confidence tiers are descriptive strata based on the validation-frozen robust uncertainty scale; "
            "they are not formal coverage guarantees."
        ),
    }
    save_json(summary, output / "screening_summary.json")
    save_json(summary, completed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
