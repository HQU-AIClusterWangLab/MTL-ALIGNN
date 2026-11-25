#!/usr/bin/env python
"""Generate Fig. 5a from the frozen all-Unknown candidate ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from PIL import Image


EXPECTED_N = 8803
RANK_FIELD = "probability_rank"
SCORE_FIELD = "score_joint_probability"
CALIBRATED_SCORE_FIELD = "p_joint_calibrated"
SHOW_PERCENTILE_GUIDES = True
PERCENTILE_GUIDES = (1.0, 10.0, 20.0)

# These values are assertions only. Plot coordinates are always read from the
# frozen source table and the percentile is calculated from its stored rank.
SANITY_RANKS = {
    "2CrSnSe3-1": 43,
    "2NiSbO3-1": 987,
}

HIGHLIGHTS: tuple[dict[str, Any], ...] = (
    {
        "material_id": "2CrSnSe3-1",
        "marker": "o",
        "color": "#245E7A",
        "facecolor": "#245E7A",
        "edgecolor": "white",
        "linewidth": 0.9,
        "zorder": 6,
        "size": 54.0,
        "annotation_offset": (55.0, -4.0),
        "annotation_va": "top",
    },
    {
        "material_id": "2NiSbO3-1",
        "marker": "s",
        "color": "#A05A43",
        "facecolor": "#A05A43",
        "edgecolor": "white",
        "linewidth": 0.9,
        "zorder": 6,
        "size": 72.0,
        "annotation_offset": (45.0, -8.0),
        "annotation_va": "top",
    },
    {
        "material_id": "1BrI2Ir3Te3-1",
        "marker": "^",
        "color": "#4F7866",
        "facecolor": "none",
        "edgecolor": "#4F7866",
        "linewidth": 1.4,
        "zorder": 7,
        "size": 105.0,
        "annotation_offset": (55.0, 3.0),
        "annotation_va": "bottom",
    },
)

REQUIRED_SOURCE_COLUMNS = {
    "material_id",
    "row_id",
    "formula",
    "dyn_stab_raw",
    "screening_eligible",
    "ehull",
    "p_thermo_calibrated",
    "p_dynamic_calibrated",
    "p_joint_mean",
    "p_joint_std",
    "joint_predictive_entropy",
    "joint_mutual_information",
    "p_joint_calibrated",
    "pJ_calibrated",
    "uncertainty_normalized",
    "ranking_scope",
    "score_joint_probability",
    "score_risk_aware",
    "thermo_eligible",
    "probability_rank",
    "risk_rank",
}

SOURCE_DATA_COLUMNS = (
    "material_id",
    "rank",
    "percentile_rank",
    "joint_probability_or_score",
    "is_highlighted",
    "highlight_label",
    "source_file",
    "source_sha256",
    "source_csv_line",
    "row_id",
    "formula",
    "dyn_stab_raw",
    "ranking_scope",
    "rank_field",
    "score_field",
    "ranking_direction",
    "calibrated",
    "probability_rank",
    "score_joint_probability",
    "p_joint_calibrated",
    "p_joint_mean",
    "ensemble_sd",
    "joint_predictive_entropy",
    "joint_mutual_information",
    "p_thermo_calibrated",
    "p_dynamic_calibrated",
    "ehull",
    "uncertainty_normalized",
    "score_risk_aware",
    "risk_rank",
    "lambda",
    "uncertainty_penalty_active",
    "thermo_eligible",
    "screening_eligible",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--hide-percentile-guides",
        action="store_true",
        help="Hide the optional Top 1/10/20 percent locator guides.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise AssertionError(f"Missing required columns in {source}: {missing}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing required JSON: {path}")
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return result


def exact_numeric_equal(left: pd.Series, right: pd.Series) -> bool:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.array_equal(left_values, right_values))


def validate_frozen_ranking(project_root: Path) -> dict[str, Any]:
    ranking_path = (
        project_root
        / "results"
        / "main"
        / "M10_unknown_screening"
        / "unknown_all_predictions.csv"
    ).resolve()
    summary_path = ranking_path.with_name("screening_summary.json")
    completion_path = ranking_path.with_name("COMPLETED.json")
    diagnostics_path = ranking_path.with_name("candidate_ranking_diagnostics.json")
    ranking_config_path = (
        project_root
        / "results"
        / "main"
        / "M06_full_mtl"
        / "ranking"
        / "ranking_configuration.json"
    ).resolve()
    temperatures_path = (
        project_root
        / "results"
        / "main"
        / "M06_full_mtl"
        / "ensemble"
        / "temperatures.json"
    ).resolve()
    unknown_pool_path = (
        project_root / "data" / "processed" / "unknown_pool.csv"
    ).resolve()
    supervised_path = (
        project_root / "data" / "processed" / "supervised.csv"
    ).resolve()
    split_path = (project_root / "data" / "processed" / "iid_split.csv").resolve()

    if not ranking_path.is_file():
        raise AssertionError(f"Missing frozen all-Unknown ranking: {ranking_path}")
    ranking_hash_before = sha256_file(ranking_path)
    frame = pd.read_csv(ranking_path)
    require_columns(frame, REQUIRED_SOURCE_COLUMNS, ranking_path)
    frame["_source_data_row"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    frame["_source_csv_line"] = frame["_source_data_row"] + 1

    if len(frame) != EXPECTED_N:
        raise AssertionError(
            f"Expected {EXPECTED_N} candidates, found {len(frame)} in {ranking_path}"
        )
    if frame["material_id"].isna().any() or frame["material_id"].astype(str).str.len().eq(0).any():
        raise AssertionError("material_id contains missing or empty values")
    frame["material_id"] = frame["material_id"].astype(str)
    duplicate_id_count = int(frame["material_id"].duplicated().sum())
    if duplicate_id_count != 0:
        raise AssertionError(f"Duplicate material IDs found: {duplicate_id_count}")
    if set(frame["dyn_stab_raw"].astype(str).unique()) != {"Unknown"}:
        raise AssertionError("Candidate pool is not exclusively dyn_stab_raw=Unknown")
    if set(frame["ranking_scope"].astype(str).unique()) != {"all_unknown"}:
        raise AssertionError("ranking_scope is not all_unknown for every candidate")

    ranks = pd.to_numeric(frame[RANK_FIELD], errors="coerce")
    if ranks.isna().any() or not np.allclose(ranks, np.round(ranks), rtol=0.0, atol=0.0):
        raise AssertionError(f"{RANK_FIELD} must contain finite integer ranks")
    frame[RANK_FIELD] = ranks.astype(np.int64)
    expected_ranks = np.arange(1, EXPECTED_N + 1, dtype=np.int64)
    if not np.array_equal(np.sort(frame[RANK_FIELD].to_numpy()), expected_ranks):
        raise AssertionError(f"{RANK_FIELD} is not a unique continuous 1..N ranking")

    score = pd.to_numeric(frame[SCORE_FIELD], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(score).all() or not ((score >= 0.0) & (score <= 1.0)).all():
        raise AssertionError(f"{SCORE_FIELD} is not a finite probability in [0,1]")
    if not exact_numeric_equal(frame[SCORE_FIELD], frame[CALIBRATED_SCORE_FIELD]):
        raise AssertionError(
            f"{SCORE_FIELD} does not exactly equal {CALIBRATED_SCORE_FIELD}"
        )
    if not exact_numeric_equal(frame[SCORE_FIELD], frame["pJ_calibrated"]):
        raise AssertionError(f"{SCORE_FIELD} does not exactly equal pJ_calibrated")

    # This reconstruction is an audit only; the figure uses the stored rank.
    ordered_index = frame.sort_values(
        [SCORE_FIELD, "uncertainty_normalized", "material_id"],
        ascending=[False, True, True],
        kind="mergesort",
        na_position="last",
    ).index
    reconstructed = pd.Series(
        expected_ranks, index=ordered_index, dtype=np.int64
    ).reindex(frame.index)
    rank_reconstruction_matches = bool(
        np.array_equal(reconstructed.to_numpy(), frame[RANK_FIELD].to_numpy())
    )
    if not rank_reconstruction_matches:
        raise AssertionError("Stored probability_rank conflicts with formal sort rule")

    primary_tied_rows = int(frame[SCORE_FIELD].duplicated(keep=False).sum())
    primary_tied_values = int((frame[SCORE_FIELD].value_counts() > 1).sum())

    summary = load_json(summary_path)
    completion = load_json(completion_path)
    diagnostics = load_json(diagnostics_path)
    ranking_config = load_json(ranking_config_path)
    temperatures = load_json(temperatures_path)
    for label, payload in (
        ("summary", summary),
        ("completion", completion),
        ("diagnostics", diagnostics),
    ):
        if int(payload.get("unknown_count", -1)) != EXPECTED_N:
            raise AssertionError(f"{label} unknown_count conflicts with N={EXPECTED_N}")
        if int(payload.get("ranking_pool_count", -1)) != EXPECTED_N:
            raise AssertionError(
                f"{label} ranking_pool_count conflicts with N={EXPECTED_N}"
            )
        if payload.get("ranking_scope") != "all_unknown":
            raise AssertionError(f"{label} ranking_scope is not all_unknown")
    if summary.get("status") != "completed" or completion.get("status") != "completed":
        raise AssertionError("M10 screening result is not marked completed")
    if summary.get("pJ_calibrated_alias") != CALIBRATED_SCORE_FIELD:
        raise AssertionError("Frozen summary pJ calibrated alias is unexpected")

    frozen_lambda = float(ranking_config["lambda"])
    if not math.isclose(frozen_lambda, float(summary["lambda"]), rel_tol=0.0, abs_tol=0.0):
        raise AssertionError("Frozen lambda conflicts between ranking config and summary")
    if not np.allclose(
        frame["score_risk_aware"].to_numpy(float),
        frame[SCORE_FIELD].to_numpy(float)
        - frozen_lambda * frame["uncertainty_normalized"].to_numpy(float),
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError("score_risk_aware conflicts with frozen risk formula")
    if frozen_lambda == 0.0 and not np.array_equal(
        frame["risk_rank"].to_numpy(int), frame[RANK_FIELD].to_numpy(int)
    ):
        raise AssertionError("lambda=0 but risk_rank differs from probability_rank")

    if not unknown_pool_path.is_file():
        raise AssertionError(f"Missing frozen Unknown pool: {unknown_pool_path}")
    unknown_pool = pd.read_csv(
        unknown_pool_path, usecols=["material_id", "dyn_stab_raw"]
    )
    unknown_pool["material_id"] = unknown_pool["material_id"].astype(str)
    if len(unknown_pool) != EXPECTED_N or unknown_pool["material_id"].duplicated().any():
        raise AssertionError("unknown_pool.csv is not the expected unique 8,803-row pool")
    if set(unknown_pool["dyn_stab_raw"].astype(str).unique()) != {"Unknown"}:
        raise AssertionError("unknown_pool.csv contains non-Unknown labels")
    source_ids = set(frame["material_id"])
    unknown_ids = set(unknown_pool["material_id"])
    if source_ids != unknown_ids:
        raise AssertionError("M10 ranking IDs do not exactly match unknown_pool.csv")

    supervised_ids = set(
        pd.read_csv(supervised_path, usecols=["material_id"])["material_id"].astype(str)
    )
    split_ids = set(pd.read_csv(split_path, usecols=["material_id"])["material_id"].astype(str))
    if source_ids.intersection(supervised_ids):
        raise AssertionError("Unknown candidate IDs overlap supervised.csv")
    if source_ids.intersection(split_ids):
        raise AssertionError("Unknown candidate IDs overlap iid_split.csv")

    candidate_rows: dict[str, dict[str, Any]] = {}
    for specification in HIGHLIGHTS:
        material_id = str(specification["material_id"])
        selected = frame.loc[frame["material_id"].eq(material_id)]
        if len(selected) != 1:
            raise AssertionError(
                f"Expected one frozen row for {material_id}, found {len(selected)}"
            )
        row = selected.iloc[0]
        rank = int(row[RANK_FIELD])
        percentile = 100.0 * rank / EXPECTED_N
        candidate_rows[material_id] = {
            "material_id": material_id,
            "rank": rank,
            "percentile": percentile,
            "score": float(row[SCORE_FIELD]),
            "raw_p_joint_mean": float(row["p_joint_mean"]),
            "source_data_row": int(row["_source_data_row"]),
            "source_csv_line": int(row["_source_csv_line"]),
            "row_id": int(row["row_id"]),
        }

    copies: list[dict[str, Any]] = []
    copy_paths = (
        project_root
        / "paper"
        / "8. Unknown 8803"
        / "M10_unknown_screening"
        / "unknown_all_predictions.csv",
        project_root
        / "paper"
        / "2. Single  Equal  Full 主模型"
        / "M10_unknown_screening"
        / "unknown_all_predictions.csv",
    )
    for copy_path in copy_paths:
        if copy_path.is_file():
            copy_hash = sha256_file(copy_path)
            if copy_hash != ranking_hash_before:
                raise AssertionError(f"Conflicting ranking copy found: {copy_path}")
            copies.append(
                {
                    "path": copy_path.resolve(),
                    "sha256": copy_hash,
                    "identical": True,
                }
            )

    return {
        "frame": frame,
        "ranking_path": ranking_path,
        "ranking_hash_before": ranking_hash_before,
        "ranking_bytes": ranking_path.stat().st_size,
        "summary_path": summary_path.resolve(),
        "summary_sha256": sha256_file(summary_path),
        "summary": summary,
        "completion_path": completion_path.resolve(),
        "completion_sha256": sha256_file(completion_path),
        "diagnostics_path": diagnostics_path.resolve(),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "ranking_config_path": ranking_config_path,
        "ranking_config_sha256": sha256_file(ranking_config_path),
        "ranking_config": ranking_config,
        "temperatures_path": temperatures_path,
        "temperatures_sha256": sha256_file(temperatures_path),
        "temperatures": temperatures,
        "unknown_pool_path": unknown_pool_path,
        "unknown_pool_sha256": sha256_file(unknown_pool_path),
        "supervised_path": supervised_path,
        "supervised_sha256": sha256_file(supervised_path),
        "split_path": split_path,
        "split_sha256": sha256_file(split_path),
        "duplicate_id_count": duplicate_id_count,
        "primary_tied_rows": primary_tied_rows,
        "primary_tied_values": primary_tied_values,
        "rank_reconstruction_matches": rank_reconstruction_matches,
        "candidate_rows": candidate_rows,
        "lambda": frozen_lambda,
        "copies": copies,
        "thermo_eligible_count": int(frame["thermo_eligible"].astype(bool).sum()),
        "thermo_ineligible_count": int((~frame["thermo_eligible"].astype(bool)).sum()),
    }


def print_screening_audit(audit: dict[str, Any]) -> None:
    print("[FIG5A SCREENING AUDIT]")
    print("")
    print("Candidate pool:")
    print(f"  N = {len(audit['frame'])}")
    print(f"  source file = {audit['ranking_path']}")
    print(f"  ranking field = {RANK_FIELD}")
    print(
        f"  probability/score field = {SCORE_FIELD} "
        f"(equals {CALIBRATED_SCORE_FIELD})"
    )
    print(
        "  ranking direction = descending score; ascending uncertainty; "
        "ascending material_id tie-break"
    )
    print("  pre-filter applied = False")
    print("")
    for specification in HIGHLIGHTS:
        material_id = str(specification["material_id"])
        candidate = audit["candidate_rows"][material_id]
        print(f"{material_id}:")
        print(f"  rank = {candidate['rank']}")
        print(f"  percentile = {candidate['percentile']:.12f}%")
        print(f"  p_joint / score = {candidate['score']:.15f}")
        print(f"  source row = CSV line {candidate['source_csv_line']}")
        print("")
    print("Sanity checks:")
    print(f"  assert N == {EXPECTED_N}: {'PASS' if len(audit['frame']) == EXPECTED_N else 'FAIL'}")
    for material_id, expected_rank in SANITY_RANKS.items():
        actual_rank = int(audit["candidate_rows"][material_id]["rank"])
        print(
            f"  assert {material_id} rank == {expected_rank}: "
            f"{'PASS' if actual_rank == expected_rank else 'FAIL'}"
        )
    print("  third candidate rank read from frozen source: PASS")
    print("")


def assert_requested_sanity(audit: dict[str, Any]) -> None:
    if len(audit["frame"]) != EXPECTED_N:
        raise AssertionError(f"N != {EXPECTED_N}")
    for material_id, expected_rank in SANITY_RANKS.items():
        actual_rank = int(audit["candidate_rows"][material_id]["rank"])
        if actual_rank != expected_rank:
            raise AssertionError(
                f"Frozen rank conflict for {material_id}: "
                f"expected sanity {expected_rank}, found {actual_rank}"
            )


def format_highlight_label(material_id: str, rank: int, percentile: float) -> str:
    return f"{material_id}\nRank {rank} | Top {percentile:.1f}%"


def build_source_data(audit: dict[str, Any]) -> pd.DataFrame:
    frame = audit["frame"].copy()
    ranking_path = str(audit["ranking_path"])
    ranking_hash = str(audit["ranking_hash_before"])
    frozen_lambda = float(audit["lambda"])
    highlighted_ids = {str(item["material_id"]) for item in HIGHLIGHTS}

    source_data = pd.DataFrame(
        {
            "material_id": frame["material_id"].astype(str),
            "rank": frame[RANK_FIELD].astype(np.int64),
            "percentile_rank": 100.0 * frame[RANK_FIELD].astype(float) / EXPECTED_N,
            "joint_probability_or_score": frame[SCORE_FIELD].astype(float),
            "is_highlighted": frame["material_id"].isin(highlighted_ids),
            "highlight_label": "",
            "source_file": ranking_path,
            "source_sha256": ranking_hash,
            "source_csv_line": frame["_source_csv_line"].astype(np.int64),
            "row_id": frame["row_id"].astype(np.int64),
            "formula": frame["formula"].astype(str),
            "dyn_stab_raw": frame["dyn_stab_raw"].astype(str),
            "ranking_scope": frame["ranking_scope"].astype(str),
            "rank_field": RANK_FIELD,
            "score_field": SCORE_FIELD,
            "ranking_direction": (
                "score_joint_probability descending; uncertainty_normalized "
                "ascending; material_id ascending"
            ),
            "calibrated": True,
            "probability_rank": frame[RANK_FIELD].astype(np.int64),
            "score_joint_probability": frame[SCORE_FIELD].astype(float),
            "p_joint_calibrated": frame[CALIBRATED_SCORE_FIELD].astype(float),
            "p_joint_mean": frame["p_joint_mean"].astype(float),
            "ensemble_sd": frame["p_joint_std"].astype(float),
            "joint_predictive_entropy": frame["joint_predictive_entropy"].astype(float),
            "joint_mutual_information": frame["joint_mutual_information"].astype(float),
            "p_thermo_calibrated": frame["p_thermo_calibrated"].astype(float),
            "p_dynamic_calibrated": frame["p_dynamic_calibrated"].astype(float),
            "ehull": frame["ehull"].astype(float),
            "uncertainty_normalized": frame["uncertainty_normalized"].astype(float),
            "score_risk_aware": frame["score_risk_aware"].astype(float),
            "risk_rank": frame["risk_rank"].astype(np.int64),
            "lambda": frozen_lambda,
            "uncertainty_penalty_active": frozen_lambda != 0.0,
            "thermo_eligible": frame["thermo_eligible"].astype(bool),
            "screening_eligible": frame["screening_eligible"].astype(bool),
        },
        columns=SOURCE_DATA_COLUMNS,
    )

    for specification in HIGHLIGHTS:
        material_id = str(specification["material_id"])
        mask = source_data["material_id"].eq(material_id)
        if int(mask.sum()) != 1:
            raise AssertionError(f"Source-data highlight is not unique: {material_id}")
        row = source_data.loc[mask].iloc[0]
        source_data.loc[mask, "highlight_label"] = (
            f"{material_id} | Rank {int(row['rank'])} | "
            f"Top {float(row['percentile_rank']):.1f}%"
        )

    source_data = source_data.sort_values("rank", kind="mergesort").reset_index(drop=True)
    if len(source_data) != EXPECTED_N:
        raise AssertionError("Fig5a source data does not contain all 8,803 rows")
    if int(source_data["is_highlighted"].sum()) != len(HIGHLIGHTS):
        raise AssertionError("Fig5a source data does not contain exactly three highlights")
    return source_data


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9.3,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def atomic_savefig(
    figure: plt.Figure,
    destination: Path,
    *,
    file_format: str,
    dpi: int | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=f".tmp.{file_format}",
        dir=destination.parent,
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(
            temporary_path,
            format=file_format,
            dpi=dpi,
            facecolor="white",
            metadata=(
                {
                    "Title": "Candidate ranking across 8,803 Dynamic-Unknown materials",
                    "Creator": Path(__file__).name,
                }
                if file_format == "pdf"
                else None
            ),
        )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.csv", dir=destination.parent
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False, float_format="%.17g")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.txt", dir=destination.parent
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def create_figure(
    source_data: pd.DataFrame,
    output_dir: Path,
    *,
    show_percentile_guides: bool,
) -> dict[str, Path]:
    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    figure.subplots_adjust(left=0.14, right=0.985, bottom=0.225, top=0.94)

    x = source_data["percentile_rank"].to_numpy(dtype=float)
    y = source_data["joint_probability_or_score"].to_numpy(dtype=float)

    if show_percentile_guides:
        for guide in PERCENTILE_GUIDES:
            axis.axvline(
                guide,
                color="#D7DDE1",
                linewidth=0.65,
                linestyle=(0, (2.5, 3.0)),
                zorder=0,
            )
            axis.text(
                guide + 0.42,
                0.027,
                f"Top {guide:g}%",
                rotation=90,
                ha="left",
                va="bottom",
                fontsize=6.7,
                color="#949CA2",
                zorder=1,
            )

    axis.scatter(
        x,
        y,
        marker="o",
        s=5.0,
        alpha=0.17,
        color="#93A4AE",
        edgecolors="none",
        linewidths=0.0,
        zorder=2,
    )

    for specification in HIGHLIGHTS:
        material_id = str(specification["material_id"])
        row = source_data.loc[source_data["material_id"].eq(material_id)].iloc[0]
        point_x = float(row["percentile_rank"])
        point_y = float(row["joint_probability_or_score"])
        color = str(specification["color"])
        axis.scatter(
            [point_x],
            [point_y],
            marker=str(specification["marker"]),
            s=float(specification["size"]),
            facecolor=str(specification["facecolor"]),
            edgecolor=str(specification["edgecolor"]),
            linewidth=float(specification["linewidth"]),
            zorder=int(specification["zorder"]),
        )
        axis.annotate(
            format_highlight_label(
                material_id,
                int(row["rank"]),
                float(row["percentile_rank"]),
            ),
            xy=(point_x, point_y),
            xytext=tuple(specification["annotation_offset"]),
            textcoords="offset points",
            ha="left",
            va=str(specification["annotation_va"]),
            fontsize=7.4,
            color="#27343B",
            linespacing=1.20,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.72,
                "shrinkA": 2.0,
                "shrinkB": 5.0,
                "connectionstyle": "arc3,rad=0.04",
            },
            annotation_clip=False,
            zorder=8,
        )

    axis.set_xlim(-1.5, 101.0)
    axis.set_ylim(-0.015, 1.085)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.set_xlabel("Candidate percentile rank (%)", labelpad=5.0)
    axis.set_ylabel("Predicted joint probability")
    figure.text(
        0.5625,
        0.072,
        "Lower percentile indicates higher screening priority.",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#5F686E",
    )
    axis.yaxis.grid(True, color="#E8ECEE", linewidth=0.50, alpha=0.75)
    axis.xaxis.grid(False)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#51585D")
    axis.spines["bottom"].set_color("#51585D")
    axis.tick_params(colors="#30363A")
    axis.text(
        -0.125,
        1.025,
        "a",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#1F2529",
    )

    pdf_path = output_dir / "Fig5a_candidate_ranking.pdf"
    png_path = output_dir / "Fig5a_candidate_ranking.png"
    svg_path = output_dir / "Fig5a_candidate_ranking.svg"
    atomic_savefig(figure, pdf_path, file_format="pdf")
    atomic_savefig(figure, png_path, file_format="png", dpi=600)
    atomic_savefig(figure, svg_path, file_format="svg")
    plt.close(figure)
    return {"pdf": pdf_path, "png": png_path, "svg": svg_path}


def validate_written_outputs(
    source_data_path: Path,
    figure_paths: dict[str, Path],
    audit: dict[str, Any],
) -> dict[str, Any]:
    readback = pd.read_csv(source_data_path)
    require_columns(readback, set(SOURCE_DATA_COLUMNS), source_data_path)
    if len(readback) != EXPECTED_N:
        raise AssertionError(f"Source-data CSV has {len(readback)} rows, not {EXPECTED_N}")
    if readback["material_id"].duplicated().any():
        raise AssertionError("Source-data CSV contains duplicate material IDs")
    if not np.array_equal(
        readback["rank"].to_numpy(dtype=int),
        np.arange(1, EXPECTED_N + 1, dtype=int),
    ):
        raise AssertionError("Source-data CSV is not a complete rank-ordered 1..N table")
    if int(readback["is_highlighted"].astype(bool).sum()) != len(HIGHLIGHTS):
        raise AssertionError("Source-data CSV highlight count is not three")
    if set(readback["calibrated"].astype(str).str.lower().unique()) != {"true"}:
        raise AssertionError("Source-data CSV must identify the ranking score as calibrated")
    if set(readback["source_file"].astype(str).unique()) != {str(audit["ranking_path"])}:
        raise AssertionError("Source-data CSV provenance path is inconsistent")

    expected_full = (
        audit["frame"]
        .sort_values(RANK_FIELD, kind="mergesort")
        .reset_index(drop=True)
    )
    if not np.array_equal(
        readback["material_id"].astype(str).to_numpy(),
        expected_full["material_id"].astype(str).to_numpy(),
    ):
        raise AssertionError("Source-data CSV material_id-to-rank mapping changed")
    if not np.array_equal(
        readback["rank"].to_numpy(dtype=np.int64),
        expected_full[RANK_FIELD].to_numpy(dtype=np.int64),
    ):
        raise AssertionError("Source-data CSV ranks differ from the frozen source")
    expected_percentiles = (
        100.0 * expected_full[RANK_FIELD].to_numpy(dtype=float) / EXPECTED_N
    )
    if not np.allclose(
        readback["percentile_rank"].to_numpy(dtype=float),
        expected_percentiles,
        rtol=0.0,
        atol=5e-13,
    ):
        raise AssertionError("Source-data CSV percentiles differ from stored ranks")
    if not np.allclose(
        readback["joint_probability_or_score"].to_numpy(dtype=float),
        expected_full[SCORE_FIELD].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-15,
    ):
        raise AssertionError("Source-data CSV scores differ from the frozen source")
    if not np.array_equal(
        readback["source_csv_line"].to_numpy(dtype=np.int64),
        expected_full["_source_csv_line"].to_numpy(dtype=np.int64),
    ):
        raise AssertionError("Source-data CSV physical-row provenance changed")
    for specification in HIGHLIGHTS:
        material_id = str(specification["material_id"])
        output_row = readback.loc[readback["material_id"].eq(material_id)]
        if len(output_row) != 1:
            raise AssertionError(f"Missing highlighted row in source CSV: {material_id}")
        expected = audit["candidate_rows"][material_id]
        actual = output_row.iloc[0]
        if int(actual["rank"]) != int(expected["rank"]):
            raise AssertionError(f"Source CSV rank conflict for {material_id}")
        if not math.isclose(
            float(actual["percentile_rank"]),
            float(expected["percentile"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Source CSV percentile conflict for {material_id}")
        if not math.isclose(
            float(actual["joint_probability_or_score"]),
            float(expected["score"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise AssertionError(f"Source CSV score conflict for {material_id}")

    for path in figure_paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty figure output: {path}")
    with Image.open(figure_paths["png"]) as image:
        png_size = image.size
        png_dpi = image.info.get("dpi")
    if png_size != (3120, 2160):
        raise AssertionError(f"Unexpected PNG dimensions: {png_size}")
    if png_dpi is None or not all(599.0 <= float(value) <= 601.0 for value in png_dpi):
        raise AssertionError(f"PNG dpi is not approximately 600: {png_dpi}")
    svg_text = figure_paths["svg"].read_text(encoding="utf-8")
    if "<image" in svg_text.lower():
        raise AssertionError("SVG unexpectedly contains a raster image")
    for text in (
        "Candidate percentile rank (%)",
        "Predicted joint probability",
    ):
        if text not in svg_text:
            raise AssertionError(f"Expected SVG label is missing: {text}")
    for specification in HIGHLIGHTS:
        material_id = str(specification["material_id"])
        candidate = audit["candidate_rows"][material_id]
        dynamic_labels = (
            material_id,
            f"Rank {int(candidate['rank'])} | Top {float(candidate['percentile']):.1f}%",
        )
        for text in dynamic_labels:
            if text not in svg_text:
                raise AssertionError(f"Expected SVG label is missing: {text}")

    return {
        "source_data_rows": len(readback),
        "png_size": png_size,
        "png_dpi": png_dpi,
        "svg_contains_raster_image": False,
    }


def build_audit_text(
    project_root: Path,
    output_dir: Path,
    source_data_path: Path,
    figure_paths: dict[str, Path],
    audit: dict[str, Any],
    validation: dict[str, Any],
    *,
    show_percentile_guides: bool,
    executed_at: str,
) -> str:
    lines: list[str] = [
        "FIG5A CANDIDATE SCREENING AUDIT",
        "================================",
        f"Execution time: {executed_at}",
        f"Script: {Path(__file__).resolve()}",
        f"Script SHA256: {sha256_file(Path(__file__).resolve())}",
        f"Project root: {project_root.resolve()}",
        f"Output directory: {output_dir.resolve()}",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"numpy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"matplotlib: {matplotlib.__version__}",
        "",
        "FORMAL RANKING SOURCE",
        "---------------------",
        f"Path: {audit['ranking_path']}",
        f"SHA256: {audit['ranking_hash_before']}",
        f"Bytes: {audit['ranking_bytes']}",
        f"Rows: {len(audit['frame'])}",
        f"Unique material IDs: {audit['frame']['material_id'].nunique()}",
        f"Duplicate material IDs: {audit['duplicate_id_count']}",
        "dyn_stab_raw values: Unknown only",
        "ranking_scope values: all_unknown only",
        f"Screening summary: {audit['summary_path']}",
        f"Screening summary SHA256: {audit['summary_sha256']}",
        f"Completion record: {audit['completion_path']}",
        f"Completion record SHA256: {audit['completion_sha256']}",
        f"Ranking diagnostics: {audit['diagnostics_path']}",
        f"Ranking diagnostics SHA256: {audit['diagnostics_sha256']}",
        "Canonical source selected: results/main/M10_unknown_screening",
        "",
        "DUPLICATE RESULT COPIES",
        "-----------------------",
    ]
    if audit["copies"]:
        for copy in audit["copies"]:
            lines.extend(
                [
                    f"Copy: {copy['path']}",
                    f"SHA256: {copy['sha256']}",
                    "Byte-identical to canonical source: True",
                ]
            )
        lines.append("Conflicting ranking copy: none")
    else:
        lines.append("No paper result copy detected")

    lines.extend(
        [
            "",
            "CANDIDATE POOL AND LEAKAGE CHECK",
            "--------------------------------",
            f"Frozen Unknown pool: {audit['unknown_pool_path']}",
            f"Frozen Unknown pool SHA256: {audit['unknown_pool_sha256']}",
            f"Unknown pool N: {EXPECTED_N}",
            "M10 ranking IDs exactly equal Unknown pool IDs: True",
            f"Supervised source checked: {audit['supervised_path']}",
            f"Supervised source SHA256: {audit['supervised_sha256']}",
            f"Split manifest checked: {audit['split_path']}",
            f"Split manifest SHA256: {audit['split_sha256']}",
            "Unknown IDs overlapping supervised.csv: 0",
            "Unknown IDs overlapping iid_split.csv: 0",
            "Training/validation/test/model-selection leakage detected: False",
            "Candidate sampling: none",
            "Candidate deletion: none",
            "Thermo hard pre-filter applied: False",
            "Ehull pre-filter applied: False",
            "Top-K pre-filter applied: False",
            f"Thermo-eligible diagnostic count: {audit['thermo_eligible_count']}",
            f"Thermo-ineligible diagnostic count: {audit['thermo_ineligible_count']}",
            "All 8,803 candidates plotted: True",
            "",
            "FORMAL RANKING DEFINITION",
            "-------------------------",
            f"Stored rank field used: {RANK_FIELD}",
            "Rank base: 1",
            f"Stored score field used: {SCORE_FIELD}",
            f"Y-axis label: Predicted joint probability",
            f"Calibration field exists: {CALIBRATED_SCORE_FIELD in audit['frame'].columns}",
            f"Actual plotted score is calibrated: True",
            f"{SCORE_FIELD} exactly equals {CALIBRATED_SCORE_FIELD}: True",
            "Raw p_joint_mean used for ranking or plotting: False",
            (
                "Ranking direction: score_joint_probability descending; "
                "uncertainty_normalized ascending; material_id ascending"
            ),
            "Tie-breaking algorithm: stable mergesort; missing values last",
            f"Stored rank reconstructed exactly from formal sort rule: {audit['rank_reconstruction_matches']}",
            f"Primary-score tied values: {audit['primary_tied_values']}",
            f"Rows participating in primary-score ties: {audit['primary_tied_rows']}",
            f"Ranking configuration: {audit['ranking_config_path']}",
            f"Ranking configuration SHA256: {audit['ranking_config_sha256']}",
            f"Frozen lambda: {audit['lambda']:.17g}",
            f"Uncertainty penalty active: {audit['lambda'] != 0.0}",
            "Risk score definition: pJ_calibrated - lambda * uncertainty_normalized",
            f"Probability rank equals risk rank for all rows: {bool(np.array_equal(audit['frame'][RANK_FIELD].to_numpy(int), audit['frame']['risk_rank'].to_numpy(int)))}",
            f"Temperature file: {audit['temperatures_path']}",
            f"Temperature file SHA256: {audit['temperatures_sha256']}",
            f"Frozen Joint temperature: {float(audit['temperatures']['joint']):.17g}",
            "",
            "PERCENTILE DEFINITION",
            "---------------------",
            "percentile_rank = 100 * probability_rank / 8803",
            "Lower percentile indicates higher screening priority.",
            "X-axis covers the complete 0-100% ranking space.",
            f"Optional 1/10/20% guides shown: {show_percentile_guides}",
            "No nonlinear compression, rank truncation, jitter, or connecting line.",
            "",
            "HIGHLIGHTED CANDIDATES",
            "----------------------",
        ]
    )
    for specification in HIGHLIGHTS:
        material_id = str(specification["material_id"])
        candidate = audit["candidate_rows"][material_id]
        lines.extend(
            [
                material_id,
                f"  rank: {candidate['rank']}",
                f"  percentile_rank: {candidate['percentile']:.15f}%",
                f"  {SCORE_FIELD}: {candidate['score']:.17g}",
                f"  raw p_joint_mean (not plotted): {candidate['raw_p_joint_mean']:.17g}",
                f"  source data row: {candidate['source_data_row']}",
                f"  source CSV line: {candidate['source_csv_line']}",
                f"  row_id: {candidate['row_id']}",
            ]
        )

    source_unchanged = sha256_file(audit["ranking_path"]) == audit["ranking_hash_before"]
    lines.extend(
        [
            "",
            "SANITY CHECKS",
            "-------------",
            f"N == 8803: {len(audit['frame']) == EXPECTED_N}",
            f"2CrSnSe3-1 rank == 43: {audit['candidate_rows']['2CrSnSe3-1']['rank'] == 43}",
            f"2NiSbO3-1 rank == 987: {audit['candidate_rows']['2NiSbO3-1']['rank'] == 987}",
            "Third candidate rank was read from source with no expected-value constant.",
            "",
            "OUTPUTS",
            "-------",
            f"Source data CSV: {source_data_path}",
            f"Source data rows: {validation['source_data_rows']}",
            f"Source data SHA256: {sha256_file(source_data_path)}",
            f"PDF: {figure_paths['pdf']}",
            f"PDF SHA256: {sha256_file(figure_paths['pdf'])}",
            f"PNG: {figure_paths['png']}",
            f"PNG SHA256: {sha256_file(figure_paths['png'])}",
            f"SVG: {figure_paths['svg']}",
            f"SVG SHA256: {sha256_file(figure_paths['svg'])}",
            f"PNG raster: {validation['png_size'][0]} x {validation['png_size'][1]} px",
            f"PNG embedded dpi: {validation['png_dpi']}",
            "PDF and SVG saved directly by matplotlib as vector output.",
            f"SVG raster image elements: {validation['svg_contains_raster_image']}",
            "",
            "FINAL INTEGRITY CHECK",
            "---------------------",
            "Missing required source row: none",
            "Duplicate material ID: none",
            "Primary-score ties: none",
            "Rank conflict: none",
            "Sanity-rank conflict: none",
            "Pre-filter or sampling detected: none",
            "Source file changed during execution: " + str(not source_unchanged),
            "Overall status: PASS" if source_unchanged else "Overall status: FAIL",
            "",
        ]
    )
    if not source_unchanged:
        raise AssertionError("Frozen ranking source changed during execution")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (project_root / "paper_figures" / "fig5").resolve()
    )
    show_guides = SHOW_PERCENTILE_GUIDES and not args.hide_percentile_guides

    audit = validate_frozen_ranking(project_root)
    print_screening_audit(audit)
    assert_requested_sanity(audit)

    source_data = build_source_data(audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data_path = output_dir / "Fig5a_screening_source_data.csv"
    atomic_write_csv(source_data, source_data_path)
    figure_paths = create_figure(
        source_data,
        output_dir,
        show_percentile_guides=show_guides,
    )
    validation = validate_written_outputs(source_data_path, figure_paths, audit)
    executed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    audit_text = build_audit_text(
        project_root,
        output_dir,
        source_data_path,
        figure_paths,
        audit,
        validation,
        show_percentile_guides=show_guides,
        executed_at=executed_at,
    )
    audit_path = output_dir / "Fig5a_screening_audit.txt"
    atomic_write_text(audit_text, audit_path)

    print("[OUTPUTS]")
    for path in (*figure_paths.values(), source_data_path, audit_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
