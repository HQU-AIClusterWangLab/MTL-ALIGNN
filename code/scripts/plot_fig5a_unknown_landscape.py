#!/usr/bin/env python3
"""Generate the full-pool Thermo-Dynamic Unknown-candidate landscape.

The script reads the frozen M10 screening export, validates that it contains the
complete 8,803-row Dynamic-Unknown pool, and maps the final calibrated Thermo,
Dynamic, and Joint probabilities without reranking or pre-filtering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_N = 8_803
EXPECTED_SOURCE_SHA256 = (
    "c09b9aa4eb76b4e70828403a294d7cecf63f4077f57dcbb9a60cddb23634e98f"
)
AXIS_THERMO_FIELD = "p_thermo_calibrated"
AXIS_DYNAMIC_FIELD = "p_dynamic_calibrated"
JOINT_FIELD = "p_joint_calibrated"
RANK_FIELD = "probability_rank"
HIGHLIGHT_IDS = (
    "2CrSnSe3-1",
    "2NiSbO3-1",
    "1BrI2Ir3Te3-1",
)
SANITY_RANKS = {
    "2CrSnSe3-1": 43,
    "2NiSbO3-1": 987,
}


@dataclass(frozen=True)
class LocatedSource:
    path: Path
    sha256: str
    result_candidates: tuple[Path, ...]
    identical_mirrors: tuple[Path, ...]


@dataclass(frozen=True)
class ValidatedPool:
    frame: pd.DataFrame
    candidates: pd.DataFrame
    source: LocatedSource
    unknown_pool_path: Path
    unknown_pool_sha256: str
    supervised_path: Path
    supervised_sha256: str
    supervised_overlap_count: int
    summary_path: Path
    summary_sha256: str
    completed_path: Path
    completed_sha256: str
    temperatures_path: Path
    temperatures_sha256: str
    temperatures: dict[str, float]
    ranking_configuration_path: Path
    ranking_configuration_sha256: str
    ranking_lambda: float
    exact_score_ties: int
    thermo_eligible_count: int
    screening_eligible_count: int


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    inferred_root = script_path.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=inferred_root,
        help="Project root (default: inferred from scripts/ location).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <project-root>/paper_figures/fig5).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen metadata file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def candidate_csv_paths(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")
    tokens = ("unknown", "screen", "candidate", "all_predictions")
    paths: list[Path] = []
    for path in results_root.rglob("*.csv"):
        lower = str(path).lower()
        if any(token in lower for token in tokens):
            paths.append(path.resolve())
    return sorted(set(paths), key=lambda value: str(value).lower())


def is_full_unknown_export(path: Path) -> bool:
    required = {
        "material_id",
        AXIS_THERMO_FIELD,
        AXIS_DYNAMIC_FIELD,
        JOINT_FIELD,
        RANK_FIELD,
        "dyn_stab_raw",
    }
    try:
        header = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    if not required.issubset(header):
        return False
    probe = pd.read_csv(path, usecols=["material_id", "dyn_stab_raw"])
    return (
        len(probe) == EXPECTED_N
        and probe["material_id"].notna().all()
        and set(probe["dyn_stab_raw"].astype(str)) == {"Unknown"}
    )


def locate_unknown_screening_file(project_root: Path) -> LocatedSource:
    """Locate the one formal full-pool export and reject conflicting copies."""
    project_root = project_root.resolve()
    result_candidates = tuple(
        path
        for path in candidate_csv_paths(project_root / "results")
        if is_full_unknown_export(path)
    )
    if not result_candidates:
        raise FileNotFoundError(
            "No formal 8,803-row Unknown prediction CSV with the required "
            "Thermo/Dynamic/Joint/rank fields was found under results/."
        )

    hashes = {path: sha256_file(path) for path in result_candidates}
    unique_hashes = set(hashes.values())
    if len(unique_hashes) != 1:
        detail = "\n".join(f"  {path}: {value}" for path, value in hashes.items())
        raise RuntimeError(f"Conflicting formal full-pool result files:\n{detail}")

    preferred = [
        path
        for path in result_candidates
        if "results" in [part.lower() for part in path.parts]
        and "main" in [part.lower() for part in path.parts]
        and "m10_unknown_screening" in [part.lower() for part in path.parts]
    ]
    if len(preferred) == 1:
        source = preferred[0]
    elif len(result_candidates) == 1:
        source = result_candidates[0]
    else:
        detail = "\n".join(f"  {path}" for path in result_candidates)
        raise RuntimeError(
            "Multiple equally plausible formal full-pool result files were found "
            f"and no unique canonical M10 path could be selected:\n{detail}"
        )

    source_hash = hashes[source]
    if source_hash.lower() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "The canonical full-pool CSV does not match the frozen source hash. "
            f"Expected {EXPECTED_SOURCE_SHA256}; found {source_hash}: {source}"
        )

    mirrors: list[Path] = []
    conflicts: list[tuple[Path, str]] = []
    for path in project_root.rglob(source.name):
        resolved = path.resolve()
        if resolved == source:
            continue
        candidate_hash = sha256_file(resolved)
        if candidate_hash == source_hash:
            mirrors.append(resolved)
        elif is_full_unknown_export(resolved):
            conflicts.append((resolved, candidate_hash))
    if conflicts:
        detail = "\n".join(f"  {path}: {value}" for path, value in conflicts)
        raise RuntimeError(
            "Conflicting copies of the 8,803-row Unknown full-pool export were "
            f"detected:\n{detail}"
        )

    return LocatedSource(
        path=source,
        sha256=source_hash,
        result_candidates=result_candidates,
        identical_mirrors=tuple(sorted(mirrors, key=lambda value: str(value).lower())),
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns in {source}: {missing}")


def finite_probability(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} contains missing or non-finite values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{column} contains values outside [0, 1].")
    return values


def validate_candidate_pool(project_root: Path, source: LocatedSource) -> ValidatedPool:
    frame = pd.read_csv(source.path)
    required = {
        "material_id",
        "dyn_stab_raw",
        "ranking_scope",
        "p_thermo_mean",
        "p_dynamic_mean",
        "p_joint_mean",
        AXIS_THERMO_FIELD,
        AXIS_DYNAMIC_FIELD,
        JOINT_FIELD,
        "pJ",
        "pJ_calibrated",
        "score_joint_probability",
        "uncertainty_normalized",
        "thermo_eligible",
        "screening_eligible",
        RANK_FIELD,
        "risk_rank",
    }
    require_columns(frame, required, source.path)

    if len(frame) != EXPECTED_N:
        raise AssertionError(f"Expected n_unknown={EXPECTED_N}; found {len(frame)}.")
    if frame["material_id"].isna().any():
        raise AssertionError("material_id contains missing values.")
    frame["material_id"] = frame["material_id"].astype(str)
    duplicate_count = int(frame["material_id"].duplicated(keep=False).sum())
    if duplicate_count:
        raise AssertionError(f"Found {duplicate_count} rows with duplicate material_id.")
    if set(frame["dyn_stab_raw"].astype(str)) != {"Unknown"}:
        raise AssertionError("The formal source contains non-Unknown dyn_stab_raw rows.")
    if set(frame["ranking_scope"].astype(str)) != {"all_unknown"}:
        raise AssertionError("ranking_scope is not uniformly 'all_unknown'.")

    for column in (
        "p_thermo_mean",
        "p_dynamic_mean",
        "p_joint_mean",
        AXIS_THERMO_FIELD,
        AXIS_DYNAMIC_FIELD,
        JOINT_FIELD,
        "pJ",
        "pJ_calibrated",
        "score_joint_probability",
    ):
        finite_probability(frame, column)

    if not np.array_equal(
        frame["p_joint_mean"].to_numpy(float), frame["pJ"].to_numpy(float)
    ):
        raise AssertionError("Frozen pJ is not identical to p_joint_mean.")
    for equivalent in ("pJ_calibrated", "score_joint_probability"):
        if not np.array_equal(
            frame[JOINT_FIELD].to_numpy(float), frame[equivalent].to_numpy(float)
        ):
            raise AssertionError(f"{JOINT_FIELD} is not identical to {equivalent}.")

    rank_numeric = pd.to_numeric(frame[RANK_FIELD], errors="coerce")
    if rank_numeric.isna().any():
        raise AssertionError(f"{RANK_FIELD} contains non-numeric or missing values.")
    rank_values = rank_numeric.to_numpy(float)
    if not np.equal(rank_values, np.floor(rank_values)).all():
        raise AssertionError(f"{RANK_FIELD} contains non-integer values.")
    rank_values_int = rank_values.astype(int)
    if not np.array_equal(np.sort(rank_values_int), np.arange(1, EXPECTED_N + 1)):
        raise AssertionError(f"{RANK_FIELD} is not a unique 1..{EXPECTED_N} permutation.")
    frame[RANK_FIELD] = rank_values_int

    formal_order = frame.sort_values(
        ["score_joint_probability", "uncertainty_normalized", "material_id"],
        ascending=[False, True, True],
        kind="mergesort",
        na_position="last",
    ).index
    reconstructed = pd.Series(
        np.arange(1, EXPECTED_N + 1, dtype=int), index=formal_order
    ).sort_index()
    if not np.array_equal(reconstructed.to_numpy(), frame[RANK_FIELD].to_numpy()):
        raise AssertionError("Stored probability_rank does not reproduce the formal sort.")

    exact_score_ties = int(frame["score_joint_probability"].duplicated(keep=False).sum())

    candidates = frame.loc[frame["material_id"].isin(HIGHLIGHT_IDS)].copy()
    counts = candidates["material_id"].value_counts().to_dict()
    if set(counts) != set(HIGHLIGHT_IDS) or any(counts.get(item) != 1 for item in HIGHLIGHT_IDS):
        raise AssertionError(
            "Each requested highlight must occur exactly once; observed counts: "
            f"{counts}"
        )
    candidates = candidates.set_index("material_id").loc[list(HIGHLIGHT_IDS)].reset_index()
    for material_id, expected_rank in SANITY_RANKS.items():
        actual_rank = int(
            candidates.loc[candidates["material_id"] == material_id, RANK_FIELD].iloc[0]
        )
        if actual_rank != expected_rank:
            raise AssertionError(
                f"Formal rank conflict for {material_id}: expected sanity rank "
                f"{expected_rank}, found {actual_rank}."
            )

    unknown_pool_path = (project_root / "data" / "processed" / "unknown_pool.csv").resolve()
    if not unknown_pool_path.is_file():
        raise FileNotFoundError(f"Frozen Unknown input pool is missing: {unknown_pool_path}")
    unknown_pool = pd.read_csv(unknown_pool_path, usecols=["material_id"])
    if len(unknown_pool) != EXPECTED_N or unknown_pool["material_id"].duplicated().any():
        raise AssertionError("Frozen unknown_pool.csv is not 8,803 unique material IDs.")
    if set(unknown_pool["material_id"].astype(str)) != set(frame["material_id"]):
        raise AssertionError("Formal prediction IDs do not exactly match unknown_pool.csv.")

    supervised_path = (project_root / "data" / "processed" / "supervised.csv").resolve()
    if not supervised_path.is_file():
        raise FileNotFoundError(f"Frozen supervised pool is missing: {supervised_path}")
    supervised = pd.read_csv(supervised_path, usecols=["material_id"])
    supervised_ids = set(supervised["material_id"].dropna().astype(str))
    supervised_overlap_count = len(supervised_ids.intersection(set(frame["material_id"])))
    if supervised_overlap_count:
        raise AssertionError(
            "Unknown screening export overlaps the supervised train/validation/test pool "
            f"by {supervised_overlap_count} material IDs."
        )

    summary_path = source.path.parent / "screening_summary.json"
    completed_path = source.path.parent / "COMPLETED.json"
    summary = read_json(summary_path)
    completed = read_json(completed_path)
    for label, payload in (("screening_summary", summary), ("COMPLETED", completed)):
        if payload.get("status") != "completed":
            raise AssertionError(f"{label} status is not completed.")
        if int(payload.get("unknown_count", -1)) != EXPECTED_N:
            raise AssertionError(f"{label} unknown_count is not {EXPECTED_N}.")
        if int(payload.get("ranking_pool_count", -1)) != EXPECTED_N:
            raise AssertionError(f"{label} ranking_pool_count is not {EXPECTED_N}.")
        if payload.get("ranking_scope") != "all_unknown":
            raise AssertionError(f"{label} ranking_scope is not all_unknown.")
        if payload.get("pJ_calibrated_alias") != JOINT_FIELD:
            raise AssertionError(f"{label} pJ calibrated alias is unexpected.")

    temperatures_path = (
        project_root
        / "results"
        / "main"
        / "M06_full_mtl"
        / "ensemble"
        / "temperatures.json"
    ).resolve()
    temperatures_payload = read_json(temperatures_path)
    temperatures = {
        task: float(temperatures_payload[task]) for task in ("thermo", "dynamic", "joint")
    }
    if not all(np.isfinite(value) and value > 0.0 for value in temperatures.values()):
        raise AssertionError("Frozen task temperatures are not finite positive values.")

    ranking_configuration_path = (
        project_root
        / "results"
        / "main"
        / "M06_full_mtl"
        / "ranking"
        / "ranking_configuration.json"
    ).resolve()
    ranking_configuration = read_json(ranking_configuration_path)
    ranking_lambda = float(ranking_configuration["lambda"])
    if ranking_lambda != 0.0 or float(summary["lambda"]) != 0.0:
        raise AssertionError("The frozen formal ranking lambda is not zero as expected.")
    if not np.array_equal(
        frame["probability_rank"].to_numpy(int), frame["risk_rank"].to_numpy(int)
    ):
        raise AssertionError("lambda=0 but risk_rank differs from probability_rank.")

    thermo_eligible_count = int(frame["thermo_eligible"].astype(bool).sum())
    screening_eligible_count = int(frame["screening_eligible"].astype(bool).sum())
    if thermo_eligible_count != int(summary["thermo_eligible_count"]):
        raise AssertionError("thermo_eligible count conflicts with screening_summary.json.")

    return ValidatedPool(
        frame=frame,
        candidates=candidates,
        source=source,
        unknown_pool_path=unknown_pool_path,
        unknown_pool_sha256=sha256_file(unknown_pool_path),
        supervised_path=supervised_path,
        supervised_sha256=sha256_file(supervised_path),
        supervised_overlap_count=supervised_overlap_count,
        summary_path=summary_path,
        summary_sha256=sha256_file(summary_path),
        completed_path=completed_path,
        completed_sha256=sha256_file(completed_path),
        temperatures_path=temperatures_path,
        temperatures_sha256=sha256_file(temperatures_path),
        temperatures=temperatures,
        ranking_configuration_path=ranking_configuration_path,
        ranking_configuration_sha256=sha256_file(ranking_configuration_path),
        ranking_lambda=ranking_lambda,
        exact_score_ties=exact_score_ties,
        thermo_eligible_count=thermo_eligible_count,
        screening_eligible_count=screening_eligible_count,
    )


def compute_top_percent(rank: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(rank, dtype=float)
    return 100.0 * values / float(EXPECTED_N)


def build_label_text(material_id: str, rank: int, top_percent: float) -> str:
    return f"{material_id}\nRank {rank} | Top {top_percent:.1f}%"


def build_source_data(audit: ValidatedPool) -> pd.DataFrame:
    frame = audit.frame
    output = pd.DataFrame(
        {
            "material_id": frame["material_id"].astype(str),
            "p_thermo": frame[AXIS_THERMO_FIELD].astype(float),
            "p_dynamic": frame[AXIS_DYNAMIC_FIELD].astype(float),
            "p_joint": frame[JOINT_FIELD].astype(float),
            "rank": frame[RANK_FIELD].astype(int),
            "top_percent": compute_top_percent(frame[RANK_FIELD]),
            "is_highlighted": frame["material_id"].isin(HIGHLIGHT_IDS),
            "label_text": "",
            "source_file": str(audit.source.path),
            "source_sha256": audit.source.sha256,
            "p_thermo_source_field": AXIS_THERMO_FIELD,
            "p_dynamic_source_field": AXIS_DYNAMIC_FIELD,
            "p_joint_source_field": JOINT_FIELD,
            "rank_source_field": RANK_FIELD,
            "calibrated_axes": True,
            "calibrated_joint": True,
            "p_thermo_mean": frame["p_thermo_mean"].astype(float),
            "p_dynamic_mean": frame["p_dynamic_mean"].astype(float),
            "p_joint_mean": frame["p_joint_mean"].astype(float),
            "p_thermo_calibrated": frame["p_thermo_calibrated"].astype(float),
            "p_dynamic_calibrated": frame["p_dynamic_calibrated"].astype(float),
            "p_joint_calibrated": frame["p_joint_calibrated"].astype(float),
            "probability_rank": frame["probability_rank"].astype(int),
            "risk_rank": frame["risk_rank"].astype(int),
            "ranking_scope": frame["ranking_scope"].astype(str),
            "thermo_eligible": frame["thermo_eligible"].astype(bool),
            "screening_eligible": frame["screening_eligible"].astype(bool),
        }
    )

    optional = (
        "p_thermo_std",
        "p_dynamic_std",
        "p_joint_std",
        "thermo_predictive_entropy",
        "dynamic_predictive_entropy",
        "joint_predictive_entropy",
        "thermo_mutual_information",
        "dynamic_mutual_information",
        "joint_mutual_information",
        "uncertainty_normalized",
        "score_joint_probability",
        "score_risk_aware",
        "ehull",
    )
    for column in optional:
        if column in frame.columns:
            output[column] = frame[column]

    highlighted = output["is_highlighted"]
    output.loc[highlighted, "label_text"] = [
        build_label_text(material_id, int(rank), float(percent))
        for material_id, rank, percent in output.loc[
            highlighted, ["material_id", "rank", "top_percent"]
        ].itertuples(index=False, name=None)
    ]
    return output.sort_values("rank", kind="mergesort").reset_index(drop=True)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_plot(source_data: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    fig, axis = plt.subplots(figsize=(5.0, 4.2), facecolor="white")
    axis.set_facecolor("white")

    axis.axvline(
        0.5, color="#D8DEE2", linewidth=0.65, linestyle=(0, (3, 3)), zorder=0
    )
    axis.axhline(
        0.5, color="#D8DEE2", linewidth=0.65, linestyle=(0, (3, 3)), zorder=0
    )

    background = axis.scatter(
        source_data["p_thermo"].to_numpy(float),
        source_data["p_dynamic"].to_numpy(float),
        s=8.0,
        marker="o",
        color="#93A3AC",
        alpha=0.18,
        linewidths=0.0,
        edgecolors="none",
        rasterized=False,
        zorder=1,
        label=f"Unknown candidates (N = {EXPECTED_N:,})",
    )
    if len(background.get_offsets()) != EXPECTED_N:
        raise AssertionError("The plotted background collection is not 8,803 points.")

    styles: dict[str, dict[str, Any]] = {
        "2CrSnSe3-1": {
            "marker": "o",
            "size": 88,
            "face": "#2D5F7A",
            "edge": "white",
            "line": "#2D5F7A",
            "offset": (-55, -1),
            "ha": "right",
            "va": "center",
        },
        "2NiSbO3-1": {
            "marker": "s",
            "size": 100,
            "face": "#A56643",
            "edge": "white",
            "line": "#A56643",
            "offset": (-55, -30),
            "ha": "right",
            "va": "center",
        },
        "1BrI2Ir3Te3-1": {
            "marker": "^",
            "size": 130,
            "face": "none",
            "edge": "#527864",
            "line": "#527864",
            "offset": (-55, -34),
            "ha": "right",
            "va": "center",
        },
    }

    for material_id in HIGHLIGHT_IDS:
        row = source_data.loc[source_data["material_id"] == material_id].iloc[0]
        style = styles[material_id]
        axis.scatter(
            [float(row["p_thermo"])],
            [float(row["p_dynamic"])],
            s=float(style["size"]),
            marker=str(style["marker"]),
            facecolor=str(style["face"]),
            edgecolor=str(style["edge"]),
            linewidth=1.5 if material_id == "1BrI2Ir3Te3-1" else 0.95,
            zorder=7 if material_id == "1BrI2Ir3Te3-1" else 6,
            clip_on=True,
        )
        axis.annotate(
            str(row["label_text"]),
            xy=(float(row["p_thermo"]), float(row["p_dynamic"])),
            xytext=style["offset"],
            textcoords="offset points",
            ha=str(style["ha"]),
            va=str(style["va"]),
            fontsize=7.5,
            color="#263238",
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": str(style["line"]),
                "linewidth": 0.75,
                "shrinkA": 2.0,
                "shrinkB": 5.0,
            },
            annotation_clip=False,
            zorder=6,
        )

    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(0.0, 1.02)
    ticks = np.linspace(0.0, 1.0, 6)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xlabel("Predicted thermo probability", labelpad=5.5)
    axis.set_ylabel("Predicted dynamic probability", labelpad=6.0)

    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#4D555A")
        axis.spines[spine].set_linewidth(0.8)
    axis.tick_params(colors="#4D555A", pad=2.5)

    axis.set_aspect("equal", adjustable="box")
    axis.text(
        -0.135,
        1.0,
        "a",
        transform=axis.transAxes,
        fontsize=11.0,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
    )

    fig.subplots_adjust(left=0.155, right=0.982, bottom=0.145, top=0.965)
    return fig


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
    os.replace(temporary, path)


def atomic_figure(fig: plt.Figure, path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, format=path.suffix.lstrip("."), **kwargs)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def validate_written_outputs(
    source_data: pd.DataFrame,
    csv_path: Path,
    pdf_path: Path,
    png_path: Path,
    svg_path: Path,
) -> None:
    reread = pd.read_csv(csv_path)
    required = {
        "material_id",
        "p_thermo",
        "p_dynamic",
        "p_joint",
        "rank",
        "top_percent",
        "is_highlighted",
        "label_text",
        "source_file",
    }
    require_columns(reread, required, csv_path)
    if len(reread) != EXPECTED_N or reread["material_id"].nunique() != EXPECTED_N:
        raise AssertionError("Written source CSV is not 8,803 unique rows.")
    if int(reread["is_highlighted"].astype(bool).sum()) != len(HIGHLIGHT_IDS):
        raise AssertionError("Written source CSV does not contain exactly 3 highlights.")
    if not np.array_equal(reread["rank"].to_numpy(int), np.arange(1, EXPECTED_N + 1)):
        raise AssertionError("Written source CSV is not in formal rank order 1..8,803.")
    expected_percent = compute_top_percent(reread["rank"])
    if not np.allclose(
        reread["top_percent"].to_numpy(float), expected_percent, rtol=0.0, atol=1e-12
    ):
        raise AssertionError("Written top_percent values do not match 100*rank/8803.")
    for column in ("p_thermo", "p_dynamic", "p_joint"):
        if not np.allclose(
            reread[column].to_numpy(float),
            source_data[column].to_numpy(float),
            rtol=0.0,
            atol=5e-16,
        ):
            raise AssertionError(
                f"Written {column} values changed beyond floating-point parser "
                "precision during CSV export."
            )

    for path in (pdf_path, png_path, svg_path, csv_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Expected non-empty output is missing: {path}")
    if not pdf_path.read_bytes().startswith(b"%PDF"):
        raise AssertionError("PDF output does not have a valid PDF header.")
    svg_text = svg_path.read_text(encoding="utf-8")
    if "<image" in svg_text.lower():
        raise AssertionError("SVG contains an embedded raster image.")
    for material_id in HIGHLIGHT_IDS:
        if material_id not in svg_text:
            raise AssertionError(f"SVG is missing annotation text for {material_id}.")

    with Image.open(png_path) as image:
        width, height = image.size
        dpi = image.info.get("dpi", (0.0, 0.0))
    if width < 2_990 or height < 2_510:
        raise AssertionError(f"PNG dimensions are unexpectedly small: {width}x{height}")
    if not (599.0 <= float(dpi[0]) <= 601.0 and 599.0 <= float(dpi[1]) <= 601.0):
        raise AssertionError(f"PNG DPI is not 600 within tolerance: {dpi}")


def candidate_rows_for_report(source_data: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for material_id in HIGHLIGHT_IDS:
        row = source_data.loc[source_data["material_id"] == material_id].iloc[0]
        records.append(
            {
                "material_id": material_id,
                "pT": float(row["p_thermo"]),
                "pD": float(row["p_dynamic"]),
                "pJ": float(row["p_joint"]),
                "rank": int(row["rank"]),
                "top_percent": float(row["top_percent"]),
                "label_text": str(row["label_text"]),
            }
        )
    return records


def make_audit_text(
    validated: ValidatedPool,
    source_data: pd.DataFrame,
    script_path: Path,
    outputs: dict[str, Path],
    execution_time: str,
) -> str:
    candidates = candidate_rows_for_report(source_data)
    output_lines: list[str] = []
    for label, path in outputs.items():
        output_lines.extend(
            [
                f"  {label}: {path}",
                f"    bytes: {path.stat().st_size}",
                f"    SHA256: {sha256_file(path)}",
            ]
        )

    mirror_lines = (
        [f"  {path} (SHA256 identical; ignored)" for path in validated.source.identical_mirrors]
        or ["  None detected"]
    )
    candidate_lines: list[str] = []
    for record in candidates:
        source_index = int(
            validated.frame.index[
                validated.frame["material_id"] == record["material_id"]
            ][0]
        )
        candidate_lines.extend(
            [
                f"{record['material_id']}:",
                f"  p_T ({AXIS_THERMO_FIELD}): {record['pT']:.17g}",
                f"  p_D ({AXIS_DYNAMIC_FIELD}): {record['pD']:.17g}",
                f"  p_J ({JOINT_FIELD}): {record['pJ']:.17g}",
                f"  rank ({RANK_FIELD}): {record['rank']}",
                f"  top_percent: {record['top_percent']:.17g}",
                f"  display label: {record['label_text'].replace(chr(10), ' | ')}",
                f"  source CSV file line (header=1): {source_index + 2}",
                "",
            ]
        )

    lines = [
        "FIG5A UNKNOWN CANDIDATE LANDSCAPE — DATA AND PROVENANCE AUDIT",
        "=" * 72,
        "",
        "Formal source",
        f"  source file: {validated.source.path}",
        f"  source bytes: {validated.source.path.stat().st_size}",
        f"  source SHA256: {validated.source.sha256}",
        f"  valid full-pool result candidates under results/: {len(validated.source.result_candidates)}",
        "  source selection: unique formal full-pool export under results/main/M10_unknown_screening",
        "  duplicate result-value conflict: none",
        "",
        "Identical mirrors",
        *mirror_lines,
        "",
        "Frozen input/metadata cross-checks",
        f"  unknown pool: {validated.unknown_pool_path}",
        f"  unknown pool SHA256: {validated.unknown_pool_sha256}",
        "  prediction IDs exactly equal unknown_pool.csv IDs: True",
        f"  supervised pool: {validated.supervised_path}",
        f"  supervised pool SHA256: {validated.supervised_sha256}",
        f"  Unknown/supervised material-ID overlap: {validated.supervised_overlap_count}",
        "  supervised train/validation/test rows mixed into figure: False",
        f"  screening summary: {validated.summary_path}",
        f"  screening summary SHA256: {validated.summary_sha256}",
        f"  completion marker: {validated.completed_path}",
        f"  completion marker SHA256: {validated.completed_sha256}",
        "  formal status completed: True",
        f"  frozen temperatures: {validated.temperatures_path}",
        f"  frozen temperatures SHA256: {validated.temperatures_sha256}",
        f"  Thermo temperature: {validated.temperatures['thermo']:.17g}",
        f"  Dynamic temperature: {validated.temperatures['dynamic']:.17g}",
        f"  Joint temperature: {validated.temperatures['joint']:.17g}",
        f"  ranking configuration: {validated.ranking_configuration_path}",
        f"  ranking configuration SHA256: {validated.ranking_configuration_sha256}",
        f"  frozen ranking lambda: {validated.ranking_lambda:.17g}",
        "",
        "Candidate pool validation",
        f"  N: {len(validated.frame)}",
        f"  expected N: {EXPECTED_N}",
        "  n_unknown == 8803: True",
        f"  unique material_id: {validated.frame['material_id'].nunique()}",
        "  duplicate material_id rows: 0",
        "  dyn_stab_raw values: Unknown only",
        "  ranking_scope values: all_unknown only",
        f"  thermo_eligible diagnostic count: {validated.thermo_eligible_count}",
        f"  screening_eligible diagnostic count: {validated.screening_eligible_count}",
        "  thermo-eligible filter applied: False",
        "  Ehull filter applied: False",
        "  Top-K filter applied: False",
        "  random sampling applied: False",
        "  any pre-filter applied: False",
        "  all 8,803 rows plotted: True",
        "",
        "Field mapping and probability definition",
        f"  p_T source field: {AXIS_THERMO_FIELD}",
        f"  p_D source field: {AXIS_DYNAMIC_FIELD}",
        f"  p_J source field: {JOINT_FIELD}",
        f"  rank source field: {RANK_FIELD}",
        "  rank read directly from formal source: True",
        "  reranking performed for figure data: False",
        "  calibrated fields present: True",
        "  axes use calibrated probabilities: True",
        "  Joint value uses calibrated probability: True",
        "  axis probability definition: mean(sigmoid(member_logit / frozen task temperature))",
        "  rationale: the formal screening export and its calibrated logical diagnostics use the frozen calibrated task probabilities; the formal full-pool rank uses calibrated p_J",
        "  raw mean fields retained in source CSV: True",
        "  p_joint_calibrated == pJ_calibrated == score_joint_probability: True",
        "  p_joint_mean == pJ: True",
        "",
        "Ranking validation",
        "  ranking direction: score_joint_probability descending; uncertainty_normalized ascending; material_id ascending",
        "  stable sort: mergesort; missing values last",
        f"  rank is unique continuous 1..{EXPECTED_N}: True",
        "  stored rank exactly reproduced by formal sort: True",
        f"  exact primary-score tied rows: {validated.exact_score_ties}",
        "  risk rank used for labels/plot: False",
        "  lambda=0 and risk_rank == probability_rank for all rows: True",
        "  top_percent formula: 100 * probability_rank / 8803",
        "  display precision: one decimal place (full precision retained in source CSV)",
        "",
        "Highlighted candidates",
        *candidate_lines,
        "Sanity checks",
        "  2CrSnSe3-1 rank == 43: PASS",
        "  2NiSbO3-1 rank == 987: PASS",
        "  1BrI2Ir3Te3-1 rank read dynamically from formal source: PASS",
        "",
        "Plot encoding",
        "  x-axis: Predicted thermo probability [0, 1.02]",
        "  y-axis: Predicted dynamic probability [0, 1.02]",
        "  background: all 8,803 rows as individual scatter markers",
        "  sampling/jitter/line/KDE/hexbin/contour: none",
        "  reference guides: x=0.5 and y=0.5 only",
        "  highlighted marker coordinates: exact source values",
        "  annotation labels: dynamically built from stored rank and computed top_percent",
        "",
        "Execution",
        f"  execution time: {execution_time}",
        f"  script path: {script_path}",
        f"  script SHA256: {sha256_file(script_path)}",
        f"  Python: {platform.python_version()}",
        f"  executable: {sys.executable}",
        f"  NumPy: {np.__version__}",
        f"  pandas: {pd.__version__}",
        f"  Matplotlib: {matplotlib.__version__}",
        "",
        "Outputs",
        *output_lines,
        "  PNG requested DPI: 600",
        "  PDF/SVG generated directly by Matplotlib as vector outputs",
        "  SVG embedded raster image: False",
        "",
        "Missing/conflict summary",
        "  missing required values: none",
        "  duplicate material_id: none",
        "  formal source conflicts: none",
        "  rank sanity conflicts: none",
        "  result: PASS",
        "",
    ]
    return "\n".join(lines)


def print_terminal_audit(source_data: pd.DataFrame, source: Path) -> None:
    records = candidate_rows_for_report(source_data)
    print("[FIG5A UNKNOWN LANDSCAPE AUDIT]")
    print("")
    print("Candidate pool:")
    print(f"  N = {len(source_data)}")
    print(f"  source file = {source}")
    print(f"  thermo field = {AXIS_THERMO_FIELD}")
    print(f"  dynamic field = {AXIS_DYNAMIC_FIELD}")
    print(f"  joint field = {JOINT_FIELD}")
    print(f"  rank field = {RANK_FIELD}")
    print("  calibrated axes = True")
    print("  pre-filter applied = False")
    for record in records:
        print("")
        print(f"{record['material_id']}:")
        print(f"  pT = {record['pT']:.16f}")
        print(f"  pD = {record['pD']:.16f}")
        print(f"  pJ = {record['pJ']:.16f}")
        print(f"  rank = {record['rank']}")
        print(f"  top_percent = {record['top_percent']:.16f}")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (project_root / "paper_figures" / "fig5").resolve()
    )
    script_path = Path(__file__).resolve()
    execution_time = datetime.now().astimezone().isoformat(timespec="seconds")

    # All failure-stop assertions are completed before any output is created.
    source = locate_unknown_screening_file(project_root)
    validated = validate_candidate_pool(project_root, source)
    source_data = build_source_data(validated)
    if len(source_data) != EXPECTED_N:
        raise AssertionError(f"assert n_unknown == {EXPECTED_N} failed")
    print_terminal_audit(source_data, source.path)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "PDF": output_dir / "Fig5a_unknown_candidate_landscape.pdf",
        "PNG": output_dir / "Fig5a_unknown_candidate_landscape.png",
        "SVG": output_dir / "Fig5a_unknown_candidate_landscape.svg",
        "source CSV": output_dir / "Fig5a_unknown_candidate_source_data.csv",
    }
    audit_path = output_dir / "Fig5a_unknown_candidate_audit.txt"

    source_hash_before = sha256_file(validated.source.path)
    atomic_csv(source_data, paths["source CSV"])
    fig = make_plot(source_data)
    try:
        atomic_figure(fig, paths["PDF"])
        atomic_figure(fig, paths["SVG"])
        atomic_figure(fig, paths["PNG"], dpi=600)
    finally:
        plt.close(fig)

    validate_written_outputs(
        source_data,
        paths["source CSV"],
        paths["PDF"],
        paths["PNG"],
        paths["SVG"],
    )
    source_hash_after = sha256_file(validated.source.path)
    if source_hash_before != source_hash_after:
        raise AssertionError("The formal source file changed while the plot was generated.")

    audit_text = make_audit_text(
        validated=validated,
        source_data=source_data,
        script_path=script_path,
        outputs=paths,
        execution_time=execution_time,
    )
    atomic_text(audit_text, audit_path)

    print("")
    print("Outputs:")
    for label, path in paths.items():
        print(f"  {label} = {path}")
    print(f"  audit = {audit_path}")


if __name__ == "__main__":
    main()
