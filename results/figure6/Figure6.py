"""Generate manuscript Figure 6 and Table 1 from verified project results.

Panel (a) ranks all 8,803 unknown candidates by the final calibrated Joint
probability. Panels (b) and (c) embed the losslessly extracted literature
phonon panels for Cr2Sn2Se6 and Ni2Sb2O6. Panel (d) embeds the user's 600-dpi
self-calculated BrI2Ir3Te3 spectrum (``ck Phon2.tif``) and checks its conclusion
against the corresponding CASTEP ``.phonon`` result.

No scientific values or phonon curves are synthesized by this script.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(r"D:\ck\two")
PROJECT_OUTPUT_DIR = ROOT / "图6"
PAPER_OUTPUT_DIR = Path(r"C:\Users\WHQ\Desktop\论文初稿\图\结果图")
PAPER_CODE_DIR = PAPER_OUTPUT_DIR / "代码"
PHONON_IMAGE_DIR = PAPER_CODE_DIR / "声子谱图"

RANKING_FILE = (
    ROOT / "results" / "main" / "M10_unknown_screening" /
    "unknown_all_predictions.csv"
)
SCREENING_SUMMARY = (
    ROOT / "results" / "main" / "M10_unknown_screening" /
    "screening_summary.json"
)
RANKING_DIAGNOSTICS = (
    ROOT / "results" / "main" / "M10_unknown_screening" /
    "candidate_ranking_diagnostics.json"
)
RANKING_CONFIGURATION = (
    ROOT / "results" / "main" / "M06_full_mtl" / "ranking" /
    "ranking_configuration.json"
)

BR_IMAGE_CANDIDATES = [
    PHONON_IMAGE_DIR / "ck Phon2.tif",
    Path(r"C:\Users\WHQ\Desktop\ck Phon2.tif"),
]
CR_IMAGE = PHONON_IMAGE_DIR / "Cr2Sn2Se6" / "Cr2Sn2Se6_phonon.png"
NI_IMAGE = PHONON_IMAGE_DIR / "Ni2Sb2O6" / "Ni2Sb2O6_phonon.png"
CR_SOURCE = PHONON_IMAGE_DIR / "Cr2Sn2Se6" / "Cr2Sn2Se6_source.txt"
NI_SOURCE = PHONON_IMAGE_DIR / "Ni2Sb2O6" / "Ni2Sb2O6_source.txt"

BR_RAW_PHONON_CANDIDATES = [
    Path(
        r"D:\ck\two dft\two_Files\Documents\phonon\BrI2Ir3Te3_phonon_recheck\03_disp_001\PHON_PhonDisp.phonon"
    ),
    Path(
        r"D:\ck\two dft\two_Files\Documents\phonon\BrI2Ir3Te3_phonon_recheck\00_original_reference\PHON_PhonDisp.phonon"
    ),
]

PAPER_FIGURE_STEM = PAPER_OUTPUT_DIR / "图6"
PROJECT_FIGURE_STEM = PROJECT_OUTPUT_DIR / "Figure6"
TABLE_NAME = "Table1_representative_candidates.csv"
RANKING_DATA_NAME = "Figure6_candidate_ranking_data.csv"
AUDIT_NAME = "Figure6_audit.txt"
MANIFEST_NAME = "Figure6_manifest.json"

EXPECTED_N = 8803
DISPLAY_MATERIALS = ["BrI2Ir3Te3", "Cr2Sn2Se6", "Ni2Sb2O6"]


# -----------------------------------------------------------------------------
# Visual style
# -----------------------------------------------------------------------------
COLORS = {
    "deep_blue": "#104E8B",
    "light_blue": "#7DC6D3",
    "point": "#9BAFBE",
    "trend": "#547AA5",
    "terracotta": "#B96D5C",
    "teal": "#4F8A8B",
    "dark_gray": "#5F6368",
    "medium_gray": "#A7ABB1",
    "light_gray": "#D9DCE1",
    "text": "#262626",
    "spine": "#333333",
    "guide": "#E6E8EB",
    "white": "#FFFFFF",
}
HIGHLIGHT_COLORS = {
    "BrI2Ir3Te3": COLORS["terracotta"],
    "Cr2Sn2Se6": COLORS["deep_blue"],
    "Ni2Sb2O6": COLORS["teal"],
}


def apply_sci_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "stix",
            "figure.facecolor": COLORS["white"],
            "axes.facecolor": COLORS["white"],
            "savefig.facecolor": COLORS["white"],
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["spine"],
            "xtick.color": COLORS["spine"],
            "ytick.color": COLORS["spine"],
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color(COLORS["spine"])
        ax.spines[name].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", direction="out")


def panel_label(
    ax: mpl.axes.Axes,
    label: str,
    x: float = -0.10,
    y: float = 1.03,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )


# -----------------------------------------------------------------------------
# Data reading and validation
# -----------------------------------------------------------------------------
def first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No {label} file was found. Checked:\n" + "\n".join(map(str, paths))
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_formula(formula: str) -> tuple[tuple[str, int], ...]:
    formula = str(formula)
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    if not tokens or "".join(element + count for element, count in tokens) != formula:
        raise ValueError(f"Unsupported chemical formula: {formula!r}")
    composition: dict[str, int] = {}
    for element, count_text in tokens:
        composition[element] = composition.get(element, 0) + (
            int(count_text) if count_text else 1
        )
    return tuple(sorted(composition.items()))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_image(
    path: Path,
    blank_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Load an image and optionally remove publisher panel markers in white space.

    ``blank_boxes`` uses Pillow coordinates (left, top, right, bottom). The
    boxes below cover only the original ``(a)``/``(b)`` tags; no curve, axis,
    tick, label, or numerical content is touched.
    """
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB")).copy()
    for left, top, right, bottom in blank_boxes or []:
        array[top:bottom, left:right, :] = 255
    return array


def image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        raw_dpi = image.info.get("dpi", (0.0, 0.0))
        dpi = [float(raw_dpi[0]), float(raw_dpi[1])]
        return {
            "path": str(path),
            "format": image.format,
            "width_px": image.width,
            "height_px": image.height,
            "dpi": dpi,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }


def parse_castep_phonon(path: Path) -> dict[str, Any]:
    """Read q-points and frequencies from a CASTEP .phonon text file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    branches_match = re.search(r"Number of branches\s+(\d+)", text)
    qpoints_match = re.search(r"Number of wavevectors\s+(\d+)", text)
    unit_match = re.search(r"Frequencies in\s+([^\r\n]+)", text)
    if not (branches_match and qpoints_match and unit_match):
        raise ValueError(f"Unrecognized CASTEP phonon header: {path}")

    branch_count = int(branches_match.group(1))
    qpoint_count = int(qpoints_match.group(1))
    lines = text.splitlines()
    qpoints: list[list[float]] = []
    frequencies: list[list[float]] = []
    for index, line in enumerate(lines):
        qmatch = re.search(
            r"q-pt=\s*\d+\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)",
            line,
        )
        if not qmatch:
            continue
        qpoints.append([float(qmatch.group(i)) for i in range(1, 4)])
        current: list[float] = []
        for subline in lines[index + 1:index + 1 + branch_count]:
            fmatch = re.match(r"\s*\d+\s+([-+0-9.Ee]+)", subline)
            if not fmatch:
                break
            current.append(float(fmatch.group(1)))
        if len(current) != branch_count:
            raise ValueError(
                f"Expected {branch_count} frequencies after q-point {len(qpoints)}, "
                f"found {len(current)}"
            )
        frequencies.append(current)

    array = np.asarray(frequencies, dtype=float)
    if array.shape != (qpoint_count, branch_count):
        raise ValueError(
            f"Parsed frequency shape {array.shape}; expected "
            f"({qpoint_count}, {branch_count})"
        )
    unit = unit_match.group(1).strip()
    min_value = float(array.min())
    max_value = float(array.max())
    cm_to_thz = 1.0 / 33.35640951981521
    return {
        "path": str(path),
        "unit": unit,
        "qpoint_count": qpoint_count,
        "branch_count": branch_count,
        "qpoints": qpoints,
        "min_frequency": min_value,
        "max_frequency": max_value,
        "min_frequency_thz": min_value * cm_to_thz if unit.lower() == "cm-1" else None,
        "max_frequency_thz": max_value * cm_to_thz if unit.lower() == "cm-1" else None,
        "imaginary_modes": bool(min_value < -1e-6),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def validate_sources() -> dict[str, Any]:
    fixed_sources = [
        RANKING_FILE,
        SCREENING_SUMMARY,
        RANKING_DIAGNOSTICS,
        RANKING_CONFIGURATION,
        CR_IMAGE,
        NI_IMAGE,
        CR_SOURCE,
        NI_SOURCE,
    ]
    missing = [str(path) for path in fixed_sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required source files are missing:\n" + "\n".join(missing))

    br_image = first_existing(BR_IMAGE_CANDIDATES, "BrI2Ir3Te3 phonon image")
    br_raw = first_existing(BR_RAW_PHONON_CANDIDATES, "BrI2Ir3Te3 CASTEP phonon")

    for label, path in (
        ("BrI2Ir3Te3", br_image),
        ("Cr2Sn2Se6", CR_IMAGE),
        ("Ni2Sb2O6", NI_IMAGE),
    ):
        metadata = image_metadata(path)
        if min(metadata["width_px"], metadata["height_px"]) < 1200:
            raise ValueError(f"{label} image is too small for the final figure: {metadata}")
        if min(metadata["dpi"]) < 599:
            raise ValueError(f"{label} source is below the 600-dpi tolerance: {metadata}")

    br_numeric = parse_castep_phonon(br_raw)
    if not br_numeric["imaginary_modes"]:
        raise ValueError("CASTEP phonon data do not support the expected imaginary-mode conclusion")

    return {
        "br_image": br_image,
        "br_raw": br_raw,
        "br_numeric": br_numeric,
        "images": {
            "BrI2Ir3Te3": image_metadata(br_image),
            "Cr2Sn2Se6": image_metadata(CR_IMAGE),
            "Ni2Sb2O6": image_metadata(NI_IMAGE),
        },
    }


def validate_ranking() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = pd.read_csv(RANKING_FILE)
    summary = read_json(SCREENING_SUMMARY)
    diagnostics = read_json(RANKING_DIAGNOSTICS)
    ranking_config = read_json(RANKING_CONFIGURATION)

    required = {
        "material_id",
        "formula",
        "ehull",
        "p_joint_calibrated",
        "pJ_calibrated",
        "p_joint_mean",
        "p_joint_std",
        "probability_rank",
        "uncertainty_normalized",
        "score_joint_probability",
        "ranking_scope",
        *{f"p_joint_member_{index}" for index in range(1, 6)},
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Ranking source is missing required columns: {missing}")
    if len(frame) != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} candidates, found {len(frame)}")
    if summary.get("status") != "completed":
        raise ValueError("The screening summary is not marked completed")
    if int(summary.get("ranking_pool_count", -1)) != EXPECTED_N:
        raise ValueError("screening_summary.json does not confirm 8,803 ranking candidates")
    if int(diagnostics.get("ranking_pool_count", -1)) != EXPECTED_N:
        raise ValueError("candidate_ranking_diagnostics.json does not confirm 8,803 candidates")
    if frame["material_id"].isna().any() or frame["material_id"].astype(str).duplicated().any():
        raise ValueError("material_id must be non-missing and unique")
    if set(frame["ranking_scope"].astype(str)) != {"all_unknown"}:
        raise ValueError("The ranking source is not exclusively the all_unknown pool")

    numeric_columns = [
        "ehull",
        "p_joint_calibrated",
        "pJ_calibrated",
        "p_joint_mean",
        "p_joint_std",
        "uncertainty_normalized",
        "score_joint_probability",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} contains missing or non-finite values")
    if not frame["p_joint_calibrated"].between(0, 1).all():
        raise ValueError("p_joint_calibrated is outside [0, 1]")
    if not np.allclose(
        frame["p_joint_calibrated"], frame["pJ_calibrated"], rtol=0, atol=1e-15
    ):
        raise ValueError("pJ_calibrated does not match p_joint_calibrated")
    if not np.allclose(
        frame["p_joint_calibrated"], frame["score_joint_probability"], rtol=0, atol=1e-15
    ):
        raise ValueError("score_joint_probability does not match p_joint_calibrated")

    member_columns = [f"p_joint_member_{index}" for index in range(1, 6)]
    member_matrix = frame[member_columns].to_numpy(float)
    raw_mean = member_matrix.mean(axis=1)
    raw_std = member_matrix.std(axis=1, ddof=1)
    if not np.allclose(raw_mean, frame["p_joint_mean"], rtol=0, atol=2e-15):
        raise ValueError("p_joint_mean is not the five-member mean")
    if not np.allclose(raw_std, frame["p_joint_std"], rtol=0, atol=2e-15):
        raise ValueError("p_joint_std is not the five-member sample SD")
    if ranking_config.get("uncertainty_column") != "p_joint_std":
        raise ValueError("The frozen ranking configuration does not use p_joint_std as u_J")

    stored_rank = frame["probability_rank"].to_numpy(int)
    if not np.array_equal(np.sort(stored_rank), np.arange(1, EXPECTED_N + 1)):
        raise ValueError("probability_rank is not a unique 1..8803 permutation")
    ordered_index = frame.sort_values(
        ["score_joint_probability", "uncertainty_normalized", "material_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).index
    recalculated = pd.Series(
        np.arange(1, EXPECTED_N + 1), index=ordered_index
    ).reindex(frame.index)
    if not np.array_equal(recalculated.to_numpy(int), stored_rank):
        raise ValueError("Stored ranks do not reproduce the frozen ranking rule")

    source_compositions = frame["formula"].map(parse_formula)
    highlights: dict[str, dict[str, Any]] = {}
    for material in DISPLAY_MATERIALS:
        target = parse_formula(material)
        subset = frame.loc[source_compositions == target]
        if len(subset) != 1:
            raise ValueError(f"Expected one row matching {material}, found {len(subset)}")
        row = subset.iloc[0]
        highlights[material] = {
            "material_id": str(row["material_id"]),
            "source_formula": str(row["formula"]),
            "rank": int(row["probability_rank"]),
            "p_joint_calibrated": float(row["p_joint_calibrated"]),
            "u_joint": float(row["p_joint_std"]),
            "ehull": float(row["ehull"]),
            "top_fraction_percent": float(row["probability_rank"] / EXPECTED_N * 100.0),
        }

    highlighted_ids = {row["material_id"] for row in highlights.values()}
    ranking = pd.DataFrame(
        {
            "candidate": frame["formula"].astype(str),
            "material_id": frame["material_id"].astype(str),
            "rank": frame["probability_rank"].astype(int),
            "p_joint_calibrated": frame["p_joint_calibrated"].astype(float),
            "u_joint": frame["p_joint_std"].astype(float),
            "ehull_eV_atom": frame["ehull"].astype(float),
            "is_highlighted": frame["material_id"].astype(str).isin(highlighted_ids),
        }
    ).sort_values("rank", kind="mergesort").reset_index(drop=True)

    evidence = {
        "BrI2Ir3Te3": "Imaginary modes observed (self-calculated CASTEP phonon dispersion)",
        "Cr2Sn2Se6": "No imaginary modes reported (literature; structure match likely)",
        "Ni2Sb2O6": "No imaginary modes reported (literature; structure match likely)",
    }
    table = pd.DataFrame(
        [
            {
                "Candidate": material,
                "Rank": highlights[material]["rank"],
                "Top fraction (%)": highlights[material]["top_fraction_percent"],
                "p_J": highlights[material]["p_joint_calibrated"],
                "u_J": highlights[material]["u_joint"],
                "E_hull (eV atom^-1)": highlights[material]["ehull"],
                "Independent dynamical evidence": evidence[material],
            }
            for material in DISPLAY_MATERIALS
        ]
    )
    return ranking, table, highlights


# -----------------------------------------------------------------------------
# Figure construction
# -----------------------------------------------------------------------------
def plot_ranking(
    ax: mpl.axes.Axes,
    ranking: pd.DataFrame,
    highlights: dict[str, dict[str, Any]],
) -> None:
    ax.scatter(
        ranking["rank"],
        ranking["p_joint_calibrated"],
        s=4.2,
        color=COLORS["point"],
        alpha=0.42,
        linewidths=0,
        zorder=1,
    )
    rolling = ranking.set_index("rank")["p_joint_calibrated"].rolling(
        window=201,
        center=True,
        min_periods=1,
    ).median()
    ax.plot(
        rolling.index,
        rolling.to_numpy(float),
        color=COLORS["trend"],
        linewidth=0.8,
        alpha=0.90,
        zorder=2,
    )

    annotation_positions = {
        "BrI2Ir3Te3": (850, 0.61),
        "Cr2Sn2Se6": (2350, 0.91),
        "Ni2Sb2O6": (3900, 0.70),
    }
    for material in DISPLAY_MATERIALS:
        row = highlights[material]
        color = HIGHLIGHT_COLORS[material]
        ax.scatter(
            row["rank"],
            row["p_joint_calibrated"],
            s=46,
            facecolor=color,
            edgecolor=COLORS["white"],
            linewidth=0.9,
            zorder=5,
        )
        label = (
            f"{material}\n"
            f"Rank {row['rank']}\n"
            f"$p_J$ = {row['p_joint_calibrated']:.3f}\n"
            f"$u_J$ = {row['u_joint']:.2e}"
        )
        ax.annotate(
            label,
            xy=(row["rank"], row["p_joint_calibrated"]),
            xytext=annotation_positions[material],
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8.2,
            color=COLORS["text"],
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.75,
                "shrinkA": 2,
                "shrinkB": 4,
                "connectionstyle": "arc3,rad=0.05",
            },
            bbox={
                "boxstyle": "round,pad=0.23,rounding_size=0.04",
                "facecolor": COLORS["white"],
                "edgecolor": color,
                "linewidth": 0.65,
                "alpha": 0.97,
            },
            zorder=6,
        )

    ax.set(
        xlim=(0, EXPECTED_N + 80),
        ylim=(0, 1.02),
        xlabel="Candidate rank",
        ylabel="Calibrated Joint probability $p_J$",
        title=f"Ranked unlabeled candidates ($N$ = {EXPECTED_N:,})",
    )
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.set_yticks(np.linspace(0, 1, 6))
    style_axis(ax)
    panel_label(ax, "(a)", x=-0.11, y=1.03)


def plot_phonon_image(
    ax: mpl.axes.Axes,
    image: np.ndarray,
    panel: str,
    material_math: str,
    rank: int,
    probability: float,
    conclusion: str,
    conclusion_color: str,
    panel_x: float = -0.15,
) -> None:
    ax.imshow(image, interpolation="none", aspect="equal")
    ax.set_axis_off()
    ax.set_title(
        f"{material_math} (Rank {rank}, $p_J$ = {probability:.3f})",
        fontsize=8.7,
        pad=3.0,
        fontweight="normal",
    )
    panel_label(ax, panel, x=panel_x, y=1.02)
    ax.text(
        0.985,
        0.02,
        conclusion,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=conclusion_color,
        bbox={
            "boxstyle": "round,pad=0.20,rounding_size=0.04",
            "facecolor": COLORS["white"],
            "edgecolor": conclusion_color,
            "linewidth": 0.55,
            "alpha": 0.93,
        },
    )


def create_figure(
    ranking: pd.DataFrame,
    highlights: dict[str, dict[str, Any]],
    sources: dict[str, Any],
) -> plt.Figure:
    apply_sci_style()
    fig = plt.figure(figsize=(7.35, 8.25), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.0, 1.0, 1.72],
        width_ratios=[2.08, 1.0],
        hspace=0.06,
        wspace=0.05,
    )
    ax_rank = fig.add_subplot(grid[0:2, 0])
    ax_cr = fig.add_subplot(grid[0, 1])
    ax_ni = fig.add_subplot(grid[1, 1])
    ax_br = fig.add_subplot(grid[2, :])

    plot_ranking(ax_rank, ranking, highlights)
    plot_phonon_image(
        ax_cr,
        load_image(CR_IMAGE, blank_boxes=[(670, 950, 850, 1100)]),
        "(b)",
        r"Cr$_2$Sn$_2$Se$_6$",
        highlights["Cr2Sn2Se6"]["rank"],
        highlights["Cr2Sn2Se6"]["p_joint_calibrated"],
        "No imaginary modes",
        COLORS["deep_blue"],
        panel_x=-0.32,
    )
    plot_phonon_image(
        ax_ni,
        load_image(NI_IMAGE, blank_boxes=[(70, 130, 260, 300)]),
        "(c)",
        r"Ni$_2$Sb$_2$O$_6$",
        highlights["Ni2Sb2O6"]["rank"],
        highlights["Ni2Sb2O6"]["p_joint_calibrated"],
        "No imaginary modes",
        COLORS["teal"],
    )
    plot_phonon_image(
        ax_br,
        load_image(Path(sources["br_image"])),
        "(d)",
        r"BrI$_2$Ir$_3$Te$_3$",
        highlights["BrI2Ir3Te3"]["rank"],
        highlights["BrI2Ir3Te3"]["p_joint_calibrated"],
        "Imaginary modes observed",
        COLORS["terracotta"],
        panel_x=-0.02,
    )
    return fig


def save_figure(fig: plt.Figure) -> list[Path]:
    output_paths: list[Path] = []
    for stem in (PAPER_FIGURE_STEM, PROJECT_FIGURE_STEM):
        stem.parent.mkdir(parents=True, exist_ok=True)
        png = stem.with_suffix(".png")
        pdf = stem.with_suffix(".pdf")
        svg = stem.with_suffix(".svg")
        fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.03)
        output_paths.extend([png, pdf, svg])
    plt.close(fig)
    return output_paths


# -----------------------------------------------------------------------------
# Audit, export, and verification
# -----------------------------------------------------------------------------
def make_audit(
    ranking: pd.DataFrame,
    table: pd.DataFrame,
    highlights: dict[str, dict[str, Any]],
    sources: dict[str, Any],
) -> str:
    br = sources["br_numeric"]
    lines = [
        "FIGURE 6 AND TABLE 1 DATA AUDIT",
        "=" * 108,
        "",
        "CANDIDATE RANKING",
        "-" * 108,
        f"Source: {RANKING_FILE}",
        f"Rows: {len(ranking)}",
        f"Columns used: material_id, formula, probability_rank, p_joint_calibrated, "
        "p_joint_std, ehull",
        "Confirmed candidate total = 8803: YES",
        "Rank rule verified: p_joint_calibrated descending, uncertainty_normalized "
        "ascending, material_id ascending.",
        "p_J source: p_joint_calibrated (identical to pJ_calibrated).",
        "u_J source: p_joint_std; verified as ddof=1 SD of five member Joint probabilities.",
        "",
        "REPRESENTATIVE CANDIDATES",
        "-" * 108,
    ]
    for material in DISPLAY_MATERIALS:
        row = highlights[material]
        lines.append(
            f"{material}: material_id={row['material_id']}; source_formula={row['source_formula']}; "
            f"rank={row['rank']}; top_fraction={row['top_fraction_percent']:.12f}%; "
            f"p_J={row['p_joint_calibrated']:.12f}; u_J={row['u_joint']:.12e}; "
            f"E_hull={row['ehull']:.12f} eV atom^-1"
        )
    lines.extend(
        [
            "",
            "PHONON SOURCES",
            "-" * 108,
            "BrI2Ir3Te3:",
            f"  self-calculated figure: {sources['br_image']}",
            f"  CASTEP numeric result: {sources['br_raw']}",
            f"  figure metadata: {sources['images']['BrI2Ir3Te3']}",
            f"  numeric unit: {br['unit']}",
            f"  q-points: {br['qpoint_count']}; branches: {br['branch_count']}",
            f"  min frequency: {br['min_frequency']:.9f} {br['unit']} "
            f"({br['min_frequency_thz']:.9f} THz)",
            f"  max frequency: {br['max_frequency']:.9f} {br['unit']} "
            f"({br['max_frequency_thz']:.9f} THz)",
            "  high-symmetry path shown in ck Phon2.tif: Gamma-M-K-Gamma",
            "  imaginary modes: YES",
            "",
            "Cr2Sn2Se6:",
            f"  phonon image: {CR_IMAGE}",
            f"  source record: {CR_SOURCE}",
            f"  figure metadata: {sources['images']['Cr2Sn2Se6']}",
            "  literature source: Supplementary Fig. 1(a), DOI 10.1038/s41699-020-00195-9",
            "  structure match to project candidate: LIKELY",
            "  imaginary modes reported by authors: NO",
            "",
            "Ni2Sb2O6:",
            f"  phonon image: {NI_IMAGE}",
            f"  source record: {NI_SOURCE}",
            f"  figure metadata: {sources['images']['Ni2Sb2O6']}",
            "  literature source: Fig. S1(b), DOI 10.60893/figshare.apl.28824896.v1",
            "  structure match to project candidate: LIKELY",
            "  imaginary modes reported by authors: NO",
            "",
            "TABLE 1 DATA",
            "-" * 108,
            table.to_csv(index=False, float_format="%.12g").rstrip(),
            "",
            "OUTPUT STATUS",
            "-" * 108,
            "Formal Figure 6 generated: YES",
            "Partial figure generated: NO",
            "All numerical ranking values come from the 8,803-row final screening table.",
            "No phonon curve was fabricated or digitized.",
            "The publisher's original (a)/(b) panel markers were removed in memory from "
            "white/empty regions; curves, axes, labels, and values were preserved.",
            "Panels (b)-(d) embed high-resolution source images; panel titles, labels, and "
            "conclusion tags are vector elements in PDF/SVG.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def export_tables_and_audit(
    ranking: pd.DataFrame,
    table: pd.DataFrame,
    audit: str,
    sources: dict[str, Any],
    highlights: dict[str, dict[str, Any]],
) -> list[Path]:
    written: list[Path] = []
    manifest = {
        "figure": "Figure 6: Candidate prioritization and phonon validation",
        "ranking_source": str(RANKING_FILE),
        "candidate_count": len(ranking),
        "highlights": highlights,
        "phonon_images": sources["images"],
        "br_castep_phonon": sources["br_numeric"],
        "partial_figure": False,
    }
    for output_dir in (PAPER_OUTPUT_DIR, PROJECT_OUTPUT_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path = output_dir / TABLE_NAME
        ranking_path = output_dir / RANKING_DATA_NAME
        audit_path = output_dir / AUDIT_NAME
        manifest_path = output_dir / MANIFEST_NAME
        table.to_csv(table_path, index=False, float_format="%.12g", lineterminator="\n")
        ranking.to_csv(ranking_path, index=False, float_format="%.12g", lineterminator="\n")
        audit_path.write_text(audit, encoding="utf-8")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.extend([table_path, ranking_path, audit_path, manifest_path])
    return written


def sync_code_copy() -> list[Path]:
    source = Path(__file__).resolve()
    targets = [PAPER_CODE_DIR / "图6.py", PROJECT_OUTPUT_DIR / "Figure6.py"]
    copied: list[Path] = []
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    return copied


def verify_outputs(
    figures: list[Path],
    data_outputs: list[Path],
    code_outputs: list[Path],
) -> dict[str, Any]:
    required = [*figures, *data_outputs, *code_outputs]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Missing or empty outputs:\n" + "\n".join(missing))

    checks: dict[str, Any] = {}
    for png in (PAPER_FIGURE_STEM.with_suffix(".png"), PROJECT_FIGURE_STEM.with_suffix(".png")):
        with Image.open(png) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
            if min(dpi) < 599:
                raise RuntimeError(f"PNG is below the 600-dpi tolerance: {png}: {dpi}")
            checks[str(png)] = {"dimensions": image.size, "dpi": dpi}
    for pdf in (PAPER_FIGURE_STEM.with_suffix(".pdf"), PROJECT_FIGURE_STEM.with_suffix(".pdf")):
        if not pdf.read_bytes()[:8].startswith(b"%PDF"):
            raise RuntimeError(f"Invalid PDF header: {pdf}")
    for svg in (PAPER_FIGURE_STEM.with_suffix(".svg"), PROJECT_FIGURE_STEM.with_suffix(".svg")):
        if "<svg" not in svg.read_text(encoding="utf-8", errors="ignore")[:1000]:
            raise RuntimeError(f"Invalid SVG: {svg}")
    if len(pd.read_csv(PAPER_OUTPUT_DIR / RANKING_DATA_NAME)) != EXPECTED_N:
        raise RuntimeError("Exported ranking table does not contain 8,803 rows")
    if len(pd.read_csv(PAPER_OUTPUT_DIR / TABLE_NAME)) != 3:
        raise RuntimeError("Exported Table 1 does not contain three candidates")
    return checks


def print_summary(
    highlights: dict[str, dict[str, Any]],
    sources: dict[str, Any],
    generated: list[Path],
    checks: dict[str, Any],
) -> None:
    print("\nFINAL FIGURE 6 SUMMARY")
    print("=" * 88)
    print(f"Candidate ranking source: {RANKING_FILE}")
    print(f"Number of unlabeled candidates: {EXPECTED_N}")
    print("Highlight materials:")
    for material in DISPLAY_MATERIALS:
        row = highlights[material]
        print(
            f"- {material}: rank={row['rank']}, "
            f"p_J={row['p_joint_calibrated']:.12f}, "
            f"u_J={row['u_joint']:.12e}, "
            f"E_hull={row['ehull']:.12f}"
        )
    print("\nPhonon data:")
    print(f"- BrI2Ir3Te3: found; self-calculated image={sources['br_image']}")
    print(
        "  CASTEP check: min="
        f"{sources['br_numeric']['min_frequency_thz']:.6f} THz; imaginary modes=YES"
    )
    print(f"- Cr2Sn2Se6: found; literature image={CR_IMAGE}")
    print(f"- Ni2Sb2O6: found; literature image={NI_IMAGE}")
    print("\nGenerated files:")
    for path in generated:
        print(path)
    for path, check in checks.items():
        print(f"PNG check: {path}: {check}")
    print("Formal Figure 6 generated: YES")


def main() -> None:
    PAPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_CODE_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = validate_sources()
    ranking, table, highlights = validate_ranking()
    audit = make_audit(ranking, table, highlights, sources)

    # Print all selected paths and key values before drawing.
    print(audit)

    data_outputs = export_tables_and_audit(
        ranking, table, audit, sources, highlights
    )
    figure = create_figure(ranking, highlights, sources)
    figure_outputs = save_figure(figure)
    code_outputs = sync_code_copy()
    checks = verify_outputs(figure_outputs, data_outputs, code_outputs)
    generated = [*figure_outputs, *data_outputs, *code_outputs]
    print_summary(highlights, sources, generated, checks)


if __name__ == "__main__":
    main()
