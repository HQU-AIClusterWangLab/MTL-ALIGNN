"""Create frozen, leakage-audited train/validation/test review splits.

This module intentionally does not alter ``supervised.csv`` or ``iid_split.csv``.
It derives a canonical reduced composition directly from the per-element atom
counts encoded by ``natoms`` and ``frac_Z_*`` and writes self-contained review
manifests below ``data/processed/review_splits``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from functools import reduce
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from data.utils import load_config


LABEL_COLUMNS = ("thermo_label", "dynamic_label", "joint_label")
SPLIT_NAMES = ("train", "val", "test")
MISSING_GROUP = "__MISSING__"
ALGORITHM_VERSION = "review-splits-v1"

# Index is atomic number. Keeping this lookup local avoids requiring ASE or
# pymatgen merely to recover integer atom counts from the prepared features.
CHEMICAL_SYMBOLS = (
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
VALID_SYMBOLS = frozenset(CHEMICAL_SYMBOLS[1:])


@dataclass(frozen=True)
class CanonicalComposition:
    """A collision-free, readable canonical representation of a composition."""

    reduced_counts: tuple[tuple[str, int], ...]
    composition_id: str
    formula: str
    chemical_system_id: str


def _hill_order(symbols: Iterable[str]) -> list[str]:
    """Return deterministic Hill-system element order."""

    unique = set(symbols)
    if "C" not in unique:
        return sorted(unique)
    ordered = ["C"]
    if "H" in unique:
        ordered.append("H")
    ordered.extend(sorted(unique.difference(ordered)))
    return ordered


def canonicalize_element_counts(
    counts: Mapping[str, Real] | Iterable[tuple[str, Real]],
    *,
    integer_tolerance: float = 1e-8,
) -> CanonicalComposition:
    """Reduce element counts and canonicalize them independent of input order.

    Counts must be non-negative integers (integer-valued floats are accepted).
    A common supercell multiplier is removed with the greatest common divisor.
    The ID contains explicit separators and therefore has no formula-token
    ambiguity; the companion formula follows standard Hill ordering.
    """

    items = counts.items() if isinstance(counts, Mapping) else counts
    merged: dict[str, int] = {}
    for raw_symbol, raw_count in items:
        symbol = str(raw_symbol).strip()
        if symbol not in VALID_SYMBOLS:
            raise ValueError(f"Unknown chemical symbol: {raw_symbol!r}")
        try:
            numeric = float(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Atom count for {symbol} is not numeric: {raw_count!r}"
            ) from exc
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"Atom count for {symbol} must be finite and >= 0")
        integer = int(round(numeric))
        if abs(numeric - integer) > integer_tolerance:
            raise ValueError(
                f"Atom count for {symbol} is not integer-valued: {numeric}"
            )
        if integer:
            merged[symbol] = merged.get(symbol, 0) + integer

    if not merged:
        raise ValueError("At least one positive element count is required")
    divisor = reduce(math.gcd, merged.values())
    reduced = {symbol: value // divisor for symbol, value in merged.items()}
    ordered_symbols = _hill_order(reduced)
    ordered_counts = tuple((symbol, reduced[symbol]) for symbol in ordered_symbols)
    composition_id = "|".join(
        f"{symbol}={count}" for symbol, count in ordered_counts
    )
    formula = "".join(
        symbol if count == 1 else f"{symbol}{count}"
        for symbol, count in ordered_counts
    )
    chemical_system_id = "-".join(sorted(reduced))
    return CanonicalComposition(
        reduced_counts=ordered_counts,
        composition_id=composition_id,
        formula=formula,
        chemical_system_id=chemical_system_id,
    )


def element_counts_from_fraction_row(
    row: Mapping[str, object] | pd.Series,
    *,
    natoms_column: str = "natoms",
    fraction_prefix: str = "frac_Z_",
    max_atomic_number: int = 118,
    integer_tolerance: float = 1e-6,
) -> dict[str, int]:
    """Recover exact element counts from one prepared metadata row."""

    try:
        natoms_value = float(row[natoms_column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {natoms_column!r} value") from exc
    natoms = int(round(natoms_value))
    if (
        not math.isfinite(natoms_value)
        or natoms <= 0
        or abs(natoms_value - natoms) > integer_tolerance
    ):
        raise ValueError(f"{natoms_column!r} must be a positive integer")

    counts: dict[str, int] = {}
    for atomic_number in range(1, max_atomic_number + 1):
        column = f"{fraction_prefix}{atomic_number:03d}"
        try:
            fraction = float(row[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid fraction column {column!r}") from exc
        if not math.isfinite(fraction) or fraction < 0:
            raise ValueError(f"Fraction {column!r} must be finite and >= 0")
        raw_count = fraction * natoms
        count = int(round(raw_count))
        if abs(raw_count - count) > integer_tolerance:
            raise ValueError(
                f"{column!r} does not encode an integer atom count: {raw_count}"
            )
        if count:
            counts[CHEMICAL_SYMBOLS[atomic_number]] = count

    if sum(counts.values()) != natoms:
        raise ValueError(
            f"Recovered {sum(counts.values())} atoms, expected {natoms}"
        )
    return counts


def add_canonical_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy augmented with canonical composition/group identifiers."""

    required = {
        "material_id",
        "natoms",
        "structure_family",
        "structure_hash",
        *LABEL_COLUMNS,
        *(f"frac_Z_{z:03d}" for z in range(1, 119)),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Supervised table is missing columns: {missing}")
    if frame["material_id"].isna().any():
        raise ValueError("material_id contains missing values")
    if not frame["material_id"].is_unique:
        raise ValueError("material_id must be unique in the supervised table")

    # Validate/recover every atom count in one vectorized pass.  The small final
    # row loop handles only non-zero elements and is much faster than 8,102 x
    # 118 labelled Series lookups.
    fraction_columns = [f"frac_Z_{z:03d}" for z in range(1, 119)]
    natoms_values = pd.to_numeric(frame["natoms"], errors="coerce").to_numpy(
        dtype=float
    )
    natoms = np.rint(natoms_values).astype(np.int64)
    if (
        np.any(~np.isfinite(natoms_values))
        or np.any(natoms <= 0)
        or np.any(np.abs(natoms_values - natoms) > 1e-6)
    ):
        raise ValueError("natoms must contain positive integers")
    fractions = frame[fraction_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("Element fractions must be finite and non-negative")
    raw_counts = fractions * natoms[:, None]
    integer_counts = np.rint(raw_counts).astype(np.int64)
    if np.any(np.abs(raw_counts - integer_counts) > 1e-6):
        raise ValueError("Element fractions do not encode integer atom counts")
    if np.any(integer_counts.sum(axis=1) != natoms):
        raise ValueError("Recovered atom counts do not sum to natoms")

    canonical = []
    for row_counts in integer_counts:
        nonzero = np.flatnonzero(row_counts)
        counts = {
            CHEMICAL_SYMBOLS[index + 1]: int(row_counts[index])
            for index in nonzero
        }
        canonical.append(canonicalize_element_counts(counts))

    result = frame.copy()
    result["canonical_reduced_composition_id"] = [
        item.composition_id for item in canonical
    ]
    result["canonical_formula"] = [item.formula for item in canonical]
    result["chemical_system_id"] = [
        item.chemical_system_id for item in canonical
    ]
    result["structure_family_id"] = (
        result["structure_family"].fillna(MISSING_GROUP).astype(str)
    )
    result["structure_hash"] = (
        result["structure_hash"].fillna(MISSING_GROUP).astype(str)
    )
    return result


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("Exactly three train/val/test ratios are required")
    values = tuple(float(value) for value in ratios)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Split ratios must be finite and positive")
    total = sum(values)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def _target_split_sizes(n_rows: int, ratios: Sequence[float]) -> dict[str, int]:
    normalized = _validate_ratios(ratios)
    raw = [n_rows * ratio for ratio in normalized]
    sizes = [math.floor(value) for value in raw]
    # Largest remainder method with a stable train -> val -> test tie-break.
    order = sorted(range(3), key=lambda i: (-(raw[i] - sizes[i]), i))
    for index in order[: n_rows - sum(sizes)]:
        sizes[index] += 1
    return dict(zip(SPLIT_NAMES, sizes))


def _stratification_key(
    frame: pd.DataFrame, label_columns: Sequence[str]
) -> pd.Series:
    missing = sorted(set(label_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    return frame[list(label_columns)].astype(str).agg("|".join, axis=1)


def make_random_stratified_assignments(
    frame: pd.DataFrame,
    *,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    label_columns: Sequence[str] = LABEL_COLUMNS,
) -> pd.DataFrame:
    """Make deterministic row-random assignments stratified on joint label state."""

    if not frame["material_id"].is_unique:
        raise ValueError("material_id must be unique")
    sizes = _target_split_sizes(len(frame), ratios)
    strata = _stratification_key(frame, label_columns)
    all_indices = np.arange(len(frame))
    first = StratifiedShuffleSplit(
        n_splits=1, test_size=sizes["test"], random_state=int(seed)
    )
    train_val_idx, test_idx = next(first.split(all_indices, strata))
    train_val_strata = strata.iloc[train_val_idx]
    second = StratifiedShuffleSplit(
        n_splits=1, test_size=sizes["val"], random_state=int(seed) + 1
    )
    train_rel, val_rel = next(
        second.split(np.arange(len(train_val_idx)), train_val_strata)
    )
    split = np.full(len(frame), "test", dtype=object)
    split[train_val_idx[train_rel]] = "train"
    split[train_val_idx[val_rel]] = "val"
    return pd.DataFrame(
        {"material_id": frame["material_id"].astype(str), "split": split}
    )


def _holdout_score(
    full: pd.DataFrame,
    holdout: pd.DataFrame,
    ratio: float,
    label_columns: Sequence[str],
) -> float:
    size_error = abs(len(holdout) / len(full) - ratio)
    label_error = sum(
        abs(float(holdout[column].mean()) - float(full[column].mean()))
        for column in label_columns
    )
    return 6.0 * size_error + label_error


def choose_stratified_group_holdout(
    frame: pd.DataFrame,
    group_column: str,
    ratio: float,
    seed: int,
    trials: int,
    *,
    label_columns: Sequence[str] = LABEL_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the best of deterministic group-shuffle holdout candidates."""

    if group_column not in frame:
        raise ValueError(f"Missing group column: {group_column}")
    if not 0 < ratio < 1:
        raise ValueError("Holdout ratio must lie strictly between zero and one")
    if int(trials) <= 0:
        raise ValueError("trials must be positive")
    groups = frame[group_column].fillna(MISSING_GROUP).astype(str)
    unique_groups = np.asarray(sorted(groups.unique()), dtype=object)
    if len(unique_groups) < 3:
        raise ValueError(f"At least three {group_column!r} groups are required")
    lookup = {group: index for index, group in enumerate(unique_groups)}
    group_codes = groups.map(lookup).to_numpy(dtype=np.int64)
    group_sizes = np.bincount(group_codes, minlength=len(unique_groups)).astype(float)
    label_values = frame[list(label_columns)].to_numpy(dtype=float)
    if np.any(~np.isfinite(label_values)) or not set(np.unique(label_values)).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("Review split labels must be finite binary values")
    group_label_sums = np.vstack(
        [
            np.bincount(
                group_codes,
                weights=label_values[:, label_index],
                minlength=len(unique_groups),
            )
            for label_index in range(len(label_columns))
        ]
    )
    total_label_sums = label_values.sum(axis=0)
    n_rows = float(len(frame))
    splitter = ShuffleSplit(
        n_splits=int(trials), test_size=float(ratio), random_state=int(seed)
    )
    best: tuple[float, np.ndarray] | None = None
    for _development_groups, holdout_groups in splitter.split(unique_groups):
        holdout_n = float(group_sizes[holdout_groups].sum())
        development_n = n_rows - holdout_n
        holdout_label_sums = group_label_sums[:, holdout_groups].sum(axis=1)
        development_label_sums = total_label_sums - holdout_label_sums
        holdout_has_both = np.all(
            (holdout_label_sums > 0) & (holdout_label_sums < holdout_n)
        )
        development_has_both = np.all(
            (development_label_sums > 0)
            & (development_label_sums < development_n)
        )
        if not holdout_has_both or not development_has_both:
            continue
        size_error = abs(holdout_n / n_rows - ratio)
        label_error = float(
            np.abs(
                holdout_label_sums / holdout_n - total_label_sums / n_rows
            ).sum()
        )
        score = 6.0 * size_error + label_error
        if best is None or score < best[0]:
            best = (score, holdout_groups.copy())
    if best is None:
        raise RuntimeError(
            f"Unable to find a label-valid holdout for {group_column!r}"
        )
    is_holdout = np.isin(group_codes, best[1])
    return np.flatnonzero(~is_holdout), np.flatnonzero(is_holdout)


def make_group_disjoint_assignments(
    frame: pd.DataFrame,
    group_column: str,
    *,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    trials: int = 800,
    label_columns: Sequence[str] = LABEL_COLUMNS,
) -> pd.DataFrame:
    """Make train/val/test assignments with zero target-group overlap."""

    normalized = _validate_ratios(ratios)
    development_idx, test_idx = choose_stratified_group_holdout(
        frame,
        group_column,
        normalized[2],
        seed,
        trials,
        label_columns=label_columns,
    )
    development = frame.iloc[development_idx].reset_index(drop=True)
    relative_val = normalized[1] / (normalized[0] + normalized[1])
    train_rel, val_rel = choose_stratified_group_holdout(
        development,
        group_column,
        relative_val,
        seed + 1,
        trials,
        label_columns=label_columns,
    )
    split = np.full(len(frame), "test", dtype=object)
    split[development_idx[train_rel]] = "train"
    split[development_idx[val_rel]] = "val"
    # Explicitly retain test_idx in the construction as an internal sanity check.
    if set(np.flatnonzero(split == "test")) != set(test_idx.tolist()):
        raise RuntimeError("Internal group split index mismatch")
    return pd.DataFrame(
        {"material_id": frame["material_id"].astype(str), "split": split}
    )


def _normalized_groups(series: pd.Series) -> pd.Series:
    return series.fillna(MISSING_GROUP).astype(str)


def pairwise_group_overlap(
    merged: pd.DataFrame, group_column: str, *, example_limit: int = 10
) -> dict[str, dict[str, object]]:
    """Report unique group overlap for each split pair."""

    group_sets = {
        name: set(_normalized_groups(merged.loc[merged["split"] == name, group_column]))
        for name in SPLIT_NAMES
    }
    result: dict[str, dict[str, object]] = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = sorted(group_sets[left].intersection(group_sets[right]))
            result[f"{left}__{right}"] = {
                "overlap_group_count": len(overlap),
                "examples": overlap[:example_limit],
            }
    return result


def cross_split_group_overlap(
    merged: pd.DataFrame, group_column: str, *, example_limit: int = 10
) -> dict[str, object]:
    """Count unique groups that occur in two or three data splits."""

    membership = pd.DataFrame(
        {
            "group": _normalized_groups(merged[group_column]),
            "split": merged["split"].astype(str),
        }
    ).drop_duplicates()
    split_counts = membership.groupby("group", sort=True)["split"].nunique()
    leaking = split_counts.loc[split_counts > 1]
    all_three = split_counts.loc[split_counts == 3]
    return {
        "unique_groups_spanning_multiple_splits": int(len(leaking)),
        "unique_groups_spanning_exactly_two_splits": int((leaking == 2).sum()),
        "unique_groups_spanning_all_three_splits": int(len(all_three)),
        "examples": leaking.index.astype(str).tolist()[:example_limit],
    }


def _group_size_distribution(series: pd.Series) -> dict[str, float | int]:
    sizes = _normalized_groups(series).value_counts(sort=False).to_numpy(dtype=float)
    return {
        "n_groups": int(len(sizes)),
        "min": int(sizes.min()),
        "q25": float(np.quantile(sizes, 0.25)),
        "median": float(np.quantile(sizes, 0.50)),
        "mean": float(sizes.mean()),
        "q75": float(np.quantile(sizes, 0.75)),
        "max": int(sizes.max()),
    }


def assignment_sha256(assignments: pd.DataFrame) -> str:
    """Hash only the logical material_id -> split mapping."""

    ordered = assignments[["material_id", "split"]].copy()
    ordered["material_id"] = ordered["material_id"].astype(str)
    ordered["split"] = ordered["split"].astype(str)
    ordered = ordered.sort_values("material_id", kind="mergesort")
    payload = "".join(
        f"{material_id}\t{split_name}\n"
        for material_id, split_name in ordered.itertuples(index=False, name=None)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_assignments(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    target_group_column: str | None,
    enforce_target_disjoint: bool,
    group_columns: Sequence[str] = (
        "canonical_reduced_composition_id",
        "chemical_system_id",
        "structure_family_id",
        "structure_hash",
    ),
    label_columns: Sequence[str] = LABEL_COLUMNS,
) -> dict[str, object]:
    """Validate full coverage and return leakage/statistics audit data."""

    if assignments["material_id"].isna().any():
        raise ValueError("Split assignments contain missing material_id")
    if not assignments["material_id"].is_unique:
        raise ValueError("Split assignments contain duplicate material_id")
    if set(assignments["split"].astype(str)) != set(SPLIT_NAMES):
        raise ValueError("Assignments must contain train, val, and test")
    source_ids = set(frame["material_id"].astype(str))
    assigned_ids = set(assignments["material_id"].astype(str))
    if source_ids != assigned_ids:
        raise ValueError(
            "Assignments do not exactly cover source IDs: "
            f"missing={len(source_ids - assigned_ids)}, "
            f"extra={len(assigned_ids - source_ids)}"
        )
    merged = frame.merge(
        assignments[["material_id", "split"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(frame):
        raise ValueError("Assignments do not cover every source row exactly once")

    overlaps = {
        column: pairwise_group_overlap(merged, column)
        for column in group_columns
    }
    unique_overlaps = {
        column: cross_split_group_overlap(merged, column)
        for column in group_columns
    }
    if target_group_column is not None:
        target_pairwise_total = sum(
            int(item["overlap_group_count"])
            for item in overlaps[target_group_column].values()
        )
        target_unique_total = int(
            unique_overlaps[target_group_column][
                "unique_groups_spanning_multiple_splits"
            ]
        )
        if enforce_target_disjoint and target_unique_total:
            raise ValueError(
                f"Target-group leakage for {target_group_column}: "
                f"{target_unique_total} unique groups"
            )
    else:
        target_pairwise_total = None
        target_unique_total = None

    split_summary: dict[str, object] = {}
    for split_name in SPLIT_NAMES:
        part = merged.loc[merged["split"] == split_name]
        split_summary[split_name] = {
            "n_rows": int(len(part)),
            "n_unique_material_ids": int(part["material_id"].nunique()),
            "label_positive_counts": {
                column: int(part[column].sum()) for column in label_columns
            },
            "label_rates": {
                column: float(part[column].mean()) for column in label_columns
            },
            "group_size_distributions": {
                column: _group_size_distribution(part[column])
                for column in group_columns
            },
        }

    nonzero_columns = [
        column
        for column, pairs in overlaps.items()
        if any(int(item["overlap_group_count"]) > 0 for item in pairs.values())
    ]
    return {
        "coverage": {
            "source_rows": int(len(frame)),
            "assigned_rows": int(len(assignments)),
            "all_source_rows_covered_once": True,
            "material_id_unique": True,
            "all_three_splits_present": True,
        },
        "assignment_sha256": assignment_sha256(assignments),
        "target_group_column": target_group_column,
        "target_group_overlap_unique_count": target_unique_total,
        "target_group_pairwise_overlap_count_sum": target_pairwise_total,
        "target_group_disjoint_enforced": bool(enforce_target_disjoint),
        "split_summary": split_summary,
        "pairwise_group_overlap": overlaps,
        "unique_cross_split_group_overlap": unique_overlaps,
        "group_columns_with_nonzero_cross_split_overlap": nonzero_columns,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _manifest_table(
    frame: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "material_id",
        "split",
        "canonical_reduced_composition_id",
        "canonical_formula",
        "chemical_system_id",
        "structure_family_id",
        "structure_hash",
        *LABEL_COLUMNS,
    ]
    if "row_id" in frame:
        columns.insert(2, "row_id")
    merged = frame.merge(
        assignments[["material_id", "split"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    return merged[columns].sort_values("material_id", kind="mergesort")


def generate_review_splits(
    supervised_path: Path,
    legacy_iid_path: Path,
    output_dir: Path,
    *,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    trials: int = 800,
    expected_rows: int | None = 8102,
) -> dict[str, object]:
    """Generate all review manifests and their machine-readable audit."""

    frame = pd.read_csv(supervised_path)
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} supervised rows, found {len(frame)}"
        )
    enriched = add_canonical_group_columns(frame)
    output_dir.mkdir(parents=True, exist_ok=True)

    scheme_specs = {
        "random_iid_stratified": {
            "filename": "random_iid_stratified.csv",
            "target": None,
            "seed": int(seed),
            "assignments": make_random_stratified_assignments(
                enriched, ratios=ratios, seed=seed
            ),
        },
        "composition_disjoint": {
            "filename": "composition_disjoint.csv",
            "target": "canonical_reduced_composition_id",
            "seed": int(seed) + 101,
            "assignments": make_group_disjoint_assignments(
                enriched,
                "canonical_reduced_composition_id",
                ratios=ratios,
                seed=int(seed) + 101,
                trials=trials,
            ),
        },
        "chemical_system_disjoint": {
            "filename": "chemical_system_disjoint.csv",
            "target": "chemical_system_id",
            "seed": int(seed) + 202,
            "assignments": make_group_disjoint_assignments(
                enriched,
                "chemical_system_id",
                ratios=ratios,
                seed=int(seed) + 202,
                trials=trials,
            ),
        },
        "structure_family_disjoint": {
            "filename": "structure_family_disjoint.csv",
            "target": "structure_family_id",
            "seed": int(seed) + 303,
            "assignments": make_group_disjoint_assignments(
                enriched,
                "structure_family_id",
                ratios=ratios,
                seed=int(seed) + 303,
                trials=trials,
            ),
        },
    }

    audit: dict[str, object] = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "frozen_manifest_set": True,
        "source": {
            "supervised_path": str(supervised_path.resolve()),
            "supervised_rows": int(len(enriched)),
            "supervised_sha256": _file_sha256(supervised_path),
            "legacy_iid_path": str(legacy_iid_path.resolve()),
            "legacy_iid_sha256": _file_sha256(legacy_iid_path),
        },
        "parameters": {
            "ratios": dict(zip(SPLIT_NAMES, _validate_ratios(ratios))),
            "base_seed": int(seed),
            "group_search_trials_per_holdout": int(trials),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "canonical_composition_definition": {
            "source": "integer counts = round(natoms * frac_Z_001..frac_Z_118)",
            "reduction": "divide all positive element counts by their gcd",
            "ordering": "Hill order (C, H, then alphabetical; alphabetical without C)",
            "id_format": "Element=count tokens separated by |",
            "formula_format": "Hill formula with count 1 omitted",
            "atom_order_invariant": True,
            "common_supercell_multiplier_invariant": True,
        },
        "near_duplicate_structure_audit": {
            "available_geometry_tolerant_fingerprint": False,
            "performed_check": "exact cross-split overlap of existing structure_hash",
            "limitation": (
                "No geometry-tolerant structure fingerprint is present in supervised.csv. "
                "The exact structure_hash check cannot rule out near duplicates represented "
                "with a different cell basis, origin, tolerance, strain, or relaxation."
            ),
        },
        "manifests": {},
    }

    manifest_audits: dict[str, object] = {}
    for scheme_name, raw_spec in scheme_specs.items():
        assignments = raw_spec["assignments"]
        assert isinstance(assignments, pd.DataFrame)
        target = raw_spec["target"]
        assert target is None or isinstance(target, str)
        filename = str(raw_spec["filename"])
        path = output_dir / filename
        manifest = _manifest_table(enriched, assignments)
        manifest.to_csv(path, index=False, lineterminator="\n")
        item = audit_assignments(
            enriched,
            assignments,
            target_group_column=target,
            enforce_target_disjoint=target is not None,
        )
        item.update(
            {
                "path": str(path.resolve()),
                "filename": filename,
                "seed": int(raw_spec["seed"]),
                "manifest_sha256": _file_sha256(path),
            }
        )
        manifest_audits[scheme_name] = item
    audit["manifests"] = manifest_audits

    legacy_assignments = pd.read_csv(legacy_iid_path)
    legacy_audit = audit_assignments(
        enriched,
        legacy_assignments,
        target_group_column="canonical_reduced_composition_id",
        enforce_target_disjoint=False,
        group_columns=(
            "canonical_reduced_composition_id",
            "chemical_system_id",
            "structure_family_id",
            "structure_hash",
            "reduced_formula",
        ),
    )
    canonical_unique_total = int(
        legacy_audit["target_group_overlap_unique_count"]
    )
    audit["current_iid_split_canonical_leakage_audit"] = {
        **legacy_audit,
        "legacy_claimed_group_column": "reduced_formula",
        "canonical_composition_leakage_detected": canonical_unique_total > 0,
        "interpretation": (
            "Any non-zero canonical reduced-composition overlap means the current "
            "iid_split.csv is not composition-disjoint under atom-count canonicalization."
        ),
    }

    audit_path = output_dir / "audit.json"
    _write_json(audit_path, audit)
    checksum_data = {
        "schema_version": 1,
        "hash_algorithm": "SHA-256",
        "self_file_excluded_to_avoid_a_circular_hash": True,
        "source_files": {
            str(supervised_path.resolve()): _file_sha256(supervised_path),
            str(legacy_iid_path.resolve()): _file_sha256(legacy_iid_path),
        },
        "output_files": {
            path.name: _file_sha256(path)
            for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != "manifest_sha256.json"
        },
    }
    _write_json(output_dir / "manifest_sha256.json", checksum_data)
    return audit


def _parse_args() -> argparse.Namespace:
    config = load_config()
    processed = Path(config["paths"]["processed_dir"])
    split = config["split"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervised", type=Path, default=processed / "supervised.csv"
    )
    parser.add_argument(
        "--legacy-iid", type=Path, default=processed / "iid_split.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=processed / "review_splits"
    )
    parser.add_argument("--seed", type=int, default=int(split["seed"]))
    parser.add_argument(
        "--trials", type=int, default=int(split["search_trials"])
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(
            float(split["train_ratio"]),
            float(split["val_ratio"]),
            float(split["test_ratio"]),
        ),
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=8102,
        help="Fail closed if supervised.csv does not contain this many rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = generate_review_splits(
        args.supervised,
        args.legacy_iid,
        args.output,
        ratios=args.ratios,
        seed=args.seed,
        trials=args.trials,
        expected_rows=args.expected_rows,
    )
    compact = {
        "output": str(args.output.resolve()),
        "rows": audit["source"]["supervised_rows"],
        "manifests": {
            name: {
                "sha256": item["manifest_sha256"],
                "target_overlap_unique_groups": item[
                    "target_group_overlap_unique_count"
                ],
                "split_sizes": {
                    split_name: item["split_summary"][split_name]["n_rows"]
                    for split_name in SPLIT_NAMES
                },
            }
            for name, item in audit["manifests"].items()
        },
        "legacy_iid_canonical_overlap_unique_groups": audit[
            "current_iid_split_canonical_leakage_audit"
        ]["target_group_overlap_unique_count"],
        "legacy_iid_canonical_pairwise_overlap_sum": audit[
            "current_iid_split_canonical_leakage_audit"
        ]["target_group_pairwise_overlap_count_sum"],
    }
    json.dump(compact, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
