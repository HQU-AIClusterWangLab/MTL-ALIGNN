from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.make_splits import choose_group_holdout
from data.utils import load_config, write_json


LABELS = ["thermo_label", "dynamic_label", "joint_label"]


def validate_ood_split(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    ood_group_column: str,
) -> dict:
    merged = frame.merge(assignments, on="material_id", validate="one_to_one")
    if len(merged) != len(frame):
        raise RuntimeError("OOD split assignment does not cover all supervised rows.")
    expected = {"train", "val", "test"}
    if set(merged["split"].unique()) != expected:
        raise RuntimeError(
            f"OOD split must contain {sorted(expected)}; found "
            f"{sorted(merged['split'].unique())}."
        )

    test_groups = set(
        merged.loc[merged["split"] == "test", ood_group_column]
        .fillna("MISSING")
        .astype(str)
    )
    development_groups = set(
        merged.loc[merged["split"].isin(["train", "val"]), ood_group_column]
        .fillna("MISSING")
        .astype(str)
    )
    overlap = test_groups.intersection(development_groups)
    if overlap:
        raise RuntimeError(
            f"OOD leakage for {ood_group_column}: {list(overlap)[:10]}"
        )

    # Train/validation are split by reduced formula and must also be disjoint.
    train_formula = set(
        merged.loc[merged["split"] == "train", "reduced_formula"].astype(str)
    )
    val_formula = set(
        merged.loc[merged["split"] == "val", "reduced_formula"].astype(str)
    )
    formula_overlap = train_formula.intersection(val_formula)
    if formula_overlap:
        raise RuntimeError(
            f"Reduced-formula leakage between train and validation: "
            f"{list(formula_overlap)[:10]}"
        )

    summary = {}
    for split, part in merged.groupby("split"):
        summary[split] = {
            "n": int(len(part)),
            "ood_groups": int(part[ood_group_column].nunique(dropna=False)),
            "reduced_formulas": int(part["reduced_formula"].nunique()),
            **{f"{label}_positive": int(part[label].sum()) for label in LABELS},
            **{f"{label}_rate": float(part[label].mean()) for label in LABELS},
        }
    summary["ood_group_column"] = ood_group_column
    summary["test_development_group_overlap"] = 0
    return summary


def make_group_ood(
    frame: pd.DataFrame,
    group_column: str,
    seed: int,
    test_ratio: float,
    trials: int,
) -> pd.DataFrame:
    remain_idx, _test_idx = choose_group_holdout(
        frame, group_column, test_ratio, seed, trials
    )
    remain = frame.iloc[remain_idx].copy().reset_index(drop=True)
    train_idx, val_idx = choose_group_holdout(
        remain,
        "reduced_formula",
        0.15 / 0.85,
        seed + 1,
        trials,
    )
    assignments = pd.DataFrame(
        {"material_id": frame["material_id"], "split": "test"}
    )
    assignments.loc[
        assignments["material_id"].isin(remain.iloc[train_idx]["material_id"]),
        "split",
    ] = "train"
    assignments.loc[
        assignments["material_id"].isin(remain.iloc[val_idx]["material_id"]),
        "split",
    ] = "val"
    return assignments


def make_element_holdouts(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    all_elements = sorted(
        {
            element
            for system in frame["chemical_system"].astype(str)
            for element in system.split("-")
        }
    )
    summary = []
    for element_index, element in enumerate(all_elements):
        mask = frame["chemical_system"].str.split("-").map(
            lambda values: element in values
        )
        test = frame[mask]
        train_val = frame[~mask].reset_index(drop=True)
        if len(test) < 100 or any(test[label].nunique() < 2 for label in LABELS):
            continue
        train_idx, val_idx = choose_group_holdout(
            train_val,
            "reduced_formula",
            0.15 / 0.85,
            1000 + element_index,
            300,
        )
        assignments = pd.DataFrame(
            {"material_id": frame["material_id"], "split": "test"}
        )
        assignments.loc[
            assignments.material_id.isin(train_val.iloc[train_idx]["material_id"]),
            "split",
        ] = "train"
        assignments.loc[
            assignments.material_id.isin(train_val.iloc[val_idx]["material_id"]),
            "split",
        ] = "val"
        path = output / f"element_{element}.csv"
        assignments.to_csv(path, index=False)
        summary.append(
            {
                "element": element,
                "test_n": int(len(test)),
                "test_joint_rate": float(test["joint_label"].mean()),
                "path": str(path),
            }
        )
    result = pd.DataFrame(summary)
    result.to_csv(output / "element_holdout_summary.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "chemical_system",
            "structure_family",
            "layergroup",
            "elements",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()
    cfg = load_config()
    processed = Path(cfg["paths"]["processed_dir"])
    output = processed / "ood_splits"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(processed / "supervised.csv")
    modes = (
        ["chemical_system", "structure_family", "layergroup", "elements"]
        if args.mode == "all"
        else [args.mode]
    )
    reports = {}
    for index, mode in enumerate(modes):
        if mode == "elements":
            element_table = make_element_holdouts(frame, output)
            reports[mode] = {
                "holdout_count": int(len(element_table)),
                "summary_path": str(output / "element_holdout_summary.csv"),
            }
            continue
        assignments = make_group_ood(
            frame,
            mode,
            int(cfg["split"]["seed"]) + 100 + index,
            float(cfg["split"]["test_ratio"]),
            int(cfg["split"]["search_trials"]),
        )
        path = output / f"{mode}_holdout.csv"
        assignments.to_csv(path, index=False)
        reports[mode] = validate_ood_split(frame, assignments, mode)
    write_json(reports, output / "ood_summary.json")
    print(reports)


if __name__ == "__main__":
    main()
