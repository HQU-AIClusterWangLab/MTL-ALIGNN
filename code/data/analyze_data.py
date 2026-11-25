from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.utils import load_config


def main() -> None:
    cfg = load_config()
    processed = Path(cfg["paths"]["processed_dir"])
    frame = pd.read_csv(processed / "supervised.csv")
    split = pd.read_csv(processed / "iid_split.csv")
    merged = frame.merge(split, on="material_id", validate="one_to_one")

    tables = {}
    tables["labels"] = pd.DataFrame(
        [
            {
                "task": task,
                "positive": int(merged[f"{task}_label"].sum()),
                "negative": int((merged[f"{task}_label"] == 0).sum()),
                "positive_rate": float(merged[f"{task}_label"].mean()),
            }
            for task in ["thermo", "dynamic", "joint"]
        ]
    )
    tables["quadrants"] = (
        merged.groupby(["thermo_label", "dynamic_label"])
        .agg(
            n=("material_id", "size"),
            joint_positive=("joint_label", "sum"),
            ehull_mean=("ehull", "mean"),
            ehull_median=("ehull", "median"),
        )
        .reset_index()
    )
    tables["split"] = (
        merged.groupby("split")
        .agg(
            n=("material_id", "size"),
            thermo_positive=("thermo_label", "sum"),
            dynamic_positive=("dynamic_label", "sum"),
            joint_positive=("joint_label", "sum"),
            formulas=("reduced_formula", "nunique"),
            chemical_systems=("chemical_system", "nunique"),
        )
        .reset_index()
    )
    tables["spacegroups"] = (
        merged["spacegroup_symbol"]
        .fillna("Missing")
        .value_counts()
        .rename_axis("spacegroup_symbol")
        .reset_index(name="n")
    )
    tables["layergroups"] = (
        merged["layergroup"]
        .fillna("Missing")
        .value_counts()
        .rename_axis("layergroup")
        .reset_index(name="n")
    )

    output = processed / "data_analysis.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
