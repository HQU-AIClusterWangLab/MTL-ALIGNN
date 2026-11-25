"""Validate the isolated five-seed rerun before any training starts."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(r"D:\ck\two")
HERE = Path(__file__).resolve().parent
SEEDS = [2025, 2026, 2027, 2028, 2029]
SOURCES = {
    "M04_single_alignn": ROOT / "train/experiments/main/M04_single_alignn.yaml",
    "M05_equal_mtl": ROOT / "train/experiments/main/M05_equal_mtl.yaml",
    "A03_no_logic": ROOT / "train/experiments/ablation/A03_no_logic.yaml",
    "A04_fixed_weight": ROOT / "train/experiments/ablation/A04_fixed_weight.yaml",
    "M06_full_mtl": ROOT / "train/experiments/main/M06_full_mtl.yaml",
}


def merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def main() -> None:
    project = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for name, source in SOURCES.items():
        original = merge(project, yaml.safe_load(source.read_text(encoding="utf-8")))
        proposed = merge(
            project, yaml.safe_load((HERE / f"{name}.yaml").read_text(encoding="utf-8"))
        )
        expected = copy.deepcopy(original)
        expected["name"] = name
        expected["category"] = "retuned_2026_11_22"
        expected["seeds"] = SEEDS
        expected["training"]["learning_rate"] = 3e-4
        expected["training"]["weight_decay"] = 1e-4
        expected["model"]["dropout"] = 0.0
        if proposed != expected:
            raise ValueError(f"{name}: unexpected change beyond tuning and isolation")
        print(f"OK {name}: five seeds, LR=3e-4, WD=1e-4, dropout=0")

    split = pd.read_csv(ROOT / "data/processed/iid_split.csv")
    counts = split["split"].value_counts().to_dict()
    if counts != {"train": 5671, "val": 1217, "test": 1214}:
        raise ValueError(f"Unexpected fixed split counts: {counts}")
    if split["material_id"].duplicated().any():
        raise ValueError("The fixed split contains duplicate material_id values")
    print(f"OK fixed split: {counts}")


if __name__ == "__main__":
    main()
