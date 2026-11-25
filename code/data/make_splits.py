from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from data.utils import load_config, write_json


LABELS = ["thermo_label", "dynamic_label", "joint_label"]


def score_holdout(
    full: pd.DataFrame,
    holdout: pd.DataFrame,
    ratio: float,
) -> float:
    size_term = abs(len(holdout) / len(full) - ratio)
    class_term = sum(
        abs(float(holdout[label].mean()) - float(full[label].mean()))
        for label in LABELS
    )
    return 4.0 * size_term + class_term


def choose_group_holdout(
    frame: pd.DataFrame,
    group_column: str,
    ratio: float,
    seed: int,
    trials: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=trials,
        test_size=ratio,
        random_state=seed,
    )
    best = None
    groups = frame[group_column].fillna("MISSING").astype(str)
    for train_idx, holdout_idx in splitter.split(frame, groups=groups):
        holdout = frame.iloc[holdout_idx]
        if any(holdout[label].nunique() < 2 for label in LABELS):
            continue
        candidate = score_holdout(frame, holdout, ratio)
        if best is None or candidate < best[0]:
            best = (candidate, train_idx, holdout_idx)
    if best is None:
        raise RuntimeError("Unable to generate a valid stratified group holdout.")
    return best[1], best[2]


def validate_split(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    group_column: str,
) -> dict:
    merged = frame.merge(assignments, on="material_id", validate="one_to_one")
    if len(merged) != len(frame):
        raise RuntimeError("Split assignment does not cover the supervised dataset.")
    group_sets = {
        split: set(part[group_column].astype(str))
        for split, part in merged.groupby("split")
    }
    names = ["train", "val", "test"]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = group_sets[left].intersection(group_sets[right])
            if overlap:
                raise RuntimeError(
                    f"Group leakage between {left} and {right}: "
                    f"{list(overlap)[:5]}"
                )
    summary = {}
    for split, part in merged.groupby("split"):
        summary[split] = {
            "n": int(len(part)),
            "groups": int(part[group_column].nunique()),
            **{
                f"{label}_positive": int(part[label].sum())
                for label in LABELS
            },
            **{
                f"{label}_rate": float(part[label].mean())
                for label in LABELS
            },
        }
    return summary


def main() -> None:
    cfg = load_config()
    processed = Path(cfg["paths"]["processed_dir"])
    frame = pd.read_csv(processed / "supervised.csv")
    split_cfg = cfg["split"]
    group_column = split_cfg["group_column"]
    seed = int(split_cfg["seed"])
    trials = int(split_cfg["search_trials"])

    train_val_idx, test_idx = choose_group_holdout(
        frame,
        group_column,
        float(split_cfg["test_ratio"]),
        seed,
        trials,
    )
    train_val = frame.iloc[train_val_idx].copy().reset_index(drop=True)
    relative_val = float(split_cfg["val_ratio"]) / (
        float(split_cfg["train_ratio"]) + float(split_cfg["val_ratio"])
    )
    train_idx, val_idx = choose_group_holdout(
        train_val,
        group_column,
        relative_val,
        seed + 1,
        trials,
    )

    assignments = pd.DataFrame(
        {"material_id": frame["material_id"], "split": "test"}
    )
    train_ids = set(train_val.iloc[train_idx]["material_id"])
    val_ids = set(train_val.iloc[val_idx]["material_id"])
    assignments.loc[
        assignments["material_id"].isin(train_ids), "split"
    ] = "train"
    assignments.loc[
        assignments["material_id"].isin(val_ids), "split"
    ] = "val"

    summary = validate_split(frame, assignments, group_column)
    assignments.to_csv(processed / "iid_split.csv", index=False)
    write_json(summary, processed / "iid_split_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
