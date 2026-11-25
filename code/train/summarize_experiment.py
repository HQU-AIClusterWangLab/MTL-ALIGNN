from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def flatten(prefix: str, value, out: dict):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, out)
    else:
        out[prefix] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    rows = []
    for path in sorted(root.glob("seed_*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"seed_dir": path.parent.name}
        flatten("", data, row)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No seed metrics found in {root}")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "seed_metrics_flat.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number])
    summary = pd.DataFrame(
        {
            "metric": numeric.columns,
            "mean": numeric.mean(axis=0).values,
            "std": numeric.std(axis=0, ddof=1).values,
            "min": numeric.min(axis=0).values,
            "max": numeric.max(axis=0).values,
            "n": numeric.notna().sum(axis=0).values,
        }
    )
    summary.to_csv(root / "metrics_mean_std.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
