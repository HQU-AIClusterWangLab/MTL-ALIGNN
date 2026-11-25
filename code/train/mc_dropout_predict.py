from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from models.uncertainty import aggregate_binary_probabilities
from train.common import (
    build_model,
    enable_mc_dropout,
    get_device,
    load_yaml,
    save_json,
    set_seed,
)
from train.dataset import CrystalGraphDataset, collate_graphs


TASKS = ["thermo", "dynamic", "joint"]


def move_batch(batch, device):
    (graph, line_graph, lattice), labels, metadata = batch
    return (
        graph.to(device),
        line_graph.to(device),
        lattice.to(device),
    ), labels.to(device), metadata


@torch.no_grad()
def stochastic_pass(model, loader, device):
    rows = []
    probabilities = {task: [] for task in TASKS}
    logits_by_task = {task: [] for task in TASKS}
    for batch in loader:
        graph_batch, labels, metadata = move_batch(batch, device)
        outputs = model(graph_batch)
        for task in TASKS:
            task_logits = outputs[task].detach().cpu().numpy()
            logits_by_task[task].extend(task_logits.tolist())
            probabilities[task].extend(
                torch.sigmoid(outputs[task]).detach().cpu().numpy().tolist()
            )
        if not rows:
            pass
        for index, item in enumerate(metadata):
            row = dict(item)
            label_values = labels[index].detach().cpu().numpy()
            row.update(
                {
                    "y_thermo": int(label_values[0]),
                    "y_dynamic": int(label_values[1]),
                    "y_joint": int(label_values[2]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), probabilities, logits_by_task


def evaluate_split(
    model,
    loader,
    device,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    base = None
    member_values = {task: [] for task in TASKS}
    member_logits = {task: [] for task in TASKS}
    for sample_index in trange(samples, desc="MC Dropout"):
        set_seed(seed + sample_index)
        model.eval()
        enable_mc_dropout(model)
        current, probabilities, logits_by_task = stochastic_pass(model, loader, device)
        if base is None:
            base = current
        elif current["material_id"].tolist() != base["material_id"].tolist():
            raise RuntimeError("MC Dropout sample order changed.")
        for task in TASKS:
            member_values[task].append(
                np.asarray(probabilities[task], dtype=float)
            )
            member_logits[task].append(
                np.asarray(logits_by_task[task], dtype=float)
            )

    result = base.copy()
    for task in TASKS:
        matrix = np.vstack(member_values[task])
        stats = aggregate_binary_probabilities(matrix)
        result[f"p_{task}_mean"] = stats["mean"]
        result[f"p_{task}_std"] = stats["std"]
        result[f"{task}_predictive_entropy"] = stats[
            "predictive_entropy"
        ]
        result[f"{task}_expected_entropy"] = stats["expected_entropy"]
        result[f"{task}_mutual_information"] = stats[
            "mutual_information"
        ]
        result[f"{task}_variation_ratio"] = stats["variation_ratio"]
        for index, values in enumerate(matrix, start=1):
            result[f"p_{task}_sample_{index}"] = values
            result[f"logit_{task}_sample_{index}"] = member_logits[task][index - 1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output = experiment_dir / "mc_dropout"
    completed = output / "COMPLETED.json"
    if completed.exists() and not args.force:
        print(f"SKIP completed: {completed}")
        return
    output.mkdir(parents=True, exist_ok=True)

    experiment = load_yaml(experiment_dir / "resolved_experiment.yaml")
    device = get_device(experiment["training"].get("device", "auto"))
    model = build_model(experiment).to(device)
    checkpoint_path = (
        experiment_dir / f"seed_{args.seed}" / "best_checkpoint.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    processed = Path(experiment["paths"]["processed_dir"])
    metadata = processed / "supervised.csv"
    split_file = processed / experiment.get("split_file", "iid_split.csv")
    batch_size = int(checkpoint.get("batch_size", experiment["training"]["batch_size"]))
    loader_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(experiment["training"].get("num_workers", 0)),
        "collate_fn": collate_graphs,
    }

    val_loader = DataLoader(
        CrystalGraphDataset(metadata, split_file, "val"),
        **loader_options,
    )
    test_loader = DataLoader(
        CrystalGraphDataset(metadata, split_file, "test"),
        **loader_options,
    )
    val = evaluate_split(
        model, val_loader, device, args.samples, args.seed + 10000
    )
    test = evaluate_split(
        model, test_loader, device, args.samples, args.seed + 20000
    )
    val.to_csv(output / "val_mc_predictions.csv", index=False)
    test.to_csv(output / "test_mc_predictions.csv", index=False)
    save_json(
        {
            "status": "completed",
            "seed": args.seed,
            "samples": args.samples,
            "checkpoint": str(checkpoint_path),
        },
        completed,
    )
    print(json.dumps(json.loads(completed.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
