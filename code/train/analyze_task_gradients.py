from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.common import build_model, get_device, load_yaml, save_json, set_seed
from train.dataset import CrystalGraphDataset, collate_graphs


TASKS = ["thermo", "dynamic", "joint"]
TASK_INDEX = {"thermo": 0, "dynamic": 1, "joint": 2}


def move_batch(batch, device):
    (graph, line_graph, lattice), labels, metadata = batch
    return (
        graph.to(device),
        line_graph.to(device),
        lattice.to(device),
    ), labels.to(device), metadata


def flatten_gradients(parameters) -> torch.Tensor:
    values = []
    for parameter in parameters:
        if parameter.grad is None:
            values.append(torch.zeros_like(parameter).reshape(-1))
        else:
            values.append(parameter.grad.detach().reshape(-1))
    return torch.cat(values)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.norm(left) * torch.linalg.norm(right)
    if denominator <= 1e-12:
        return float("nan")
    return float(torch.dot(left, right).cpu() / denominator.cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-experiment-dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.full_experiment_dir)
    output = root / "task_gradient_analysis"
    output.mkdir(parents=True, exist_ok=True)
    experiment = load_yaml(root / "resolved_experiment.yaml")
    device = get_device(experiment["training"].get("device", "auto"))
    set_seed(args.seed)

    model = build_model(experiment).to(device)
    checkpoint = torch.load(
        root / f"seed_{args.seed}" / "best_checkpoint.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    processed = Path(experiment["paths"]["processed_dir"])
    dataset = CrystalGraphDataset(
        processed / "supervised.csv",
        processed / experiment.get("split_file", "iid_split.csv"),
        "train",
    )
    loader = DataLoader(
        dataset,
        batch_size=min(
            8,
            int(checkpoint.get("batch_size", experiment["training"]["batch_size"])),
        ),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graphs,
    )
    positive_weights = checkpoint["positive_weights"]
    encoder_parameters = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    rows = []
    for batch_index, batch in enumerate(
        tqdm(loader, total=min(len(loader), args.max_batches))
    ):
        if batch_index >= args.max_batches:
            break
        graph_batch, labels, _ = move_batch(batch, device)
        logits = model(graph_batch)
        gradients = {}
        losses = {}
        for task in TASKS:
            model.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                logits[task],
                labels[:, TASK_INDEX[task]].float(),
                pos_weight=torch.tensor(
                    float(positive_weights[task]),
                    device=device,
                ),
            )
            loss.backward(retain_graph=True)
            gradients[task] = flatten_gradients(encoder_parameters).cpu()
            losses[task] = float(loss.detach().cpu())
        rows.append(
            {
                "batch": batch_index,
                "loss_thermo": losses["thermo"],
                "loss_dynamic": losses["dynamic"],
                "loss_joint": losses["joint"],
                "cosine_joint_thermo": cosine(
                    gradients["joint"], gradients["thermo"]
                ),
                "cosine_joint_dynamic": cosine(
                    gradients["joint"], gradients["dynamic"]
                ),
                "cosine_thermo_dynamic": cosine(
                    gradients["thermo"], gradients["dynamic"]
                ),
                "norm_thermo": float(torch.linalg.norm(gradients["thermo"])),
                "norm_dynamic": float(torch.linalg.norm(gradients["dynamic"])),
                "norm_joint": float(torch.linalg.norm(gradients["joint"])),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "gradient_cosines_by_batch.csv", index=False)
    summary = {}
    for column in [
        "cosine_joint_thermo",
        "cosine_joint_dynamic",
        "cosine_thermo_dynamic",
    ]:
        values = frame[column].dropna()
        summary[column] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=1)),
            "negative_fraction": float((values < 0).mean()),
            "n": int(len(values)),
        }
    save_json(summary, output / "gradient_conflict_summary.json")
    save_json(
        {"status": "completed", "batches": len(frame)},
        output / "COMPLETED.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
