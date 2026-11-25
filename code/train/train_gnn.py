from __future__ import annotations

import argparse
import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.common import (
    build_model,
    build_objective,
    capture_environment,
    completed,
    configure_runtime,
    experiment_result_root,
    get_device,
    load_yaml,
    resolve_experiment,
    save_json,
    save_yaml,
    set_seed,
)
from train.dataset import CrystalGraphDataset, collate_graphs
from train.metrics import (
    best_mcc_threshold,
    classification_metrics,
)


TASKS = ["thermo", "dynamic", "joint"]
TASK_INDEX = {task: index for index, task in enumerate(TASKS)}


def positive_weights(
    dataset: CrystalGraphDataset,
    active_tasks: list[str],
) -> dict[str, float]:
    result = {}
    for task in active_tasks:
        values = dataset.labels[:, TASK_INDEX[task]]
        positive = float((values == 1).sum())
        negative = float((values == 0).sum())
        if positive == 0 or negative == 0:
            raise RuntimeError(f"Task {task} has only one class in training.")
        result[task] = negative / positive
    return result


def make_loader(
    dataset: CrystalGraphDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_graphs,
        pin_memory=False,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def fit_conditioning_stats(
    dataset: CrystalGraphDataset,
    features: list[str],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for feature in features:
        if feature not in dataset.frame.columns:
            raise KeyError(f"Conditioning feature {feature!r} is missing from metadata.")
        values = pd.to_numeric(dataset.frame[feature], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Conditioning feature {feature!r} contains missing/non-finite values."
            )
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        stats[feature] = {"mean": mean, "std": std if std > 1e-12 else 1.0}
    return stats


def move_batch(
    batch,
    device: torch.device,
    conditioning_features: list[str] | None = None,
    conditioning_stats: dict[str, dict[str, float]] | None = None,
):
    (graph, line_graph, lattice), labels, metadata = batch
    graph_parts: list[torch.Tensor | object] = [
        graph.to(device),
        line_graph.to(device),
        lattice.to(device),
    ]
    features = list(conditioning_features or [])
    if features:
        if conditioning_stats is None:
            raise ValueError("Conditioning statistics are required.")
        rows = []
        for item in metadata:
            rows.append(
                [
                    (
                        float(item[feature])
                        - float(conditioning_stats[feature]["mean"])
                    )
                    / float(conditioning_stats[feature]["std"])
                    for feature in features
                ]
            )
        conditioning = torch.tensor(rows, dtype=torch.float32, device=device)
        graph_parts.append(conditioning)
    return tuple(graph_parts), labels.to(device), metadata


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    conditioning_features: list[str] | None = None,
    conditioning_stats: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        graph_batch, labels, metadata = move_batch(
            batch,
            device,
            conditioning_features,
            conditioning_stats,
        )
        outputs = model(graph_batch)
        labels_np = labels.cpu().numpy()
        for index, item in enumerate(metadata):
            row = dict(item)
            row.update(
                {
                    "y_thermo": int(labels_np[index, 0]),
                    "y_dynamic": int(labels_np[index, 1]),
                    "y_joint": int(labels_np[index, 2]),
                }
            )
            for task, task_logits in outputs.items():
                if task == "state_logits":
                    for state_index, state_name in enumerate(("00", "01", "10", "11")):
                        row[f"logit_state_{state_name}"] = float(
                            task_logits[index, state_index].detach().cpu()
                        )
                    continue
                value = float(task_logits[index].detach().cpu())
                row[f"logit_{task}"] = value
                row[f"p_{task}"] = float(
                    torch.sigmoid(task_logits[index]).detach().cpu()
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values("material_id").reset_index(drop=True)


def validation_score(predictions: pd.DataFrame, task: str = "joint") -> float:
    from sklearn.metrics import average_precision_score

    return float(
        average_precision_score(
            predictions[f"y_{task}"],
            predictions[f"p_{task}"],
        )
    )


def save_last_checkpoint(
    path: Path,
    *,
    epoch: int,
    model,
    objective,
    optimizer,
    scheduler,
    scaler,
    best_score: float,
    stale: int,
    history: list[dict],
    batch_size: int,
) -> None:
    atomic_torch_save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "stale": stale,
            "history": history,
            "batch_size": batch_size,
            "rng_state": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        },
        path,
    )


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint atomically so an interruption cannot corrupt it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_seed(
    experiment: dict,
    seed: int,
    output_dir: Path,
    batch_size: int,
    force: bool,
) -> dict:
    if completed(output_dir) and not force:
        print(f"SKIP completed: {output_dir}")
        return load_yaml(output_dir / "summary.yaml")

    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        # Invalidate the old run before doing any work. Otherwise an interrupted
        # forced rerun would still look completed to the next invocation.
        completion_marker = output_dir / "COMPLETED.json"
        if completion_marker.exists():
            completion_marker.unlink()
    set_seed(seed)
    configure_runtime(experiment)
    device = get_device(experiment["training"].get("device", "auto"))
    processed = Path(experiment["paths"]["processed_dir"])
    metadata_csv = processed / "supervised.csv"
    split_csv = processed / experiment.get("split_file", "iid_split.csv")
    shuffle_auxiliary = bool(experiment.get("shuffle_auxiliary", False))

    train_dataset = CrystalGraphDataset(
        metadata_csv,
        split_csv,
        split="train",
        shuffle_auxiliary=shuffle_auxiliary,
        seed=seed,
    )
    val_dataset = CrystalGraphDataset(
        metadata_csv,
        split_csv,
        split="val",
    )
    test_dataset = CrystalGraphDataset(
        metadata_csv,
        split_csv,
        split="test",
    )
    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        raise RuntimeError("At least one data split is empty.")

    conditioning_features = list(
        experiment.get("model", {}).get("conditioning_features", [])
    )
    conditioning_stats = fit_conditioning_stats(
        train_dataset, conditioning_features
    )
    if conditioning_features:
        save_json(conditioning_stats, output_dir / "conditioning_stats.json")

    workers = int(experiment["training"].get("num_workers", 0))
    train_loader = make_loader(
        train_dataset, batch_size, True, device, workers
    )
    val_loader = make_loader(
        val_dataset, batch_size, False, device, workers
    )
    test_loader = make_loader(
        test_dataset, batch_size, False, device, workers
    )

    weighting_mode = str(
        experiment["training"].get("class_weighting", "none")
    ).lower()
    if weighting_mode == "balanced":
        weights = positive_weights(
            train_dataset, list(experiment["active_tasks"])
        )
    elif weighting_mode == "none":
        weights = {task: 1.0 for task in experiment["active_tasks"]}
    else:
        raise ValueError(
            "training.class_weighting must be 'none' or 'balanced'."
        )
    model = build_model(experiment).to(device)
    monitor_task = str(experiment.get("monitor_task", "joint"))
    if monitor_task not in model.prediction_tasks:
        raise ValueError(
            f"monitor_task={monitor_task!r} is not one of the model heads "
            f"{list(experiment['heads'])}."
        )
    pretrained_report = None
    pretrained_path = experiment.get("pretrained_backbone_path")
    if pretrained_path:
        from models.pretrained import load_alignn_backbone

        pretrained_report = load_alignn_backbone(model, pretrained_path)
        save_json(pretrained_report, output_dir / "pretrained_load_report.json")
    objective = build_objective(experiment, weights).to(device)
    model_parameters = list(model.parameters())
    objective_parameters = list(objective.parameters())
    parameters = model_parameters + objective_parameters
    parameter_groups = [
        {
            "params": model_parameters,
            "weight_decay": float(experiment["training"]["weight_decay"]),
        }
    ]
    if objective_parameters:
        # Do not apply L2 regularization to global task log-variance parameters.
        parameter_groups.append(
            {"params": objective_parameters, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(experiment["training"]["learning_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(experiment["training"]["scheduler_factor"]),
        patience=int(experiment["training"]["scheduler_patience"]),
        min_lr=1e-6,
    )
    amp_enabled = (
        bool(experiment["training"].get("amp", True))
        and device.type == "cuda"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_path = output_dir / "best_checkpoint.pt"
    last_path = output_dir / "last_checkpoint.pt"
    history: list[dict] = []
    best_score = -math.inf
    stale = 0
    start_epoch = 1

    if last_path.exists() and not force:
        state = torch.load(last_path, map_location=device)
        if int(state.get("batch_size", batch_size)) != batch_size:
            raise RuntimeError(
                "Resume checkpoint batch size does not match current batch size."
            )
        model.load_state_dict(state["model"])
        objective.load_state_dict(state["objective"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = list(state.get("history", []))
        best_score = float(state.get("best_score", -math.inf))
        stale = int(state.get("stale", 0))
        start_epoch = int(state["epoch"]) + 1
        if "rng_state" in state:
            rng = state["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng["cuda"]:
                torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        print(f"Resume {output_dir} from epoch {start_epoch}")

    start_time = time.time()
    maximum_epochs = int(experiment["training"]["max_epochs"])
    patience = int(experiment["training"]["patience"])
    gradient_clip = float(
        experiment["training"]["gradient_clip_norm"]
    )

    for epoch in range(start_epoch, maximum_epochs + 1):
        if stale >= patience:
            print("Resumed checkpoint already reached early stopping; finalizing.")
            break
        model.train()
        objective.train()
        total_loss = 0.0
        diagnostic_totals: dict[str, float] = {}
        for batch in tqdm(
            train_loader,
            desc=f"{experiment['name']} seed={seed} epoch={epoch}",
            leave=False,
        ):
            graph_batch, labels, _metadata = move_batch(
                batch,
                device,
                conditioning_features,
                conditioning_stats,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(graph_batch)
                loss, diagnostics = objective(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}: {loss}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            for key, value in diagnostics.items():
                diagnostic_totals[key] = (
                    diagnostic_totals.get(key, 0.0)
                    + float(value.cpu())
                )

        val_predictions = predict(
            model,
            val_loader,
            device,
            conditioning_features,
            conditioning_stats,
        )
        score = validation_score(val_predictions, monitor_task)
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            f"val_{monitor_task}_pr_auc": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": batch_size,
        }
        for key, value in diagnostic_totals.items():
            record[key] = value / max(len(train_loader), 1)
        history.append(record)
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
        )
        print(record)

        if score > best_score + 1e-7:
            best_score = score
            stale = 0
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "objective": objective.state_dict(),
                    "epoch": epoch,
                    "monitor_task": monitor_task,
                    "val_monitor_pr_auc": score,
                    **(
                        {"val_joint_pr_auc": score}
                        if monitor_task == "joint"
                        else {}
                    ),
                    "seed": seed,
                    "positive_weights": weights,
                    "batch_size": batch_size,
                    "heads": experiment["heads"],
                    "model_type": experiment.get("model_type", "alignn"),
                    "output_mode": model.output_mode,
                    "conditioning_features": conditioning_features,
                    "conditioning_stats": conditioning_stats,
                },
                best_path,
            )
        else:
            stale += 1

        save_last_checkpoint(
            last_path,
            epoch=epoch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_score=best_score,
            stale=stale,
            history=history,
            batch_size=batch_size,
        )

        if stale >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was produced.")
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"])
    objective.load_state_dict(best["objective"])

    val_predictions = predict(
        model,
        val_loader,
        device,
        conditioning_features,
        conditioning_stats,
    )
    test_predictions = predict(
        model,
        test_loader,
        device,
        conditioning_features,
        conditioning_stats,
    )
    thresholds = {}
    metrics = {
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "monitor_task": monitor_task,
        "best_val_monitor_pr_auc": float(
            best.get("val_monitor_pr_auc", best.get("val_joint_pr_auc"))
        ),
        "batch_size": batch_size,
        "runtime_seconds": time.time() - start_time,
        "device": str(device),
        "pretrained_backbone": pretrained_report,
        "conditioning_features": conditioning_features,
        "conditioning_stats": conditioning_stats,
        "output_mode": model.output_mode,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    for task in model.prediction_tasks:
        threshold = best_mcc_threshold(
            val_predictions[f"y_{task}"],
            val_predictions[f"p_{task}"],
        )
        thresholds[task] = threshold
        metrics[f"val_{task}"] = classification_metrics(
            val_predictions[f"y_{task}"],
            val_predictions[f"p_{task}"],
            threshold,
            int(experiment["uncertainty"]["calibration_bins"]),
        )
        metrics[f"test_{task}"] = classification_metrics(
            test_predictions[f"y_{task}"],
            test_predictions[f"p_{task}"],
            threshold,
            int(experiment["uncertainty"]["calibration_bins"]),
        )
        val_predictions[f"pred_{task}"] = (
            val_predictions[f"p_{task}"] >= threshold
        ).astype(int)
        test_predictions[f"pred_{task}"] = (
            test_predictions[f"p_{task}"] >= threshold
        ).astype(int)

    val_predictions.to_csv(
        output_dir / "val_predictions.csv",
        index=False,
    )
    test_predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )
    save_json(thresholds, output_dir / "thresholds.json")
    save_json(metrics, output_dir / "metrics.json")
    save_yaml(experiment, output_dir / "resolved_experiment.yaml")
    save_json(capture_environment(), output_dir / "environment.json")
    summary = {
        "name": experiment["name"],
        "category": experiment["category"],
        "seed": seed,
        "status": "completed",
        "best_epoch": int(best["epoch"]),
        "batch_size": batch_size,
        "monitor_task": monitor_task,
        f"test_{monitor_task}_pr_auc": metrics[f"test_{monitor_task}"]["pr_auc"],
        f"test_{monitor_task}_mcc": metrics[f"test_{monitor_task}"]["mcc"],
    }
    save_yaml(summary, output_dir / "summary.yaml")
    save_json(summary, output_dir / "COMPLETED.json")
    if last_path.exists():
        last_path.unlink()
    return summary


def train_with_batch_fallback(
    experiment: dict,
    seed: int,
    output_dir: Path,
    force: bool,
) -> dict:
    configured = int(experiment["training"]["batch_size"])
    minimum = int(experiment["training"]["minimum_batch_size"])
    candidates = []
    resume_path = output_dir / "last_checkpoint.pt"
    if resume_path.exists() and not force:
        resume_state = torch.load(resume_path, map_location="cpu")
        configured = int(resume_state.get("batch_size", configured))
        print(f"Detected resumable batch_size={configured} in {resume_path}")
    current = configured
    while current >= minimum:
        if current not in candidates:
            candidates.append(current)
        current //= 2
    if minimum not in candidates:
        candidates.append(minimum)

    for index, batch_size in enumerate(candidates):
        try:
            return train_seed(
                experiment,
                seed,
                output_dir,
                batch_size,
                force=force,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            gc.collect()
            (output_dir / f"OOM_batch_{batch_size}.txt").write_text(
                repr(exc),
                encoding="utf-8",
            )
            for path in (
                output_dir / "last_checkpoint.pt",
                output_dir / "best_checkpoint.pt",
            ):
                if path.exists():
                    path.unlink()
            if index == len(candidates) - 1:
                raise
            print(
                f"CUDA OOM with batch_size={batch_size}; "
                f"retrying with {candidates[index + 1]}."
            )
    raise RuntimeError("No batch size candidate succeeded.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project = load_yaml("config.yaml")
    experiment_file = load_yaml(args.experiment)
    experiment_file["experiment_file"] = args.experiment
    experiment = resolve_experiment(project, experiment_file)
    output_root = experiment_result_root(experiment)
    output_root.mkdir(parents=True, exist_ok=True)
    save_yaml(experiment, output_root / "resolved_experiment.yaml")

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(value) for value in experiment["seeds"]]

    summaries = []
    for seed in seeds:
        summaries.append(
            train_with_batch_fallback(
                experiment,
                seed,
                output_root / f"seed_{seed}",
                force=args.force,
            )
        )
    save_json(summaries, output_root / "all_seed_summaries.json")


if __name__ == "__main__":
    main()
