from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train.common import (
    build_model,
    build_objective,
    configure_runtime,
    get_device,
    load_yaml,
    resolve_experiment,
    set_seed,
)
from train.dataset import CrystalGraphDataset, collate_graphs


TASKS = ["thermo", "dynamic", "joint"]


def json_default(value: Any) -> Any:
    """Convert scientific-Python objects to JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )



def tensor_report(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    finite = torch.isfinite(value)
    report: dict[str, Any] = {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": bool(finite.all().item()),
        "nonfinite_count": int((~finite).sum().item()),
    }
    if finite.any():
        finite_value = value[finite].float()
        report.update(
            {
                "finite_min": float(finite_value.min().cpu()),
                "finite_max": float(finite_value.max().cpu()),
                "finite_mean": float(finite_value.mean().cpu()),
            }
        )
    return report


def graph_tensor_reports(
    graph: dgl.DGLGraph,
    prefix: str,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for key, value in graph.ndata.items():
        if torch.is_tensor(value):
            reports.append(tensor_report(f"{prefix}.ndata[{key!r}]", value))
    for key, value in graph.edata.items():
        if torch.is_tensor(value):
            reports.append(tensor_report(f"{prefix}.edata[{key!r}]", value))
    return reports


def all_finite(mapping: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all().item() for value in mapping.values())


def output_reports(outputs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        tensor_report(f"logits.{task}", value)
        for task, value in outputs.items()
    ]


def evaluate_batch(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    graph_batch,
    labels: torch.Tensor,
    use_amp: bool,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    objective.zero_grad(set_to_none=True)
    context = torch.cuda.amp.autocast(enabled=use_amp)
    with context:
        outputs = model(graph_batch)
        loss, diagnostics = objective(outputs, labels)
    result = {
        "amp": use_amp,
        "loss": float(loss.detach().float().cpu())
        if torch.isfinite(loss)
        else str(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "outputs": output_reports(outputs),
        "diagnostics": {
            key: (
                float(value.detach().float().cpu())
                if torch.isfinite(value).all()
                else str(value.detach().cpu())
            )
            for key, value in diagnostics.items()
        },
    }
    if torch.isfinite(loss):
        loss.backward()
        gradients = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                gradients.append(tensor_report(f"grad.{name}", parameter.grad))
        result["gradients_finite"] = all(x["finite"] for x in gradients)
        result["first_nonfinite_gradients"] = [
            x for x in gradients if not x["finite"]
        ][:20]
    else:
        result["gradients_finite"] = None
        result["first_nonfinite_gradients"] = []
    return result


def isolate_samples(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    graph_batch,
    labels: torch.Tensor,
    metadata: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    graph, line_graph, lattice = graph_batch
    graphs = dgl.unbatch(graph)
    line_graphs = dgl.unbatch(line_graph)
    reports = []
    for index, item in enumerate(metadata):
        single_batch = (
            graphs[index].to(device),
            line_graphs[index].to(device),
            lattice[index : index + 1].to(device),
        )
        single_labels = labels[index : index + 1]
        try:
            fp32 = evaluate_batch(
                model, objective, single_batch, single_labels, use_amp=False
            )
        except Exception as exc:
            fp32 = {"exception": f"{type(exc).__name__}: {exc}"}
        try:
            amp = evaluate_batch(
                model, objective, single_batch, single_labels, use_amp=True
            )
        except Exception as exc:
            amp = {"exception": f"{type(exc).__name__}: {exc}"}
        reports.append(
            {
                "material_id": item.get("material_id"),
                "row_id": item.get("row_id"),
                "formula": item.get("formula"),
                "graph_path": item.get("graph_path"),
                "labels": labels[index].detach().cpu().tolist(),
                "fp32": fp32,
                "amp": amp,
            }
        )
    return reports


def scan_graph_files(dataset: CrystalGraphDataset) -> list[dict[str, Any]]:
    failures = []
    for index in range(len(dataset)):
        graph, line_graph, lattice, labels, metadata = dataset[index]
        reports = (
            graph_tensor_reports(graph, "graph")
            + graph_tensor_reports(line_graph, "line_graph")
            + [tensor_report("lattice", lattice), tensor_report("labels", labels)]
        )
        bad = [item for item in reports if not item["finite"]]
        if bad:
            failures.append(
                {
                    **metadata,
                    "bad_tensors": bad,
                }
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default="train/experiments/main/M04_single_alignn.yaml",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Scan every training graph tensor for NaN/Inf.",
    )
    args = parser.parse_args()

    project = load_yaml("config.yaml")
    experiment = resolve_experiment(project, load_yaml(args.experiment))
    seed = int(args.seed or experiment.get("seeds", [2025])[0])
    set_seed(seed)
    configure_runtime(experiment)
    device = get_device(experiment["training"].get("device", "auto"))

    processed = Path(experiment["paths"]["processed_dir"])
    metadata_csv = processed / "supervised.csv"
    split_csv = processed / experiment.get("split_file", "iid_split.csv")
    dataset = CrystalGraphDataset(
        metadata_csv,
        split_csv,
        split="train",
    )

    batch_size = int(experiment["training"]["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_graphs,
        pin_memory=False,
        drop_last=False,
    )

    # Match train_gnn.py RNG consumption: construct model/objective before
    # requesting the first shuffled batch.
    model = build_model(experiment).to(device)
    objective = build_objective(
        experiment,
        {task: 1.0 for task in experiment["active_tasks"]},
    ).to(device)

    (graph, line_graph, lattice), labels, metadata = next(iter(loader))
    graph_batch = (
        graph.to(device),
        line_graph.to(device),
        lattice.to(device),
    )
    labels = labels.to(device)

    input_reports = (
        graph_tensor_reports(graph_batch[0], "graph")
        + graph_tensor_reports(graph_batch[1], "line_graph")
        + [
            tensor_report("lattice", graph_batch[2]),
            tensor_report("labels", labels),
        ]
    )

    report: dict[str, Any] = {
        "experiment": args.experiment,
        "seed": seed,
        "device": str(device),
        "batch_size": batch_size,
        "materials": metadata,
        "input_tensors": input_reports,
        "input_all_finite": all(x["finite"] for x in input_reports),
    }

    try:
        report["fp32"] = evaluate_batch(
            model, objective, graph_batch, labels, use_amp=False
        )
    except Exception as exc:
        report["fp32"] = {"exception": f"{type(exc).__name__}: {exc}"}

    # Rebuild model/objective so AMP and FP32 tests use identical initialization
    # without contamination from the preceding backward pass.
    set_seed(seed)
    model_amp = build_model(experiment).to(device)
    objective_amp = build_objective(
        experiment,
        {task: 1.0 for task in experiment["active_tasks"]},
    ).to(device)
    try:
        report["amp"] = evaluate_batch(
            model_amp, objective_amp, graph_batch, labels, use_amp=True
        )
    except Exception as exc:
        report["amp"] = {"exception": f"{type(exc).__name__}: {exc}"}

    fp32_finite = bool(report.get("fp32", {}).get("loss_finite", False))
    amp_finite = bool(report.get("amp", {}).get("loss_finite", False))

    if not fp32_finite or not amp_finite:
        # Use a fresh model for per-sample isolation.
        set_seed(seed)
        isolate_model = build_model(experiment).to(device)
        isolate_objective = build_objective(
            experiment,
            {task: 1.0 for task in experiment["active_tasks"]},
        ).to(device)
        report["sample_isolation"] = isolate_samples(
            isolate_model,
            isolate_objective,
            graph_batch,
            labels,
            metadata,
            device,
        )

    if args.scan_all:
        report["all_graph_nonfinite_failures"] = scan_graph_files(dataset)
        report["all_graph_nonfinite_failure_count"] = len(
            report["all_graph_nonfinite_failures"]
        )

    out_dir = Path(experiment["paths"]["results_dir"]) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"nan_diagnostic_{Path(args.experiment).stem}_seed{seed}.json"
    out_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print(f"Diagnostic report: {out_path}")
    print(f"Input tensors finite: {report['input_all_finite']}")
    print(f"FP32 loss finite: {fp32_finite}")
    print(f"AMP loss finite: {amp_finite}")

    if fp32_finite and not amp_finite:
        print(
            "Diagnosis: mixed-precision overflow/instability. "
            "Keep training.amp=false for ALIGNN."
        )
    elif not fp32_finite:
        print(
            "Diagnosis: the failure also occurs in FP32. Inspect "
            "sample_isolation and input tensors in the JSON report."
        )
    elif fp32_finite and amp_finite:
        print(
            "The reproduced first batch is finite in both modes. "
            "Run again with --scan-all to inspect all graph tensors."
        )


if __name__ == "__main__":
    main()
