from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train.common import (
    build_model,
    build_objective,
    get_device,
    load_yaml,
    resolve_experiment,
)
from train.dataset import CrystalGraphDataset, collate_graphs


def _build_smoke_metadata(processed: Path, sample_count: int = 2) -> Path:
    """Create a tiny metadata file containing only graphs already built.

    The smoke stage intentionally builds only eight graphs. Therefore the
    forward/backward smoke test must not instantiate the full training split,
    which would require every supervised graph to exist.
    """
    supervised_path = processed / "supervised.csv"
    if not supervised_path.exists():
        raise FileNotFoundError(
            f"Missing {supervised_path}. Run the metadata preparation step first."
        )

    frame = pd.read_csv(supervised_path)
    if "graph_path" not in frame.columns:
        raise KeyError(f"'graph_path' is missing from {supervised_path}.")

    available = frame[
        frame["graph_path"].map(lambda value: Path(str(value)).exists())
    ].copy()

    if len(available) < sample_count:
        raise FileNotFoundError(
            f"Only {len(available)} built graph file(s) were found, but "
            f"{sample_count} are required for the smoke test. Run:\n"
            "python data/build_graphs.py --subset smoke"
        )

    smoke_path = processed / "smoke_test_metadata.csv"
    available.head(sample_count).to_csv(smoke_path, index=False)
    return smoke_path


def main() -> None:
    project = load_yaml("config.yaml")
    experiment_file = load_yaml(
        "train/experiments/main/M06_full_mtl.yaml"
    )
    experiment = resolve_experiment(project, experiment_file)
    processed = Path(experiment["paths"]["processed_dir"])

    # Use only the small graph subset created by run_smoke.ps1.
    smoke_metadata = _build_smoke_metadata(processed, sample_count=2)
    dataset = CrystalGraphDataset(
        smoke_metadata,
        split_csv=None,
        split=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graphs,
    )
    (graph, line_graph, lattice), labels, metadata = next(iter(loader))

    device = get_device("auto")
    graph_batch = (
        graph.to(device),
        line_graph.to(device),
        lattice.to(device),
    )
    labels = labels.to(device)

    model = build_model(experiment).to(device)
    positive_weights = {
        "thermo": 1.0,
        "dynamic": 1.0,
        "joint": 1.0,
    }
    objective = build_objective(experiment, positive_weights).to(device)

    model.train()
    outputs = model(graph_batch)
    loss, diagnostics = objective(outputs, labels)
    loss.backward()

    print(
        {
            "device": str(device),
            "smoke_metadata": str(smoke_metadata),
            "materials": [item["material_id"] for item in metadata],
            "graph_nodes": graph.num_nodes(),
            "graph_edges": graph.num_edges(),
            "line_graph_nodes": line_graph.num_nodes(),
            "line_graph_edges": line_graph.num_edges(),
            "output_shapes": {
                task: tuple(value.shape) for task, value in outputs.items()
            },
            "loss": float(loss.detach().cpu()),
            "diagnostics": {
                key: float(value.cpu()) for key, value in diagnostics.items()
            },
        }
    )


if __name__ == "__main__":
    main()
