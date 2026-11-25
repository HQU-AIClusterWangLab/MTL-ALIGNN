from __future__ import annotations

from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_COLUMNS = ["thermo_label", "dynamic_label", "joint_label"]


class CrystalGraphDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        split_csv: str | Path | None = None,
        split: str | None = None,
        shuffle_auxiliary: bool = False,
        seed: int = 0,
        require_labels: bool = True,
    ) -> None:
        frame = pd.read_csv(metadata_csv)
        if split_csv is not None:
            assignments = pd.read_csv(split_csv)
            required_assignment_columns = {"material_id", "split"}
            missing_assignment_columns = required_assignment_columns - set(
                assignments.columns
            )
            if missing_assignment_columns:
                raise KeyError(
                    "Split manifest is missing required columns: "
                    f"{sorted(missing_assignment_columns)}"
                )
            if assignments["material_id"].duplicated().any():
                raise ValueError("Split manifest contains duplicate material_id rows.")
            frame = frame.merge(
                assignments[["material_id", "split"]],
                on="material_id",
                how="inner",
                validate="one_to_one",
            )
            if split is not None:
                frame = frame[frame["split"] == split].copy()

        frame = frame.sort_values("material_id").reset_index(drop=True)

        missing_graphs = frame[
            ~frame["graph_path"].map(lambda value: Path(value).exists())
        ]
        if not missing_graphs.empty:
            example = missing_graphs[["material_id", "graph_path"]].head()
            raise FileNotFoundError(
                f"{len(missing_graphs)} graph files are missing. "
                f"Examples:\n{example}"
            )

        self.frame = frame
        if require_labels:
            self.labels = frame[LABEL_COLUMNS].to_numpy(dtype=np.float32)
            if np.any(self.labels < 0):
                raise ValueError("Labeled dataset contains unknown labels.")
        else:
            self.labels = np.full((len(frame), 3), -1.0, dtype=np.float32)
            for index, column in enumerate(LABEL_COLUMNS):
                if column in frame:
                    self.labels[:, index] = frame[column].to_numpy(
                        dtype=np.float32
                    )

        if shuffle_auxiliary and split == "train":
            generator = np.random.default_rng(seed)
            self.labels[:, 0] = generator.permutation(self.labels[:, 0])
            self.labels[:, 1] = generator.permutation(self.labels[:, 1])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        graph_path = str(row["graph_path"])

        try:
            graphs, graph_labels = dgl.load_graphs(graph_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DGL graph bundle. "
                f"material_id={row['material_id']}, "
                f"row_id={row['row_id']}, "
                f"graph_path={graph_path}. "
                "The file may be truncated or from an incompatible format. "
                "Run data/build_graphs.py --subset <subset> --verify-existing "
                "to validate and repair existing graph files."
            ) from exc

        if len(graphs) != 2:
            raise RuntimeError(
                f"Expected graph and line graph in {graph_path}; "
                f"got {len(graphs)} graph object(s)."
            )

        graph, line_graph = graphs
        lattice = graph_labels.get("lattice")
        if lattice is None:
            lattice = torch.zeros((1, 3, 3), dtype=torch.float32)
        lattice = lattice.squeeze(0)

        labels = torch.tensor(self.labels[index], dtype=torch.float32)
        metadata = {
            "material_id": str(row["material_id"]),
            "unique_id": str(row.get("unique_id", "")),
            "row_id": int(row["row_id"]),
            "formula": str(row["formula"]),
            "reduced_formula": str(row["reduced_formula"]),
            "chemical_system": str(row["chemical_system"]),
            "structure_hash": str(row.get("structure_hash", "")),
            "dyn_stab_raw": str(row.get("dyn_stab_raw", "")),
            "screening_eligible": int(row.get("screening_eligible", -1)),
            "ehull": float(row["ehull"]),
            "spacegroup_symbol": str(row.get("spacegroup_symbol", "")),
            "spacegroup_number": row.get("spacegroup_number", np.nan),
            "layergroup": str(row.get("layergroup", "")),
            "structure_family": str(row.get("structure_family", "")),
            "graph_path": graph_path,
        }
        return graph, line_graph, lattice, labels, metadata


def collate_graphs(samples):
    graphs, line_graphs, lattices, labels, metadata = map(
        list, zip(*samples)
    )
    return (
        (
            dgl.batch(graphs),
            dgl.batch(line_graphs),
            torch.stack(lattices),
        ),
        torch.stack(labels),
        metadata,
    )
