from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from alignn.graphs import Graph
from ase.db import connect
from dgl import load_graphs, save_graphs
from jarvis.core.atoms import ase_to_atoms
from tqdm import tqdm

from data.utils import load_config


def select_frame(processed: Path, subset: str) -> pd.DataFrame:
    if subset == "supervised":
        return pd.read_csv(processed / "supervised.csv")
    if subset == "unknown":
        return pd.read_csv(processed / "unknown_pool.csv")
    if subset == "eligible":
        return pd.read_csv(processed / "unknown_thermo_eligible.csv")
    if subset == "smoke":
        return pd.read_csv(processed / "supervised.csv").head(8)
    raise ValueError(subset)


def validate_graph_file(path: Path) -> tuple[bool, str]:
    """Check that an existing file is a readable DGL graph bundle.

    This catches truncated/stale .bin files that merely exist on disk but
    cannot be deserialized (e.g. DGL 'Invalid Magic Number').
    """
    if not path.exists():
        return False, "missing"
    if path.stat().st_size <= 0:
        return False, "empty_file"
    try:
        graphs, graph_labels = load_graphs(str(path))
        if len(graphs) != 2:
            return False, f"expected_2_graphs_got_{len(graphs)}"
        lattice = graph_labels.get("lattice")
        if lattice is None:
            return False, "missing_lattice"
        if not torch.isfinite(lattice).all().item():
            return False, "nonfinite_lattice"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def save_graph_bundle_atomic(
    target: Path,
    graph,
    line_graph,
    lattice: torch.Tensor,
) -> None:
    """Write to a temporary file, validate it, then atomically replace target."""
    temp = target.with_name(target.name + ".part")
    if temp.exists():
        temp.unlink()

    try:
        save_graphs(str(temp), [graph, line_graph], {"lattice": lattice})
        valid, reason = validate_graph_file(temp)
        if not valid:
            raise RuntimeError(
                f"Newly written graph bundle failed validation: {reason}"
            )
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()


def build_graph(
    row,
    db,
    cfg,
    overwrite: bool,
    verify_existing: bool,
) -> tuple[str, str | None]:
    target = Path(row.graph_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    repair_reason = None
    if target.exists() and not overwrite:
        if not verify_existing:
            return "skipped", None
        valid, reason = validate_graph_file(target)
        if valid:
            return "skipped", None
        repair_reason = reason

    ase_atoms = db.get(id=int(row.row_id)).toatoms()
    jarvis_atoms = ase_to_atoms(ase_atoms)
    graph_cfg = cfg["graph"]

    result = Graph.atom_dgl_multigraph(
        atoms=jarvis_atoms,
        neighbor_strategy=graph_cfg["neighbor_strategy"],
        cutoff=float(graph_cfg["cutoff"]),
        cutoff_extra=float(graph_cfg["cutoff_extra"]),
        max_neighbors=int(graph_cfg["max_neighbors"]),
        atom_features=graph_cfg["atom_features"],
        use_canonize=bool(graph_cfg["use_canonize"]),
        compute_line_graph=True,
        id=str(row.material_id),
        dtype=graph_cfg.get("dtype", "float32"),
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "Graph.atom_dgl_multigraph did not return (graph, line_graph)."
        )

    graph, line_graph = result
    lattice = torch.tensor(
        ase_atoms.cell.array, dtype=torch.float32
    ).unsqueeze(0)

    save_graph_bundle_atomic(
        target=target,
        graph=graph,
        line_graph=line_graph,
        lattice=lattice,
    )

    if repair_reason is not None:
        return "repaired", repair_reason
    return "built", None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=["supervised", "unknown", "eligible", "smoke"],
        default="supervised",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Validate existing DGL .bin files before skipping them. "
            "Unreadable/corrupt files are rebuilt automatically."
        ),
    )
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    processed = Path(cfg["paths"]["processed_dir"])
    frame = select_frame(processed, args.subset)
    db = connect(cfg["paths"]["db_path"])

    failures = []
    repairs = []
    counts = {"built": 0, "repaired": 0, "skipped": 0}

    for i, row in enumerate(
        tqdm(
            frame.itertuples(index=False),
            total=len(frame),
            desc=f"graphs:{args.subset}",
        ),
        start=1,
    ):
        try:
            status, repair_reason = build_graph(
                row=row,
                db=db,
                cfg=cfg,
                overwrite=args.overwrite,
                verify_existing=args.verify_existing,
            )
            counts[status] += 1
            if status == "repaired":
                repairs.append(
                    {
                        "row_id": row.row_id,
                        "material_id": row.material_id,
                        "graph_path": row.graph_path,
                        "reason": repair_reason,
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "row_id": row.row_id,
                    "material_id": row.material_id,
                    "graph_path": row.graph_path,
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                }
            )

        if i % 500 == 0:
            gc.collect()

    failure_path = processed / f"graph_failures_{args.subset}.csv"
    repair_path = processed / f"graph_repairs_{args.subset}.csv"
    pd.DataFrame(failures).to_csv(failure_path, index=False)
    pd.DataFrame(repairs).to_csv(repair_path, index=False)

    summary = {
        **counts,
        "failed": len(failures),
        "subset": args.subset,
        "verify_existing": bool(args.verify_existing),
        "repair_log": str(repair_path),
    }
    print(summary)

    if failures and not args.allow_failures:
        raise RuntimeError(
            f"{len(failures)} graph(s) failed. Inspect {failure_path}."
        )


if __name__ == "__main__":
    main()
