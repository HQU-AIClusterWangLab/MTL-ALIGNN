"""Frozen internal revision matrix: three controls x two splits x 3/5 seeds."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train.common import (capture_environment, experiment_result_root, load_yaml,
                          resolve_experiment, save_json, save_yaml)
from train.revision_evaluation import read_members, summarize_matrix
from train.train_gnn import train_with_batch_fallback

SEEDS = [2025, 2026, 2027, 2028, 2029]
SPLITS = {"composition": "canonical_reduced_composition_id", "structure": "structure_family_id"}
BASELINES = {"composition": [(25, "Single"), (26, "Equal"), (27, "Full"), (28, "Full-noLC")],
             "structure": [(33, "Single"), (34, "Equal"), (35, "Full"), (36, "Full-noLC")]}
METHODS = {"equal_lc": "Equal+LC", "four_state": "Four-state", "dual_product": "T+D-product"}
REQUIRED_OUTPUTS = ("best_checkpoint.pt", "metrics.json", "summary.yaml", "environment.json",
                    "resolved_experiment.yaml", "val_predictions.csv", "test_predictions.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_json(value, temporary)
    temporary.replace(path)


def configurations(project_root: Path, results_override: str | None = None):
    project = load_yaml(project_root / "config.yaml")
    configs = []
    for path in sorted((project_root / "train/experiments/revision").glob("P*.yaml")):
        setting = load_yaml(path)
        setting["experiment_file"] = path.relative_to(project_root).as_posix()
        config = resolve_experiment(project, setting)
        if results_override:
            config["paths"]["results_dir"] = str(Path(results_override).resolve())
        configs.append(config)
    if len(configs) != 6:
        raise ValueError(f"Expected six revision configurations; found {len(configs)}.")
    return project, configs


def check_split(metadata: pd.DataFrame, split: pd.DataFrame, group_column: str):
    if metadata.material_id.isna().any() or metadata.material_id.duplicated().any():
        raise ValueError("Supervised metadata IDs must be unique and nonempty.")
    if split.material_id.isna().any() or split.material_id.duplicated().any():
        raise ValueError("Split IDs must be unique and nonempty.")
    if set(metadata.material_id) != set(split.material_id):
        raise ValueError("Split must cover precisely the supervised data.")
    if set(split.split) != {"train", "val", "test"}:
        raise ValueError("Expected nonempty train, val and test partitions.")
    if split[group_column].isna().any() or split.groupby(group_column).split.nunique().max() > 1:
        raise ValueError(f"Missing or leaking grouping key: {group_column}")
    columns = ["thermo_label", "dynamic_label", "joint_label"]
    values = metadata[columns].to_numpy()
    if not np.isin(values, [0, 1]).all() or not np.array_equal(values[:, 2], values[:, 0] * values[:, 1]):
        raise ValueError("Supervised labels must be binary and J=T AND D.")
    pd.testing.assert_frame_equal(
        metadata.set_index("material_id")[columns].sort_index(),
        split.set_index("material_id")[columns].sort_index(), check_dtype=False,
    )
    for part, rows in split.groupby("split"):
        if any(rows[c].nunique() != 2 for c in columns):
            raise ValueError(f"{part} must contain both classes for each task.")
    return {part: {"n": len(rows), "groups": rows[group_column].nunique(),
                   "state_counts": (2 * rows.thermo_label + rows.dynamic_label).value_counts().sort_index().to_dict()}
            for part, rows in split.groupby("split")}


def validate_existing_baseline(root, seeds, metadata, split, config):
    for part in ("val", "test"):
        expected = metadata[metadata.material_id.isin(split.loc[split.split == part, "material_id"])]
        expected = expected.sort_values("material_id").reset_index(drop=True)
        for frame in read_members(root, seeds, part):
            if frame.material_id.tolist() != expected.material_id.tolist():
                raise ValueError(f"Historical {part} predictions do not match frozen split: {root}")
            for task in ("thermo", "dynamic", "joint"):
                if not np.array_equal(frame[f"y_{task}"], expected[f"{task}_label"]):
                    raise ValueError(f"Historical labels mismatch: {root}, {task}")
    for seed in seeds:
        folder = root / f"seed_{seed}"
        old = load_yaml(folder / "resolved_experiment.yaml")
        for section, fields in {
            "training": ("max_epochs", "patience", "learning_rate", "weight_decay", "batch_size",
                         "scheduler_patience", "scheduler_factor", "class_weighting", "amp",
                         "gradient_clip_norm", "num_workers"),
            "model": ("alignn_layers", "gcn_layers", "embedding_features", "hidden_features", "dropout"),
        }.items():
            for field in fields:
                if old[section].get(field) != config[section].get(field):
                    raise ValueError(f"Training budget/architecture mismatch {root}: {section}.{field}")
        if old.get("model_type", "alignn") != "alignn" or old.get("model", {}).get("conditioning_features"):
            raise ValueError(f"Unexpected baseline architecture: {root}")


def contract_for(config, seed, data_signature, code_signature, environment):
    # Expanding the requested seed subset must not invalidate the first three runs.
    resolved = json.loads(json.dumps(config))
    resolved.pop("seeds", None)
    return {"schema": 1, "seed": seed, "experiment": resolved,
            "data": data_signature, "code": code_signature,
            "runtime": {k: environment[k] for k in ("torch", "torch_cuda", "dgl", "alignn", "python_executable")}}


def check_contract(folder: Path, contract: dict, create=False):
    path = folder / "run_contract.json"
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != contract:
            raise RuntimeError(f"Input/config/code/runtime changed; refusing to mix runs: {folder}. "
                               "Use a new --results-dir for a revised experiment.")
    elif folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f"Untracked output already exists: {folder}")
    elif create:
        atomic_json(contract, path)
    if (folder / "COMPLETED.json").exists():
        missing = [name for name in REQUIRED_OUTPUTS if not (folder / name).is_file()]
        if missing:
            raise RuntimeError(f"Incomplete completed member {folder}: {missing}")


def preflight(project_root, project, configs, seeds, allow_cpu=False):
    environment = capture_environment()
    if not environment["cuda_available"] and not allow_cpu:
        raise RuntimeError("CUDA is unavailable. Check the two conda environment; use --allow-cpu only intentionally.")
    metadata_path = Path(project["paths"]["processed_dir"]) / "supervised.csv"
    metadata = pd.read_csv(metadata_path)
    data_hash = sha256(metadata_path)
    graph_stats = []
    for path_text in sorted(metadata.graph_path.unique()):
        path = Path(path_text)
        stat = path.stat()
        if stat.st_size == 0:
            raise ValueError(f"Empty graph cache: {path}")
        graph_stats.append((str(path), stat.st_size, stat.st_mtime_ns))
    graph_signature = hashlib.sha256(json.dumps(graph_stats).encode()).hexdigest()
    code_paths = [*sorted((project_root / "models").glob("*.py")),
                  *[project_root / "train" / n for n in (
                      "common.py", "dataset.py", "train_gnn.py", "metrics.py",
                      "aggregate_ensemble.py", "revision_evaluation.py", "run_paper_revision.py")]]
    code_signature = {p.relative_to(project_root).as_posix(): sha256(p) for p in code_paths}
    split_reports, entries, jobs = {}, [], []
    for config in configs:
        split_name = config["revision"]["split"]
        split_path = metadata_path.parent / config["split_file"]
        split_hash = sha256(split_path)
        frozen = config["revision"]["frozen"]
        if frozen != {"supervised_sha256": data_hash, "split_sha256": split_hash}:
            raise ValueError(f"Frozen metadata/split changed: {config['name']}; do not regenerate splits.")
        split = pd.read_csv(split_path)
        if split_name not in split_reports:
            split_reports[split_name] = check_split(metadata, split, SPLITS[split_name])
            for number, method in BASELINES[split_name]:
                matches = list((Path(project["paths"]["results_dir"]) / "review").glob(f"R{number:02d}_*"))
                if len(matches) != 1:
                    raise ValueError(f"Expected one historical baseline R{number}; got {matches}")
                root = matches[0]
                validate_existing_baseline(root, seeds, metadata, split, config)
                entries.append({"root": str(root), "name": root.name, "split": split_name,
                                "method": method, "source": "existing", "mode": "independent"})
        root = experiment_result_root(config)
        entries.append({"root": str(root), "name": config["name"], "split": split_name,
                        "method": METHODS[config["revision"]["method"]], "source": "revision",
                        "mode": config["model"].get("output_mode", "independent")})
        for seed in seeds:
            contract = contract_for(config, seed,
                                    {"supervised_sha256": data_hash, "split_sha256": split_hash,
                                     "graph_path_size_mtime_sha256": graph_signature},
                                    code_signature, environment)
            folder = root / f"seed_{seed}"
            check_contract(folder, contract)
            jobs.append({"config": config, "seed": seed, "folder": folder, "contract": contract})
    report = {"environment": environment, "splits": split_reports, "seeds": seeds,
              "training_jobs": len(jobs), "historical_baselines": 8,
              "graph_files_checked": len(graph_stats), "graph_check": "existence, size and modification time; not binary SHA256",
              "scope": "internal retrospective supplement, not external DFT validation",
              "jobs": [{"experiment": j["config"]["name"], "seed": j["seed"],
                        "state": "completed" if (j["folder"] / "COMPLETED.json").exists()
                        else "resume" if (j["folder"] / "last_checkpoint.pt").exists() else "pending"} for j in jobs]}
    return jobs, entries, report


@contextlib.contextmanager
def queue_lock(path: Path):
    """OS-held lock survives neither normal exit nor a killed process."""
    import msvcrt

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another paper revision task is already using this output directory.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "pilot", "full5", "summary3", "summary5"], default="pilot")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-dir", help="Separate output root (also useful for a revised configuration).")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    project, configs = configurations(project_root, args.results_dir)
    seeds = SEEDS if args.mode in {"full5", "summary5"} else SEEDS[:3]
    result_root = Path(configs[0]["paths"]["results_dir"]) / "paper_revision"
    run_dir = result_root / "_runner"
    run_dir.mkdir(parents=True, exist_ok=True)
    with queue_lock(run_dir / "queue.lock"):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        with (run_dir / f"{args.mode}_{stamp}.log").open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                status = {"mode": args.mode, "started": stamp, "pid": os.getpid(), "status": "preflight"}
                atomic_json(status, run_dir / "status.json")
                try:
                    jobs, entries, report = preflight(project_root, project, configs, seeds,
                                                     args.allow_cpu or args.mode.startswith("summary"))
                    atomic_json(report, run_dir / "preflight.json")
                    print(json.dumps(report["jobs"], indent=2), flush=True)
                    print(f"Preflight passed: {len(jobs)} jobs, {len(entries)} compared methods/splits.", flush=True)
                    if args.mode != "preflight":
                        if not args.mode.startswith("summary"):
                            for index, job in enumerate(jobs, 1):
                                config, seed, folder = job["config"], job["seed"], job["folder"]
                                status.update(status="training", job=index, total=len(jobs), experiment=config["name"], seed=seed)
                                atomic_json(status, run_dir / "status.json")
                                check_contract(folder, job["contract"], create=True)
                                save_yaml(config, folder.parent / "resolved_experiment.yaml")
                                print(f"\n[{index}/{len(jobs)}] {config['name']} seed={seed}", flush=True)
                                train_with_batch_fallback(config, seed, folder, force=False)
                                info = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                                if info["batch_size"] != config["training"]["batch_size"]:
                                    print("NOTE: OOM fallback changed batch size; see member_comparison.csv.", flush=True)
                        status.update(status="summarizing")
                        atomic_json(status, run_dir / "status.json")
                        summary_dir = result_root / f"summary_{len(seeds)}seeds"
                        summarize_matrix(entries, seeds, summary_dir, configs[0]["uncertainty"]["calibration_bins"])
                        print(f"Summary saved: {summary_dir}", flush=True)
                    status.update(status="completed", finished=time.strftime("%Y%m%d_%H%M%S"))
                    atomic_json(status, run_dir / "status.json")
                except BaseException as exc:
                    status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
                    atomic_json(status, run_dir / "status.json")
                    raise


if __name__ == "__main__":
    main()
