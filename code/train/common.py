from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from models.losses import MultiTaskObjective
from models.joint_outputs import FourStateObjective
from models.multitask import CrystalMultiTaskModel


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(data: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def save_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_experiment(project_cfg: dict, experiment_cfg: dict) -> dict:
    resolved = deep_merge(project_cfg, experiment_cfg)
    if "name" not in resolved or "category" not in resolved:
        raise ValueError("Experiment YAML must define name and category.")
    resolved["experiment_file"] = experiment_cfg.get(
        "experiment_file", ""
    )
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_runtime(config: dict) -> None:
    precision = config.get("runtime", {}).get(
        "torch_float32_matmul_precision", "high"
    )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def build_model(experiment: dict) -> CrystalMultiTaskModel:
    return CrystalMultiTaskModel(
        model_cfg=experiment["model"],
        heads=list(experiment["heads"]),
        model_type=experiment.get("model_type", "alignn"),
    )


def build_objective(
    experiment: dict,
    positive_weights: dict[str, float],
) -> torch.nn.Module:
    mode = experiment.get("model", {}).get("output_mode", "independent")
    if mode == "four_state":
        if (experiment["training"].get("class_weighting", "none") != "none"
                or experiment.get("logic_beta", 0) != 0
                or experiment.get("shuffle_auxiliary", False)):
            raise ValueError("four_state uses unweighted CE only, with consistent labels.")
        return FourStateObjective()
    if mode == "dual_product" and (
        experiment["active_tasks"] != ["thermo", "dynamic"]
        or experiment.get("logic_beta", 0) != 0
    ):
        raise ValueError("dual_product trains only T,D, with no joint/logic loss.")
    return MultiTaskObjective(
        active_tasks=list(experiment["active_tasks"]),
        weighting=experiment.get("weighting", "equal"),
        positive_weights=positive_weights,
        fixed_weights=experiment.get("fixed_weights", {}),
        logic_beta=float(experiment.get("logic_beta", 0.0)),
        logic_lower_bound=bool(experiment.get("logic_lower_bound", True)),
    )


def enable_mc_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def capture_environment() -> dict:
    import alignn
    import ase
    import dgl
    import pandas
    import sklearn

    try:
        import xgboost
    except ImportError:
        xgboost = None

    result = {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dgl": dgl.__version__,
        "alignn": getattr(alignn, "__version__", "unknown"),
        "ase": ase.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": getattr(xgboost, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cwd": os.getcwd(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = props.name
        result["gpu_memory_gb"] = props.total_memory / 1024**3
    try:
        result["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except Exception:
        result["git_commit"] = None
    return result


def experiment_result_root(experiment: dict) -> Path:
    return (
        Path(experiment["paths"]["results_dir"])
        / experiment["category"]
        / experiment["name"]
    )


def completed(output_dir: Path) -> bool:
    return (output_dir / "COMPLETED.json").exists()
