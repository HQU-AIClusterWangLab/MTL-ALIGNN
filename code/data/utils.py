from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def write_json(data: Any, path: str | Path) -> None:
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


def safe_filename(value: Any, max_length: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return (text or "unknown")[:max_length]


def row_to_mapping(row: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        for key in row.keys():
            try:
                result[str(key)] = row[key]
            except Exception:
                pass
    except Exception:
        pass
    try:
        result.update(dict(row.key_value_pairs))
    except Exception:
        pass
    try:
        data = row.data
        if isinstance(data, dict):
            result.update(data)
    except Exception:
        pass
    for key in ("id", "unique_id", "formula", "natoms"):
        try:
            result.setdefault(key, getattr(row, key))
        except Exception:
            pass
    return result


def normalize_dynamic(value: Any, config: dict) -> tuple[int, str]:
    raw = "" if value is None else str(value).strip()
    normalized = raw.lower()
    labels = config["labels"]
    positive = {str(item).strip().lower() for item in labels["dynamic_positive"]}
    negative = {str(item).strip().lower() for item in labels["dynamic_negative"]}
    unknown = {str(item).strip().lower() for item in labels["dynamic_unknown"]}
    if normalized in positive:
        return 1, raw
    if normalized in negative:
        return 0, raw
    if normalized in unknown:
        return -1, raw
    raise ValueError(f"Unrecognized dyn_stab value: {raw!r}")


def structure_hash(atoms, decimals: int = 6) -> str:
    scaled = np.mod(atoms.get_scaled_positions(wrap=True), 1.0)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    order = np.lexsort((scaled[:, 2], scaled[:, 1], scaled[:, 0], numbers))
    payload = {
        "numbers": numbers[order].tolist(),
        "cell": np.round(np.asarray(atoms.cell.array), decimals).tolist(),
        "scaled": np.round(scaled[order], decimals).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=int).tolist(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def anonymous_formula(formula: str) -> str:
    try:
        from pymatgen.core import Composition
        return str(Composition(formula).anonymized_formula)
    except Exception:
        return "unknown"


def chemical_system(atoms) -> str:
    return "-".join(sorted(set(atoms.get_chemical_symbols())))


def composition_feature_dict(atoms, max_z: int = 118) -> dict[str, float]:
    from ase.data import atomic_masses, covalent_radii

    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    fractions = np.bincount(numbers, minlength=max_z + 1).astype(float)
    fractions /= max(float(len(numbers)), 1.0)
    result: dict[str, float] = {
        f"frac_Z_{z:03d}": float(fractions[z]) for z in range(1, max_z + 1)
    }
    masses = atomic_masses[numbers]
    radii = covalent_radii[numbers]
    result.update(
        {
            "n_elements": float(len(np.unique(numbers))),
            "z_mean": float(numbers.mean()),
            "z_std": float(numbers.std()),
            "z_min": float(numbers.min()),
            "z_max": float(numbers.max()),
            "mass_mean": float(masses.mean()),
            "mass_std": float(masses.std()),
            "radius_mean": float(radii.mean()),
            "radius_std": float(radii.std()),
        }
    )
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
