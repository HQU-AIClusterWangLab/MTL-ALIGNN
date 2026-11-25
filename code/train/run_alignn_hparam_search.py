"""Validation-only basic hyperparameter exploration for the ALIGNN baseline.

The script reuses the existing ``train/train_gnn.py`` training implementation
and the frozen IID split.  It never overwrites the main experiment.  Model
selection reads validation Joint AP only; test metrics produced by the common
trainer are deliberately not read by this script.

Workflow
--------
1. Screen one factor at a time using the configured screening seed(s).
2. Select the best validation value for each factor and combine those values.
3. Confirm the baseline and combined candidate over the configured seeds.
4. Recommend the setting with the larger mean validation Joint AP.

Graph parameters are intentionally frozen because the current graph files are
cached at fixed paths.  A cutoff search requires separate, non-overwriting graph
caches and is outside this basic model/training search.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = PROJECT_ROOT / "configs" / "alignn_hparam_search.yaml"


@dataclass(frozen=True)
class Trial:
    trial_id: str
    factor: str
    value: Any
    changes: dict[str, Any]
    config_path: Path
    result_dir: Path


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_nested(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing configuration field: {dotted}")
        value = value[key]
    return value


def set_nested(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    target = data
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise TypeError(f"Cannot set nested field below non-mapping: {dotted}")
        target = child
    target[keys[-1]] = value


def stable_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def slug(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.8g}"
    else:
        text = str(value)
    return (
        text.replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def scientific_payload(resolved: dict[str, Any]) -> dict[str, Any]:
    """Fields that determine a trial, excluding output identity and seeds."""
    keep = (
        "model_type",
        "heads",
        "active_tasks",
        "weighting",
        "logic_beta",
        "logic_lower_bound",
        "split_file",
        "graph",
        "model",
        "training",
    )
    return {key: copy.deepcopy(resolved.get(key)) for key in keep}


def validate_spec(
    project: dict[str, Any], base: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    resolved = deep_merge(project, base)
    if resolved.get("model_type") != "alignn":
        raise ValueError("Base experiment must use model_type=alignn")
    if resolved.get("heads") != ["joint"] or resolved.get("active_tasks") != ["joint"]:
        raise ValueError("Base experiment must be the single-task Joint ALIGNN baseline")
    if resolved.get("split_file") != "iid_split.csv":
        raise ValueError("Hyperparameter search must use the frozen iid_split.csv")
    if spec.get("selection_metric") != "val_joint.pr_auc":
        raise ValueError("Only validation Joint AP is permitted for model selection")
    for key, expected in spec["frozen_graph_parameters"].items():
        actual = resolved.get("graph", {}).get(key)
        if actual != expected:
            raise ValueError(
                f"Frozen graph parameter mismatch for {key}: {actual!r} != {expected!r}"
            )
    for dotted, candidates in spec["parameters"].items():
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError(f"Each search factor needs at least two values: {dotted}")
        baseline = get_nested(resolved, dotted)
        if baseline not in candidates:
            raise ValueError(
                f"Candidate list for {dotted} must include baseline value {baseline!r}"
            )
        if dotted.startswith("graph."):
            raise ValueError("Graph parameters cannot be searched with the shared graph cache")
    return resolved


def materialize_config(
    base: dict[str, Any],
    name: str,
    category: str,
    seeds: list[int],
    changes: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["name"] = name
    config["category"] = category
    config["seeds"] = [int(seed) for seed in seeds]
    config["experiment_file"] = "generated by train/run_alignn_hparam_search.py"
    for dotted, value in changes.items():
        set_nested(config, dotted, value)
    return config


def result_dir_for(project: dict[str, Any], config: dict[str, Any]) -> Path:
    resolved = deep_merge(project, config)
    return Path(resolved["paths"]["results_dir"]) / resolved["category"] / resolved["name"]


def run_training(config_path: Path, force: bool) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train" / "train_gnn.py"),
        "--experiment",
        str(config_path),
    ]
    if force:
        command.append("--force")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONUTF8"] = "1"
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def read_validation_score(result_dir: Path, seed: int) -> dict[str, Any]:
    seed_dir = result_dir / f"seed_{seed}"
    marker = seed_dir / "COMPLETED.json"
    metrics_path = seed_dir / "metrics.json"
    if not marker.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Incomplete trial output: {seed_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    # Deliberately read validation only. Test metrics must not influence HPO.
    validation = metrics.get("val_joint")
    if not isinstance(validation, dict) or "pr_auc" not in validation:
        raise KeyError(f"Missing val_joint.pr_auc in {metrics_path}")
    summary_path = seed_dir / "summary.yaml"
    summary = load_yaml(summary_path) if summary_path.is_file() else {}
    return {
        "seed": int(seed),
        "val_joint_ap": float(validation["pr_auc"]),
        "best_epoch": summary.get("best_epoch"),
        "batch_size": summary.get("batch_size"),
        "result_dir": str(seed_dir),
    }


def build_screen_trials(
    project: dict[str, Any],
    base: dict[str, Any],
    resolved_base: dict[str, Any],
    spec: dict[str, Any],
    config_dir: Path,
) -> tuple[Trial, dict[str, dict[Any, Trial]]]:
    screen_seeds = [int(seed) for seed in spec["screen_seeds"]]
    base_signature = stable_hash(scientific_payload(resolved_base))
    category = "hparam_search/alignn_baseline/screen"

    baseline_name = f"H00_baseline_{base_signature}"
    baseline_config = materialize_config(base, baseline_name, category, screen_seeds, {})
    baseline_path = config_dir / f"{baseline_name}.yaml"
    save_yaml(baseline_config, baseline_path)
    baseline_trial = Trial(
        trial_id=baseline_name,
        factor="baseline",
        value="baseline",
        changes={},
        config_path=baseline_path,
        result_dir=result_dir_for(project, baseline_config),
    )

    factors: dict[str, dict[Any, Trial]] = {}
    for index, (factor, candidates) in enumerate(spec["parameters"].items(), start=1):
        baseline_value = get_nested(resolved_base, factor)
        factor_trials: dict[Any, Trial] = {baseline_value: baseline_trial}
        short = factor.split(".")[-1]
        for candidate in candidates:
            if candidate == baseline_value:
                continue
            changes = {factor: candidate}
            candidate_resolved = copy.deepcopy(resolved_base)
            set_nested(candidate_resolved, factor, candidate)
            signature = stable_hash(scientific_payload(candidate_resolved))
            name = f"H{index:02d}_{short}_{slug(candidate)}_{signature}"
            config = materialize_config(base, name, category, screen_seeds, changes)
            config_path = config_dir / f"{name}.yaml"
            save_yaml(config, config_path)
            factor_trials[candidate] = Trial(
                trial_id=name,
                factor=factor,
                value=candidate,
                changes=changes,
                config_path=config_path,
                result_dir=result_dir_for(project, config),
            )
        factors[factor] = factor_trials
    return baseline_trial, factors


def unique_trials(baseline: Trial, factors: dict[str, dict[Any, Trial]]) -> list[Trial]:
    found = {baseline.trial_id: baseline}
    for candidate_map in factors.values():
        for trial in candidate_map.values():
            found[trial.trial_id] = trial
    return list(found.values())


def collect_screening(
    factors: dict[str, dict[Any, Trial]], seeds: list[int]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_values: dict[str, Any] = {}
    for factor, candidate_map in factors.items():
        candidate_summaries = []
        for candidate, trial in candidate_map.items():
            scores = [read_validation_score(trial.result_dir, seed) for seed in seeds]
            for score in scores:
                rows.append(
                    {
                        "factor": factor,
                        "candidate": candidate,
                        "trial_id": trial.trial_id,
                        **score,
                    }
                )
            candidate_summaries.append(
                (float(pd.Series([row["val_joint_ap"] for row in scores]).mean()), candidate)
            )
        # Deterministic tie break: keep the first candidate from the YAML list.
        best_mean = max(item[0] for item in candidate_summaries)
        best_values[factor] = next(
            candidate for mean, candidate in candidate_summaries if mean == best_mean
        )
    return pd.DataFrame(rows), best_values


def screening_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["factor", "candidate"], sort=False)["val_joint_ap"]
        .agg([("mean_val_joint_ap", "mean"), ("sd_val_joint_ap", "std"), ("n", "count")])
        .reset_index()
    )


def build_confirmation_configs(
    project: dict[str, Any],
    base: dict[str, Any],
    resolved_base: dict[str, Any],
    best_values: dict[str, Any],
    seeds: list[int],
    config_dir: Path,
) -> dict[str, Trial]:
    category = "hparam_search/alignn_baseline/confirm"
    configurations = {
        "baseline": {},
        "selected": dict(best_values),
    }
    trials: dict[str, Trial] = {}
    for label, changes in configurations.items():
        resolved = copy.deepcopy(resolved_base)
        for dotted, value in changes.items():
            set_nested(resolved, dotted, value)
        signature = stable_hash(scientific_payload(resolved))
        name = f"H90_confirm_{label}_{signature}"
        config = materialize_config(base, name, category, seeds, changes)
        config_path = config_dir / f"{name}.yaml"
        save_yaml(config, config_path)
        trials[label] = Trial(
            trial_id=name,
            factor="confirmation",
            value=label,
            changes=changes,
            config_path=config_path,
            result_dir=result_dir_for(project, config),
        )
    return trials


def collect_confirmation(trials: dict[str, Trial], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for label, trial in trials.items():
        for seed in seeds:
            rows.append({"configuration": label, **read_validation_score(trial.result_dir, seed)})
    return pd.DataFrame(rows)


def confirmation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("configuration", sort=False)["val_joint_ap"]
        .agg([("mean_val_joint_ap", "mean"), ("sd_val_joint_ap", "std"), ("n", "count")])
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--mode", choices=("plan", "screen", "confirm", "all"), default="all"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed HPO trials. Never affects results/main.",
    )
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    spec = load_yaml(spec_path)
    project = load_yaml(PROJECT_ROOT / "config.yaml")
    base_path = PROJECT_ROOT / spec["base_experiment"]
    base = load_yaml(base_path)
    resolved_base = validate_spec(project, base, spec)

    output_dir = PROJECT_ROOT / spec["output_dir"]
    config_dir = output_dir / "generated_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    baseline, factors = build_screen_trials(
        project, base, resolved_base, spec, config_dir
    )
    trials = unique_trials(baseline, factors)
    plan_rows = [
        {
            "trial_id": trial.trial_id,
            "factor": trial.factor,
            "candidate": trial.value,
            "seeds": ",".join(map(str, spec["screen_seeds"])),
            "config_path": str(trial.config_path),
            "result_dir": str(trial.result_dir),
        }
        for trial in trials
    ]
    pd.DataFrame(plan_rows).to_csv(output_dir / "search_plan.csv", index=False)

    manifest = {
        "schema": 1,
        "spec": str(spec_path),
        "base_experiment": str(base_path),
        "selection_metric": "validation Joint AP only",
        "test_used_for_selection": False,
        "search_design": "one factor at a time, then combined 3-seed confirmation",
        "screen_seeds": spec["screen_seeds"],
        "confirm_seeds": spec["confirm_seeds"],
        "frozen_graph_parameters": spec["frozen_graph_parameters"],
        "screen_trial_count": len(trials),
    }
    (output_dir / "search_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Search plan: {output_dir / 'search_plan.csv'}")
    print(f"Unique screening trials: {len(trials)}")
    if args.mode == "plan":
        print("PLAN ONLY: no training was started.")
        return

    screen_seeds = [int(seed) for seed in spec["screen_seeds"]]
    if args.mode in {"screen", "all"}:
        for trial in trials:
            run_training(trial.config_path, force=args.force)

    screen_frame, best_values = collect_screening(factors, screen_seeds)
    screen_frame.to_csv(output_dir / "screening_seed_results.csv", index=False)
    screening_summary(screen_frame).to_csv(
        output_dir / "screening_summary.csv", index=False
    )
    recommended = {
        "selection_metric": "mean validation Joint AP",
        "selected_values": best_values,
        "test_used_for_selection": False,
    }
    save_yaml(recommended, output_dir / "recommended_parameters.yaml")
    print("Selected factor values:")
    for factor, value in best_values.items():
        print(f"  {factor}: {value}")

    if args.mode == "screen":
        return

    confirm_seeds = [int(seed) for seed in spec["confirm_seeds"]]
    confirmation = build_confirmation_configs(
        project, base, resolved_base, best_values, confirm_seeds, config_dir
    )
    if args.mode in {"confirm", "all"}:
        for trial in confirmation.values():
            run_training(trial.config_path, force=args.force)

    confirm_frame = collect_confirmation(confirmation, confirm_seeds)
    confirm_frame.to_csv(output_dir / "confirmation_seed_results.csv", index=False)
    confirm_summary = confirmation_summary(confirm_frame)
    confirm_summary.to_csv(output_dir / "confirmation_summary.csv", index=False)
    winner = confirm_summary.sort_values(
        ["mean_val_joint_ap", "configuration"], ascending=[False, True]
    ).iloc[0]
    final = {
        "recommended_configuration": str(winner["configuration"]),
        "mean_validation_joint_ap": float(winner["mean_val_joint_ap"]),
        "sd_validation_joint_ap": float(winner["sd_val_joint_ap"]),
        "n_seeds": int(winner["n"]),
        "selected_values": best_values if winner["configuration"] == "selected" else {},
        "test_used_for_selection": False,
    }
    save_yaml(final, output_dir / "final_recommendation.yaml")
    print(confirm_summary.to_string(index=False))
    print(f"Final recommendation: {output_dir / 'final_recommendation.yaml'}")


if __name__ == "__main__":
    main()
