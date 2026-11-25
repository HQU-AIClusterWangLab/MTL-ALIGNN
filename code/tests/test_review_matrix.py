from __future__ import annotations

from pathlib import Path
import unittest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / "train" / "experiments" / "review"
SEEDS = [2025, 2026, 2027, 2028, 2029]
SPLITS = {
    "random": "review_splits/random_iid_stratified.csv",
    "composition": "review_splits/composition_disjoint.csv",
    "chemical": "review_splits/chemical_system_disjoint.csv",
    "structure": "review_splits/structure_family_disjoint.csv",
}
VARIANTS = {
    "single_joint": {
        "heads": ["joint"],
        "active_tasks": ["joint"],
        "weighting": "equal",
        "logic_beta": 0.0,
        "logic_lower_bound": False,
    },
    "equal_mtl": {
        "heads": ["thermo", "dynamic", "joint"],
        "active_tasks": ["thermo", "dynamic", "joint"],
        "weighting": "equal",
        "logic_beta": 0.0,
        "logic_lower_bound": False,
    },
    "full_mtl": {
        "heads": ["thermo", "dynamic", "joint"],
        "active_tasks": ["thermo", "dynamic", "joint"],
        "weighting": "adaptive_auxiliary",
        "logic_beta": 0.10,
        "logic_lower_bound": True,
    },
    "full_no_logic": {
        "heads": ["thermo", "dynamic", "joint"],
        "active_tasks": ["thermo", "dynamic", "joint"],
        "weighting": "adaptive_auxiliary",
        "logic_beta": 0.0,
        "logic_lower_bound": False,
    },
}


class ReviewMatrixTests(unittest.TestCase):
    def test_complete_joint_matrix_matches_templates(self) -> None:
        paths = sorted(REVIEW_DIR.glob("R2[1-9]_*.yaml")) + sorted(
            REVIEW_DIR.glob("R3[0-6]_*.yaml")
        )
        self.assertEqual(len(paths), 16)
        names = set()
        combinations = set()
        for path in paths:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(config["name"], path.stem)
            self.assertNotIn(config["name"], names)
            names.add(config["name"])
            self.assertEqual(config["category"], "review")
            self.assertEqual(config["model_type"], "alignn")
            self.assertNotIn("monitor_task", config)
            self.assertEqual(config["seeds"], SEEDS)

            split = next(key for key in SPLITS if path.stem.endswith(f"_{key}"))
            variant = next(
                key for key in VARIANTS if f"_{key}_alignn_" in path.stem
            )
            self.assertEqual(config["split_file"], SPLITS[split])
            self.assertTrue(
                (PROJECT_ROOT / "data" / "processed" / SPLITS[split]).is_file()
            )
            for key, value in VARIANTS[variant].items():
                self.assertEqual(config[key], value)
            combinations.add((split, variant))

        self.assertEqual(
            combinations,
            {(split, variant) for split in SPLITS for variant in VARIANTS},
        )

    def test_runner_lists_every_joint_matrix_config_and_scopes(self) -> None:
        runner = (PROJECT_ROOT / "scripts" / "run_review_baselines.ps1").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn('ValidateSet("All", "Deployment", "FullMatrix")', runner)
        self.assertIn('[string]$GnnSet = "All"', runner)
        for path in sorted(REVIEW_DIR.glob("R2[1-9]_*.yaml")) + sorted(
            REVIEW_DIR.glob("R3[0-6]_*.yaml")
        ):
            self.assertEqual(runner.count(f'"{path.name}"'), 1)


if __name__ == "__main__":
    unittest.main()
