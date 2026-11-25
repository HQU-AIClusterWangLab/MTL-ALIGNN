import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from train.summarize_review_gnn import (
    DEFAULT_SEEDS,
    REQUIRED_ARTIFACTS,
    execute,
    experiment_specs,
    inspect_progress,
)


class ReviewGnnSummaryTests(unittest.TestCase):
    @staticmethod
    def _frame(probability: np.ndarray, task: str) -> pd.DataFrame:
        y_thermo = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_dynamic = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_joint = y_thermo * y_dynamic
        return pd.DataFrame(
            {
                "material_id": [f"m{index}" for index in range(8)],
                "ehull": [0.01, 0.03, 0.08, 0.10, 0.11, 0.20, 0.30, 0.40],
                "y_thermo": y_thermo,
                "y_dynamic": y_dynamic,
                "y_joint": y_joint,
                f"p_{task}": probability,
            }
        )

    @classmethod
    def _write_complete_scope(cls, review_dir: Path, scope: str) -> None:
        for spec in experiment_specs(scope):
            labels = (
                np.array([0, 1, 0, 1, 0, 1, 0, 1])
                if spec.task == "dynamic"
                else np.array([0, 1, 0, 1, 0, 0, 0, 0])
            )
            base = np.where(labels == 1, 0.78, 0.22).astype(float)
            adjustment = {
                "dynamic_only": -0.02,
                "dynamic_ehull": 0.02,
                "single": -0.03,
                "equal": -0.01,
                "full": 0.03,
                "no_logic": 0.00,
            }[spec.variant]
            probability = np.clip(base + np.where(labels == 1, adjustment, -adjustment), 0, 1)
            frame = cls._frame(probability, spec.task)
            for seed in DEFAULT_SEEDS:
                seed_dir = review_dir / spec.experiment / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                frame.to_csv(seed_dir / "val_predictions.csv", index=False)
                frame.to_csv(seed_dir / "test_predictions.csv", index=False)
                (seed_dir / "thresholds.json").write_text(
                    json.dumps({spec.task: 0.5}), encoding="utf-8"
                )
                (seed_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
                (seed_dir / "resolved_experiment.yaml").write_text(
                    f"name: {spec.experiment}\n", encoding="utf-8"
                )
                (seed_dir / "COMPLETED.json").write_text(
                    json.dumps({"status": "completed", "seed": seed}),
                    encoding="utf-8",
                )

    def test_missing_runs_are_explicitly_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress, status = inspect_progress(root / "review", "deployment")
            self.assertEqual(status["state"], "pending")
            self.assertEqual(status["expected_seed_runs"], 40)
            self.assertEqual(status["pending_seed_runs"], 40)
            self.assertTrue((progress["state"] == "pending").all())

            output_dir = root / "summary"
            code = execute(
                root / "review",
                output_dir,
                "deployment",
                bootstrap_replicates=3,
                allow_pending=True,
            )
            self.assertEqual(code, 0)
            written = json.loads((output_dir / "status.json").read_text())
            self.assertFalse(written["summary_artifacts_valid"])
            self.assertFalse((output_dir / "paired_comparisons.csv").exists())

    def test_completed_marker_with_missing_artifacts_is_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            review_dir = Path(temporary) / "review"
            spec = experiment_specs("deployment")[0]
            seed_dir = review_dir / spec.experiment / f"seed_{DEFAULT_SEEDS[0]}"
            seed_dir.mkdir(parents=True)
            (seed_dir / "COMPLETED.json").write_text(
                json.dumps({"status": "completed", "seed": DEFAULT_SEEDS[0]}),
                encoding="utf-8",
            )
            progress, status = inspect_progress(review_dir, "deployment")
            self.assertEqual(status["state"], "invalid")
            invalid = progress.loc[progress["state"] == "invalid"].iloc[0]
            self.assertEqual(
                set(invalid["missing_artifacts"].split(";")), set(REQUIRED_ARTIFACTS)
            )

    def test_all_scope_five_seed_summary_and_pair_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_dir = root / "review"
            output_dir = root / "summary"
            self._write_complete_scope(review_dir, "all")
            code = execute(
                review_dir,
                output_dir,
                "all",
                bootstrap_replicates=5,
                bootstrap_seed=7,
            )
            self.assertEqual(code, 0)
            status = json.loads((output_dir / "status.json").read_text())
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(status["state"], "completed")
            self.assertTrue(status["summary_artifacts_valid"])
            self.assertEqual(status["complete_seed_runs"], 120)
            self.assertEqual(summary["counts"]["experiments"], 24)
            self.assertEqual(summary["counts"]["seed_runs"], 120)
            self.assertEqual(summary["counts"]["per_seed_metric_rows"], 240)
            self.assertEqual(summary["counts"]["experiment_summary_rows"], 48)
            self.assertEqual(summary["counts"]["paired_metric_rows"], 192)
            self.assertEqual(summary["counts"]["pairing_audit_rows"], 32)
            paired = pd.read_csv(output_dir / "paired_comparisons.csv")
            self.assertTrue((paired["n_seeds"] == 5).all())
            self.assertTrue((paired["bootstrap_replicates_requested"] == 5).all())
            self.assertTrue(
                paired["pair"].isin(
                    [
                        "dynamic_ehull_minus_dynamic_only",
                        "full_minus_single",
                        "full_minus_equal",
                        "full_minus_no_logic",
                    ]
                ).all()
            )


if __name__ == "__main__":
    unittest.main()
