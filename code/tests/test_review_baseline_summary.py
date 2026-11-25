import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from train.metrics import best_mcc_threshold
from train.summarize_review_baselines import (
    DEFAULT_SEEDS,
    EXPERIMENTS,
    _bootstrap_metric_differences,
    align_frames,
    classification_metrics,
    subset_mask,
    summarize,
)


class ReviewBaselineSummaryTests(unittest.TestCase):
    def test_metrics_and_thermo_subset(self):
        frame = pd.DataFrame(
            {
                "material_id": ["a", "b", "c", "d"],
                "ehull": [0.0, 0.10, 0.11, 0.4],
                "y_thermo": [1, 1, 0, 0],
                "y_dynamic": [0, 1, 0, 1],
                "p_dynamic": [0.1, 0.9, 0.2, 0.8],
            }
        )
        np.testing.assert_array_equal(
            subset_mask(frame, "thermo_positive"), [True, True, False, False]
        )
        metrics = classification_metrics(
            frame["y_dynamic"], frame["p_dynamic"], threshold=0.5
        )
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["mcc"], 1.0)

    def test_alignment_rejects_id_mismatch(self):
        left = pd.DataFrame(
            {
                "material_id": ["a", "b"],
                "ehull": [0.0, 0.2],
                "y_thermo": [1, 0],
                "y_dynamic": [1, 0],
            }
        )
        right = left.copy()
        right.loc[1, "material_id"] = "c"
        with self.assertRaisesRegex(ValueError, "material_id mismatch"):
            align_frames(left, right, "left", "right")

    def test_bootstrap_is_reproducible(self):
        labels = np.array([0] * 10 + [1] * 10)
        only = np.linspace(0.2, 0.8, 20)
        ehull = np.linspace(0.01, 0.99, 20)
        first = _bootstrap_metric_differences(
            labels, only, ehull, 0.5, 0.5, 100, 17, 10
        )
        second = _bootstrap_metric_differences(
            labels, only, ehull, 0.5, 0.5, 100, 17, 10
        )
        self.assertEqual(first, second)
        self.assertEqual(first["pr_auc"]["bootstrap_replicates_valid"], 100)

    def test_end_to_end_summary_pairs_all_experiments(self):
        with tempfile.TemporaryDirectory() as temporary:
            review_dir = Path(temporary)
            labels = np.array([0, 1, 0, 1, 0, 1])
            for variants in EXPERIMENTS.values():
                for variant, experiment in variants.items():
                    for seed in DEFAULT_SEEDS:
                        seed_dir = review_dir / experiment / f"seed_{seed}"
                        seed_dir.mkdir(parents=True)
                        base = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
                        probability = (
                            np.clip(base + 0.02, 0.0, 1.0)
                            if variant == "dynamic_ehull"
                            else base
                        )
                        frame = pd.DataFrame(
                            {
                                "material_id": [f"m{i}" for i in range(6)],
                                "ehull": [0.0, 0.03, 0.08, 0.10, 0.2, 0.3],
                                "y_thermo": [1, 1, 1, 1, 0, 0],
                                "y_dynamic": labels,
                                "p_dynamic": probability,
                            }
                        )
                        frame.to_csv(seed_dir / "test_predictions.csv", index=False)
                        frame.to_csv(seed_dir / "val_predictions.csv", index=False)
                        (seed_dir / "thresholds.json").write_text(
                            json.dumps({"dynamic": 0.5}), encoding="utf-8"
                        )
            baseline, deployment, paired, summary = summarize(
                review_dir,
                bootstrap_replicates=10,
                bootstrap_seed=9,
            )
            self.assertEqual(len(baseline), 16)
            self.assertEqual(len(deployment), 40)
            self.assertEqual(len(paired), 48)
            self.assertEqual(len(summary["pairing_audit"]), 8)
            self.assertTrue(
                np.allclose(baseline["validation_threshold_mean"], 0.5)
            )
            self.assertEqual(len(summary["input_manifest"]), 120)
            self.assertTrue(
                all(
                    row["paired_across_variants_and_seeds"]
                    for row in summary["pairing_audit"]
                )
            )
            self.assertTrue(
                all(
                    row["validation_paired_across_variants_and_seeds"]
                    for row in summary["pairing_audit"]
                )
            )
            expected_only = best_mcc_threshold(labels, base)
            expected_ehull = best_mcc_threshold(labels, base + 0.02)
            self.assertEqual(
                paired["dynamic_only_ensemble_threshold"].unique().tolist(),
                [expected_only],
            )
            self.assertEqual(
                paired["dynamic_ehull_ensemble_threshold"].unique().tolist(),
                [expected_ehull],
            )
            self.assertNotEqual(expected_only, 0.5)
            self.assertNotEqual(expected_ehull, 0.5)


if __name__ == "__main__":
    unittest.main()
