import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from train.run_review_supplements import (
    COMMON_SEEDS,
    build_frozen_ood_evaluations,
    calibration_metrics,
    fit_temperature,
    fixed_bin_edges,
    logic_violation_magnitudes,
    load_unknown_ensemble_calibration,
    paired_bootstrap_difference,
    probability_uncertainty_table,
    quadrant_counts,
    reliability_rows,
    summarize_logic_violations,
)


class ReviewSupplementTests(unittest.TestCase):
    def test_each_ood_scenario_fits_its_own_development_validation(self):
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        base_logits = np.array([-2.0, 1.0, -1.0, 2.0, -0.5, 0.5, 1.0, -1.0])
        roots = {
            "IID": ("results/main/M06_full_mtl", 1.0),
            "chemical_OOD": (
                "results/optional/O01_chemical_system_ood",
                2.0,
            ),
            "structure_OOD": (
                "results/optional/O02_structure_family_ood",
                3.0,
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            for _, (relative_root, scale) in roots.items():
                for seed_index, seed in enumerate(COMMON_SEEDS):
                    seed_root = project_root / relative_root / f"seed_{seed}"
                    seed_root.mkdir(parents=True)
                    for split in ("val", "test"):
                        logits = scale * base_logits + (seed_index - 1) * 0.05
                        probability = 1.0 / (1.0 + np.exp(-logits))
                        frame = pd.DataFrame(
                            {
                                "material_id": [
                                    f"{split}_{index}" for index in range(len(labels))
                                ],
                                "y_joint": labels,
                                "p_joint": probability,
                                "logit_joint": logits,
                            }
                        )
                        frame.to_csv(
                            seed_root / f"{split}_predictions.csv", index=False
                        )

            calibration, reliability, _, probability_uncertainty, protocol, _, _ = (
                build_frozen_ood_evaluations(project_root, n_bins=3)
            )
            parameters = protocol["scenario_parameters"]
            temperatures = [
                parameters[evaluation]["temperature_joint"]
                for evaluation in roots
            ]
            self.assertEqual(len({round(value, 3) for value in temperatures}), 3)
            for evaluation, (relative_root, _) in roots.items():
                fit_sources = parameters[evaluation]["fit_source"]
                self.assertTrue(
                    all(source.startswith(relative_root) for source in fit_sources)
                )
                row = calibration[
                    (calibration["evaluation"] == evaluation)
                    & calibration["probability_scale"].str.startswith("calibrated")
                ].iloc[0]
                self.assertAlmostEqual(
                    row["temperature"],
                    parameters[evaluation]["temperature_joint"],
                )
                self.assertEqual(
                    row["threshold"],
                    parameters[evaluation][
                        "calibrated_decision_threshold_joint"
                    ],
                )
                reliability_counts = reliability[
                    reliability["evaluation"] == evaluation
                ].groupby(["probability_scale", "binning_scheme"])["count"].sum()
                self.assertTrue(
                    (reliability_counts == parameters[evaluation]["held_out_test_count"]).all()
                )
                self.assertEqual(
                    int(
                        probability_uncertainty[
                            probability_uncertainty["evaluation"] == evaluation
                        ]["count"].sum()
                    ),
                    parameters[evaluation]["held_out_test_count"],
                )
            self.assertFalse(protocol["test_or_OOD_refit_performed"])

    def test_unknown_threshold_requires_matching_ensemble_width(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            ensemble = project_root / "results/main/M06_full_mtl/ensemble"
            screening = project_root / "results/main/M10_unknown_screening"
            ensemble.mkdir(parents=True)
            screening.mkdir(parents=True)
            threshold = 0.4703030303030303
            temperature = 4.497624956625517
            (ensemble / "thresholds.json").write_text(
                json.dumps({"joint": threshold}), encoding="utf-8"
            )
            (ensemble / "temperatures.json").write_text(
                json.dumps({"joint": temperature}), encoding="utf-8"
            )
            (ensemble / "ensemble_metrics.json").write_text(
                json.dumps(
                    {
                        "member_count": 5,
                        "member_paths": [f"member_{index}" for index in range(5)],
                        "calibrated_thresholds": {"joint": threshold},
                        "temperatures": {"joint": temperature},
                        "temperature_application": (
                            "shared temperature applied to each member logit "
                            "before probability averaging"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            (screening / "screening_summary.json").write_text(
                json.dumps({"pJ_calibrated_alias": "p_joint_calibrated"}),
                encoding="utf-8",
            )
            prediction = pd.DataFrame(
                {
                    "pJ_calibrated": [0.2, 0.8],
                    "p_joint_calibrated": [0.2, 0.8],
                    **{
                        f"p_joint_member_{index}": [0.1, 0.9]
                        for index in range(1, 6)
                    },
                }
            )
            metadata, _ = load_unknown_ensemble_calibration(
                project_root, prediction, "pJ_calibrated"
            )
            self.assertEqual(metadata["ensemble_members"], 5)
            self.assertEqual(
                metadata["calibrated_decision_threshold_joint"], threshold
            )
            with self.assertRaisesRegex(ValueError, "Cross-ensemble"):
                load_unknown_ensemble_calibration(
                    project_root,
                    prediction.drop(columns=["p_joint_member_5"]),
                    "pJ_calibrated",
                )

    def test_paired_bootstrap_is_reproducible_and_directional(self):
        labels = np.array([0] * 10 + [1] * 10)
        full = np.linspace(0.01, 0.99, 20)
        comparator = np.array(
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.85, 0.80, 0.75, 0.70]
            + [0.30, 0.25, 0.20, 0.15, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        )
        first = paired_bootstrap_difference(
            labels, full, comparator, "roc_auc", replicates=250, seed=17
        )
        second = paired_bootstrap_difference(
            labels, full, comparator, "roc_auc", replicates=250, seed=17
        )
        self.assertEqual(first, second)
        self.assertGreater(first["difference_full_minus_comparator"], 0.0)
        self.assertEqual(first["bootstrap_replicates_valid"], 250)

    def test_logic_violation_definitions(self):
        values = logic_violation_magnitudes(
            p_thermo=[0.4, 0.9],
            p_dynamic=[0.3, 0.8],
            p_joint=[0.5, 0.6],
        )
        np.testing.assert_allclose(values["V1_pJ_gt_pT"], [0.1, 0.0])
        np.testing.assert_allclose(values["V2_pJ_gt_pD"], [0.2, 0.0])
        np.testing.assert_allclose(values["V3_pJ_below_Frechet"], [0.0, 0.1])
        summary = summarize_logic_violations(
            "toy", "raw", [0.4, 0.9], [0.3, 0.8], [0.5, 0.6]
        )
        v1 = summary.loc[summary["violation"] == "V1_pJ_gt_pT"].iloc[0]
        self.assertEqual(v1["violation_count"], 1)
        self.assertAlmostEqual(v1["violation_rate"], 0.5)

    def test_reliability_and_calibration_rows_cover_every_sample(self):
        labels = np.array([0, 0, 1, 1])
        probability = np.array([0.05, 0.25, 0.75, 1.0])
        table = reliability_rows(labels, probability, fixed_bin_edges(4))
        self.assertEqual(int(table["count"].sum()), len(labels))
        metrics = calibration_metrics(
            labels,
            probability,
            threshold=0.5,
            n_bins=4,
            adaptive_edges=np.array([0.05, 0.5, 1.0]),
        )
        self.assertEqual(metrics["error_count"], 0)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertGreaterEqual(metrics["classwise_ece"], 0.0)

    def test_temperature_is_positive_and_finite(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        logits = np.array(
            [
                [-3.0, 2.0, -1.5, 1.0, 0.5, 3.0],
                [-2.5, 1.5, -1.0, 0.5, -0.2, 2.0],
            ]
        )
        temperature = fit_temperature(labels, logits)
        self.assertTrue(np.isfinite(temperature))
        self.assertGreater(temperature, 0.0)

    def test_probability_uncertainty_bins_preserve_counts(self):
        labels = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.3, 0.7, 0.9])
        uncertainty = np.array([0.1, 0.8, 0.2, 0.9])
        table = probability_uncertainty_table(
            labels,
            probability,
            uncertainty,
            threshold=0.5,
            uncertainty_edges=[0.0, 0.5, 1.0],
        )
        self.assertEqual(int(table["count"].sum()), 4)
        self.assertEqual(int(table["error_count"].sum()), 0)

    def test_quadrants_and_conditional_probability(self):
        result = quadrant_counts(
            ehull=[0.01, 0.04, 0.20, 0.30],
            dynamic=[1, 0, 1, 0],
            threshold=0.10,
        )
        self.assertEqual(result["Tplus_Dplus"], 1)
        self.assertEqual(result["Tplus_Dminus"], 1)
        self.assertEqual(result["Tminus_Dplus"], 1)
        self.assertEqual(result["Tminus_Dminus"], 1)
        self.assertAlmostEqual(result["p_Dplus_given_Tplus"], 0.5)


if __name__ == "__main__":
    unittest.main()
