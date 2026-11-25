import unittest

import numpy as np
import pandas as pd

from train.screen_unknown import (
    add_pj_aliases_and_logic_diagnostics,
    joint_logic_violation_summary,
    rank_all_unknown,
    screening_population_summary,
    select_top_k_by_rank,
)


class UnknownScreeningTests(unittest.TestCase):
    @staticmethod
    def ranking_frame() -> pd.DataFrame:
        # Deliberately shuffled: deterministic ties must resolve by material_id,
        # not by incoming row order or Thermo eligibility.
        return pd.DataFrame(
            {
                "material_id": [
                    "b_tie",
                    "z_high_ineligible",
                    "a_tie",
                    "c_low_ineligible",
                ],
                "thermo_label": [1, 0, 1, 0],
                "p_joint_mean": [0.79, 0.94, 0.79, 0.19],
                "p_joint_calibrated": [0.80, 0.95, 0.80, 0.20],
                "p_thermo_calibrated": [0.90, 0.97, 0.90, 0.40],
                "p_dynamic_calibrated": [0.90, 0.96, 0.90, 0.50],
                "p_joint_std": [0.10, 0.20, 0.10, 0.90],
            }
        )

    def rank(self, frame: pd.DataFrame) -> pd.DataFrame:
        return rank_all_unknown(
            frame,
            uncertainty_column="p_joint_std",
            uncertainty_lower=0.0,
            uncertainty_upper=1.0,
            selected_lambda=0.0,
            top_k=2,
        )

    def test_all_unknown_rows_are_ranked_with_stable_tie_breaks(self):
        ranked = self.rank(self.ranking_frame())
        by_id = ranked.set_index("material_id")

        # The highest-pJ row is Thermo-ineligible but must still rank first.
        self.assertEqual(
            int(by_id.loc["z_high_ineligible", "probability_rank"]),
            1,
        )
        # Equal pJ and uncertainty are resolved lexically by material_id.
        self.assertEqual(int(by_id.loc["a_tie", "probability_rank"]), 2)
        self.assertEqual(int(by_id.loc["b_tie", "probability_rank"]), 3)

        for column in ["probability_rank", "risk_rank", "uncertainty_rank"]:
            self.assertEqual(
                sorted(ranked[column].astype(int).tolist()),
                [1, 2, 3, 4],
            )
            self.assertTrue(ranked[column].is_unique)

        self.assertTrue((ranked["ranking_scope"] == "all_unknown").all())
        self.assertTrue(np.allclose(ranked["pJ"], ranked["p_joint_mean"]))
        self.assertTrue(
            np.allclose(ranked["pJ_calibrated"], ranked["p_joint_calibrated"])
        )

        probability_top = select_top_k_by_rank(
            ranked,
            "probability_rank",
            2,
        )
        self.assertEqual(
            probability_top["material_id"].tolist(),
            ["z_high_ineligible", "a_tie"],
        )

        # Reordering input rows cannot change any material's rank.
        reordered = self.rank(
            self.ranking_frame().sample(frac=1.0, random_state=42)
        )
        expected = ranked.set_index("material_id")[
            ["probability_rank", "risk_rank", "uncertainty_rank"]
        ].sort_index()
        actual = reordered.set_index("material_id")[expected.columns].sort_index()
        pd.testing.assert_frame_equal(expected, actual)

    def test_eligible_subset_is_diagnostic_only_and_may_be_empty(self):
        ranked = self.rank(self.ranking_frame())
        population = screening_population_summary(ranked)
        self.assertEqual(population["unknown_count"], 4)
        self.assertEqual(population["ranking_pool_count"], 4)
        self.assertEqual(population["ranking_scope"], "all_unknown")
        self.assertEqual(population["thermo_eligible_count"], 2)
        self.assertEqual(population["thermo_ineligible_count"], 2)

        no_eligible = self.ranking_frame().copy()
        no_eligible["thermo_label"] = 0
        ranked_no_eligible = self.rank(no_eligible)
        empty_population = screening_population_summary(ranked_no_eligible)
        self.assertEqual(empty_population["ranking_pool_count"], 4)
        self.assertEqual(empty_population["thermo_eligible_count"], 0)
        self.assertEqual(
            sorted(ranked_no_eligible["probability_rank"].tolist()),
            [1, 2, 3, 4],
        )

    def test_calibrated_pj_logic_flags_and_magnitudes(self):
        frame = pd.DataFrame(
            {
                "material_id": ["above_marginals", "below_frechet", "feasible"],
                "p_joint_mean": [0.80, 0.60, 0.75],
                "p_joint_calibrated": [0.80, 0.60, 0.75],
                "p_thermo_calibrated": [0.60, 0.90, 0.90],
                "p_dynamic_calibrated": [0.70, 0.80, 0.80],
            }
        )
        diagnosed = add_pj_aliases_and_logic_diagnostics(frame)
        by_id = diagnosed.set_index("material_id")

        self.assertTrue(by_id.loc["above_marginals", "pJ_gt_pT_violation"])
        self.assertTrue(by_id.loc["above_marginals", "pJ_gt_pD_violation"])
        self.assertAlmostEqual(
            float(
                by_id.loc[
                    "above_marginals",
                    "pJ_gt_pT_violation_magnitude",
                ]
            ),
            0.20,
        )
        self.assertAlmostEqual(
            float(
                by_id.loc[
                    "above_marginals",
                    "pJ_gt_pD_violation_magnitude",
                ]
            ),
            0.10,
        )

        self.assertTrue(
            by_id.loc["below_frechet", "pJ_below_frechet_violation"]
        )
        self.assertAlmostEqual(
            float(
                by_id.loc[
                    "below_frechet",
                    "pJ_below_frechet_violation_magnitude",
                ]
            ),
            0.10,
        )
        self.assertFalse(by_id.loc["feasible", "pJ_any_logic_violation"])

        summary = joint_logic_violation_summary(diagnosed)
        self.assertEqual(summary["evaluated_count"], 3)
        self.assertEqual(summary["pJ_gt_pT_count"], 1)
        self.assertEqual(summary["pJ_gt_pD_count"], 1)
        self.assertEqual(summary["pJ_below_frechet_count"], 1)
        self.assertEqual(summary["any_violation_count"], 2)
        self.assertAlmostEqual(summary["any_violation_fraction"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
