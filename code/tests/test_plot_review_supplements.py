import unittest

import pandas as pd

from train.plot_review_supplements import require_columns, top_shift_rows


class PlotReviewSupplementTests(unittest.TestCase):
    def test_missing_schema_column_fails_closed(self):
        frame = pd.DataFrame({"status": ["ok"]})
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            require_columns(frame, {"status", "metric"}, "toy.csv")

    def test_top_shift_rows_uses_absolute_effect_and_deterministic_ties(self):
        frame = pd.DataFrame(
            {
                "record_type": ["continuous_shift"] * 4,
                "comparison": ["labeled_vs_unknown_all"] * 4,
                "feature": ["b", "a", "c", "d"],
                "standardized_mean_difference_target_minus_known": [
                    -0.8,
                    0.8,
                    0.2,
                    -0.1,
                ],
            }
        )
        result = top_shift_rows(frame, "labeled_vs_unknown_all", n=2)
        self.assertEqual(result["feature"].tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
