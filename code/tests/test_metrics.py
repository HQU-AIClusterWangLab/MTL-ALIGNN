import unittest

import numpy as np

from train.metrics import (
    best_mcc_threshold,
    classification_metrics,
    risk_coverage_curve,
    uncertainty_metrics,
)


class MetricTests(unittest.TestCase):
    def test_perfect_classifier(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.01, 0.10, 0.90, 0.99])
        threshold = best_mcc_threshold(y, p)
        metrics = classification_metrics(y, p, threshold)
        self.assertAlmostEqual(metrics["mcc"], 1.0)
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)

    def test_uncertainty_orders_errors(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.9, 0.8, 0.2])
        u = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = uncertainty_metrics(y, p, u, threshold=0.5)
        self.assertGreater(metrics["error_auroc"], 0.99)
        curve = risk_coverage_curve(y, p, u, threshold=0.5)
        self.assertEqual(len(curve), len(y))
        self.assertAlmostEqual(curve["coverage"].iloc[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
