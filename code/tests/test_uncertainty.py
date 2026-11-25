import unittest

import numpy as np

from models.uncertainty import aggregate_binary_probabilities


class UncertaintyTests(unittest.TestCase):
    def test_identical_members_have_zero_disagreement(self):
        values = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
        result = aggregate_binary_probabilities(values)
        self.assertTrue(np.allclose(result["std"], 0.0))
        self.assertTrue(np.allclose(result["mutual_information"], 0.0))

    def test_disagreement_increases_mutual_information(self):
        values = np.array([[0.01], [0.99]])
        result = aggregate_binary_probabilities(values)
        self.assertGreater(float(result["mutual_information"][0]), 0.5)


if __name__ == "__main__":
    unittest.main()
