import unittest

import torch

from models.losses import MultiTaskObjective


class LossTests(unittest.TestCase):
    def test_logic_penalty_zero_for_consistent_probabilities(self):
        objective = MultiTaskObjective(
            active_tasks=["thermo", "dynamic", "joint"],
            weighting="equal",
            positive_weights={"thermo": 1, "dynamic": 1, "joint": 1},
            logic_beta=1.0,
            logic_lower_bound=True,
        )
        logits = {
            "thermo": torch.tensor([2.0, -2.0]),
            "dynamic": torch.tensor([2.0, -2.0]),
            "joint": torch.tensor([1.5, -3.0]),
        }
        labels = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.float32)
        _loss, diagnostics = objective(logits, labels)
        self.assertLess(float(diagnostics["loss_logic"]), 1e-6)

    def test_logic_penalty_positive_for_joint_above_marginal(self):
        objective = MultiTaskObjective(
            active_tasks=["thermo", "dynamic", "joint"],
            weighting="equal",
            positive_weights={"thermo": 1, "dynamic": 1, "joint": 1},
            logic_beta=1.0,
            logic_lower_bound=True,
        )
        logits = {
            "thermo": torch.tensor([-2.0]),
            "dynamic": torch.tensor([-2.0]),
            "joint": torch.tensor([2.0]),
        }
        labels = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        _loss, diagnostics = objective(logits, labels)
        self.assertGreater(float(diagnostics["loss_logic"]), 0.0)


if __name__ == "__main__":
    unittest.main()
