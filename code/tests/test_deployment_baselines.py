from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.multitask import CrystalMultiTaskModel
from train.train_gnn import fit_conditioning_stats, validation_score


class _DatasetStub:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame


class _EncoderStub(nn.Module):
    def forward(self, _graph_batch):
        return torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            dtype=torch.float32,
        )


class DeploymentBaselineTests(unittest.TestCase):
    def test_conditioning_stats_are_fit_on_supplied_training_rows(self):
        dataset = _DatasetStub(pd.DataFrame({"ehull": [0.0, 0.1, 0.2]}))
        stats = fit_conditioning_stats(dataset, ["ehull"])
        self.assertAlmostEqual(stats["ehull"]["mean"], 0.1)
        self.assertAlmostEqual(
            stats["ehull"]["std"], np.std([0.0, 0.1, 0.2], ddof=0)
        )

    def test_validation_score_can_monitor_dynamic_head(self):
        frame = pd.DataFrame(
            {"y_dynamic": [0, 0, 1, 1], "p_dynamic": [0.1, 0.2, 0.8, 0.9]}
        )
        self.assertAlmostEqual(validation_score(frame, "dynamic"), 1.0)

    def test_conditioned_head_consumes_standardized_scalar(self):
        model = CrystalMultiTaskModel.__new__(CrystalMultiTaskModel)
        nn.Module.__init__(model)
        model.head_names = ["dynamic"]
        model.conditioning_features = ["ehull"]
        model.encoder = _EncoderStub()
        model.embedding_dropout = nn.Identity()
        model.heads = nn.ModuleDict({"dynamic": nn.Linear(5, 1)})

        outputs = model(
            (None, None, None, torch.tensor([[0.25], [-0.25]]))
        )
        self.assertEqual(tuple(outputs["dynamic"].shape), (2,))

        with self.assertRaisesRegex(ValueError, "conditioning tensor"):
            model((None, None, None))


if __name__ == "__main__":
    unittest.main()
