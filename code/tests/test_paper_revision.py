from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from models.joint_outputs import FourStateObjective, four_state_logits, product_joint_logit
from models.multitask import CrystalMultiTaskModel
from train.common import build_objective
from train.revision_evaluation import fit_calibration, probability_matrices, violation_stats
from train.run_paper_revision import check_contract, check_split, configurations, contract_for, queue_lock


class StubEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, batch):
        return batch[0]


class RevisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.cfg = {'alignn_layers': 1, 'gcn_layers': 1, 'embedding_features': 8,
                    'hidden_features': 16, 'dropout': 0.0}

    def test_four_state_marginals_and_extreme_logit_gradients(self):
        logits = torch.tensor([[1000., -1000., -1000., 999.], [-1000., 10., 20., -1000.]],
                              dtype=torch.float64, requires_grad=True)
        out = four_state_logits(logits)
        q = logits.softmax(1)
        for task, p in {'thermo': q[:, 2] + q[:, 3], 'dynamic': q[:, 1] + q[:, 3], 'joint': q[:, 3]}.items():
            torch.testing.assert_close(out[task].sigmoid(), p)
        sum(out[t].sum() for t in ('thermo', 'dynamic', 'joint')).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_four_state_ce_uses_native_state_labels(self):
        labels = torch.tensor([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]])
        z = torch.randn(4, 4, requires_grad=True)
        loss, _ = FourStateObjective()(four_state_logits(z), labels)
        torch.testing.assert_close(loss, F.cross_entropy(z, torch.arange(4)))
        labels[-1, -1] = 0
        with self.assertRaisesRegex(ValueError, 'J must equal'):
            FourStateObjective()(four_state_logits(z), labels)

    def test_product_logit_is_stable(self):
        t = torch.tensor([-1000., -3., 0., 3., 1000.], dtype=torch.float64, requires_grad=True)
        d = torch.tensor([1000., 2., 0., -3., 1000.], dtype=torch.float64, requires_grad=True)
        value = product_joint_logit(t, d)
        torch.testing.assert_close(value.sigmoid(), t.sigmoid() * d.sigmoid())
        value.sum().backward()
        self.assertTrue(torch.isfinite(t.grad).all())

    @patch('models.multitask.ALIGNNEncoder', StubEncoder)
    def test_dual_has_no_joint_head_or_joint_training_loss(self):
        model = CrystalMultiTaskModel({**self.cfg, 'output_mode': 'dual_product'}, ['thermo', 'dynamic'])
        self.assertEqual(set(model.heads), {'thermo', 'dynamic'})
        self.assertFalse(any('joint' in name for name, _ in model.named_parameters()))
        out = model((torch.randn(4, 16), None, None))
        objective = build_objective({'model': {'output_mode': 'dual_product'},
                                     'active_tasks': ['thermo', 'dynamic'], 'logic_beta': 0},
                                    {'thermo': 1., 'dynamic': 1.})
        y = torch.tensor([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]])
        loss, diagnostics = objective(out, y)
        changed = y.clone()
        changed[:, 2] = 1 - changed[:, 2]
        torch.testing.assert_close(loss, objective(out, changed)[0])
        torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(out['thermo'], y[:, 0])
                                   + F.binary_cross_entropy_with_logits(out['dynamic'], y[:, 1]))
        self.assertNotIn('loss_joint', diagnostics)

    @patch('models.multitask.ALIGNNEncoder', StubEncoder)
    def test_four_state_has_one_native_head(self):
        model = CrystalMultiTaskModel({**self.cfg, 'output_mode': 'four_state'}, ['thermo', 'dynamic', 'joint'])
        self.assertEqual(len(model.heads), 0)
        self.assertEqual(model((torch.randn(3, 16), None, None))['state_logits'].shape, (3, 4))

    def test_equal_lc_has_three_unit_weights(self):
        objective = build_objective({'model': {}, 'active_tasks': ['thermo', 'dynamic', 'joint'],
                                     'weighting': 'equal', 'logic_beta': .1, 'logic_lower_bound': True},
                                    {'thermo': 1., 'dynamic': 1., 'joint': 1.})
        out = {'thermo': torch.tensor([-2.]), 'dynamic': torch.tensor([-2.]), 'joint': torch.tensor([2.])}
        loss, diagnostic = objective(out, torch.tensor([[0., 0., 0.]]))
        expected = sum(diagnostic[f'loss_{t}'] for t in ('thermo', 'dynamic', 'joint')) + .1 * diagnostic['loss_logic']
        torch.testing.assert_close(loss, expected)
        self.assertEqual(len(list(objective.parameters())), 0)

    def test_four_state_calibration_preserves_bounds(self):
        rng = np.random.default_rng(9)
        labels = np.tile(np.arange(4), 8)
        frames = []
        for _ in range(3):
            frame = pd.DataFrame(rng.normal(size=(32, 4)) * 4, columns=[f'logit_state_{s}' for s in ('00', '01', '10', '11')])
            frame['y_thermo'], frame['y_dynamic'] = labels // 2, labels % 2
            frames.append(frame)
        temperatures = fit_calibration(frames, 'four_state')
        self.assertEqual(set(temperatures), {'four_state'})
        for temps in (None, temperatures):
            probabilities = probability_matrices(frames, 'four_state', temps)
            self.assertEqual(violation_stats(probabilities)['any_violation'], 0.)
            self.assertEqual(violation_stats({k: p.mean(0) for k, p in probabilities.items()})['any_violation'], 0.)

    def test_dual_fuses_members_before_averaging(self):
        frames = [pd.DataFrame({'p_thermo': [t], 'p_dynamic': [d],
                                'logit_thermo': [np.log(t / (1-t))], 'logit_dynamic': [np.log(d / (1-d))]})
                  for t, d in ((.9, .1), (.1, .9))]
        for temperatures in (None, {'thermo': 1., 'dynamic': 1.}):
            matrices = probability_matrices(frames, 'dual_product', temperatures)
            self.assertAlmostEqual(matrices['joint'].mean(), .09)
            self.assertEqual(violation_stats(matrices)['any_violation'], 0.)

    def test_split_guard_detects_group_leak_and_bad_joint(self):
        rows = [{'material_id': f'{part}{i}', 'split': part, 'g': f'{part}{i}',
                 'thermo_label': i // 2, 'dynamic_label': i % 2, 'joint_label': int(i == 3)}
                for part in ('train', 'val', 'test') for i in range(4)]
        split = pd.DataFrame(rows)
        metadata = split.drop(columns=['split', 'g'])
        self.assertEqual(check_split(metadata, split, 'g')['train']['n'], 4)
        split.loc[4, 'g'] = split.loc[0, 'g']
        with self.assertRaisesRegex(ValueError, 'leaking'):
            check_split(metadata, split, 'g')
        split.loc[4, 'g'] = 'val0'
        metadata.loc[3, 'joint_label'] = 0
        with self.assertRaisesRegex(ValueError, 'binary and J'):
            check_split(metadata, split, 'g')

    def test_contract_refuses_silent_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'seed'
            check_contract(root, {'code': 'a'}, create=True)
            check_contract(root, {'code': 'a'})
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                check_contract(root, {'code': 'b'})
            (root / 'COMPLETED.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                check_contract(root, {'code': 'a'})

    def test_extending_seed_subset_does_not_change_contract(self):
        environment = {k: 'v' for k in ('torch', 'torch_cuda', 'dgl', 'alignn', 'python_executable')}
        a = contract_for({'seeds': [1, 2, 3]}, 1, {}, {}, environment)
        b = contract_for({'seeds': [1, 2, 3, 4, 5]}, 1, {}, {}, environment)
        self.assertEqual(a, b)

    def test_queue_lock_prevents_two_concurrent_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lock'
            with queue_lock(path):
                with self.assertRaisesRegex(RuntimeError, 'already'):
                    with queue_lock(path):
                        pass
            with queue_lock(path):
                pass

    def test_six_configs_make_18_or_30_jobs(self):
        _, configs = configurations(Path(__file__).resolve().parents[1])
        self.assertEqual(len(configs) * 3, 18)
        self.assertEqual(sum(len(c['seeds']) for c in configs), 30)
        self.assertEqual({c['revision']['split'] for c in configs}, {'composition', 'structure'})
        for config in configs:
            self.assertEqual(config['training']['max_epochs'], 200)
            self.assertEqual(config['monitor_task'], 'joint')


if __name__ == '__main__':
    unittest.main()
