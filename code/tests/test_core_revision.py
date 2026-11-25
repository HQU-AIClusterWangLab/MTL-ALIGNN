"""CPU-only checks of scientific invariants and leakage boundaries."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from train.common import load_yaml, resolve_experiment, build_objective
from train.core_revision import (require_equivalent, require_ids, require_labels,
    hard_and_thresholds, hard_and, simple_fusions, paired_ap, screening_curve,
    target_recall_budget, screening_bootstrap, fit_risk_detector, beta_config,
    scientific_config, immutable_output, finish_output)
from train.run_core_revision import check_resume_contract, configurations, check_data, main as run_main
from train.analyze_literature_validation import analyze_literature, FIELDS

ROOT = Path(__file__).resolve().parents[1]


def frame(part='val'):
    t = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    d = np.array([0, 1, 0, 1, 1, 1, 0, 0])
    return pd.DataFrame({'material_id': [f'{part}{i}' for i in range(len(t))],
                         'y_thermo': t, 'y_dynamic': d, 'y_joint': t * d,
                         'p_thermo': [.1, .2, .7, .8, .15, .65, .2, .9],
                         'p_dynamic': [.2, .8, .3, .9, .7, .85, .1, .1],
                         'p_joint': [.1, .7, .1, .8, .2, .9, .1, .8],
                         'uT': [.1, .2, .1, .3, .3, .1, .2, .4],
                         'uD': [.2, .3, .2, .1, .4, .2, .1, .5],
                         'uJ': [.2, .3, .2, .1, .4, .2, .1, .5],
                         'r_conf': [.1, .3, .1, .2, .2, .1, .1, .2],
                         '_partition': part})


class CoreRevisionTests(unittest.TestCase):
    def setUp(self):
        self.project = load_yaml(ROOT / 'config.yaml')
        self.full = resolve_experiment(self.project, load_yaml(ROOT / 'train/experiments/main/M06_full_mtl.yaml'))

    def test_five_seed_merge_rejects_scientific_changes(self):
        for section, key, value in [('training', 'max_epochs', 201), ('training', 'amp', True),
                                    ('training', 'learning_rate', .002), ('training', 'class_weighting', 'balanced'),
                                    ('model', 'dropout', .2), ('graph', 'cutoff', 9.)]:
            changed = copy.deepcopy(self.full)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'scientific configs'):
                require_equivalent(self.full, changed)
        for key, value in [('weighting', 'equal'), ('monitor_task', 'dynamic'), ('split_file', 'random_iid_stratified.csv')]:
            changed = dict(self.full, **{key: value})
            with self.assertRaises(ValueError):
                require_equivalent(self.full, changed)

    def test_real_m04_q1_configs_equivalent(self):
        a = resolve_experiment(self.project, load_yaml(ROOT / 'train/experiments/main/M04_single_alignn.yaml'))
        b = resolve_experiment(self.project, load_yaml(ROOT / 'train/experiments/q1/Q1_joint_only_alignn_iid_2028_2029.yaml'))
        require_equivalent(a, b)

    def test_test_material_id_mismatch_and_duplicates_rejected(self):
        a, b = frame('test'), frame('test')
        b.loc[0, 'material_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'material_id mismatch'):
            require_ids(b, a)
        b.loc[0, 'material_id'] = b.loc[1, 'material_id']
        with self.assertRaises(ValueError):
            require_ids(b)

    def test_labels_must_satisfy_intersection(self):
        f = frame()
        require_labels(f)
        f.loc[0, 'y_joint'] = 1
        with self.assertRaisesRegex(ValueError, 'J=T AND D'):
            require_labels(f)

    def test_hard_and_threshold_only_validation(self):
        val, test = frame(), frame('test')
        thresholds = hard_and_thresholds(val)
        first = hard_and(test, thresholds)
        test['y_thermo'] = 1 - test.y_thermo
        test['y_dynamic'] = 1 - test.y_dynamic
        np.testing.assert_array_equal(first, hard_and(test, thresholds))
        with self.assertRaisesRegex(ValueError, 'validation'):
            hard_and_thresholds(test)
        with self.assertRaises(ValueError):
            hard_and_thresholds(pd.concat([val, test], ignore_index=True))

    def test_product_minimum_math(self):
        p = simple_fusions([.2, .8, 1], [.3, .4, 0])
        np.testing.assert_allclose(p['Product'], [.06, .32, 0])
        np.testing.assert_allclose(p['Minimum'], [.2, .4, 0])
        with self.assertRaises(ValueError):
            simple_fusions([1.2], [.3])

    def test_bootstrap_reproducible_and_id_paired(self):
        f = frame('test')[['material_id', 'y_joint']]
        a, b = f.assign(score=[.1, .2, .3, .9, .2, .8, .3, .2]), f.assign(score=[.1, .9, .3, .2, .1, .8, .3, .2])
        first = paired_ap(a, b, replicates=60, seed=37)
        second = paired_ap(a.sample(frac=1, random_state=8), b.sample(frac=1, random_state=9), replicates=60, seed=37)
        self.assertEqual(first, second)
        self.assertGreater(first['difference_full_minus_comparator'], 0)
        self.assertIn('two_sided_bootstrap_p', first)

    def test_screening_recall_monotone_and_enrichment(self):
        y = [1, 0, 1, 0]
        result = screening_curve(y, [1., .9, .8, .1])
        self.assertTrue(np.all(np.diff(result.joint_positive_recall) >= 0))
        self.assertEqual(result.iloc[-1].joint_positive_recall, 1.)
        at = screening_curve(y, [1., .9, .8, .1], [.25]).iloc[0]
        self.assertEqual(at.enrichment_factor, 2.)
        self.assertEqual(at.recovered_joint_positive_count, 1)

    def test_target_recall_minimum_attainable_count(self):
        y, scores = [0, 1, 0, 1, 1], [5, 4, 3, 2, 1]
        self.assertEqual(target_recall_budget(y, scores, .5)['verified_count'], 4)
        self.assertEqual(target_recall_budget(y, scores, .8)['verified_count'], 5)
        self.assertEqual(target_recall_budget(y, scores, 1)['budget_fraction'], 1)

    def test_random_baseline_expectation_uses_actual_budget(self):
        result = screening_curve([1, 1, 0], None, [.1, .5, 1.], random_expected=True)
        np.testing.assert_allclose(result.joint_positive_recall, result.verification_budget_fraction)
        np.testing.assert_allclose(result.enrichment_factor, 1.)

    def test_screening_material_bootstrap_reproducible(self):
        f = frame('test').assign(score=np.arange(8))
        a = screening_bootstrap(f, {'x': 'score'}, replicates=30, seed=11)
        b = screening_bootstrap(f.iloc[::-1], {'x': 'score'}, replicates=30, seed=11)
        pd.testing.assert_frame_equal(a, b)

    def test_risk_detector_fits_validation_only(self):
        val, test = frame(), frame('test')
        a = fit_risk_detector(val, ['uT', 'uD'], .5)
        self.assertEqual(a.fit_partition_, 'val')
        self.assertEqual(set(a.validation_material_ids_), set(val.material_id))
        before = a.predict_proba(test[['uT', 'uD']].to_numpy())
        test['y_joint'] = 1 - test.y_joint
        after = a.predict_proba(test[['uT', 'uD']].to_numpy())
        np.testing.assert_array_equal(before, after)
        with self.assertRaisesRegex(ValueError, 'validation'):
            fit_risk_detector(test, ['uT', 'uD'], .5)

    def test_beta_config_changes_only_beta(self):
        for beta in [0, .01, .05, .1, .2, .5]:
            cfg = beta_config(self.full, beta)
            self.assertTrue(cfg['logic_lower_bound'])
            cfg['logic_beta'] = self.full['logic_beta']
            self.assertEqual(cfg, self.full)
        entries = configurations(ROOT, self.project)
        for entry in entries:
            if entry['method'].startswith('beta='):
                cfg = copy.deepcopy(entry['config'])
                cfg['logic_beta'] = .1
                require_equivalent(cfg, self.full)

    def test_a03_zero_beta_lower_flag_inactive(self):
        old = resolve_experiment(self.project, load_yaml(ROOT / 'train/experiments/ablation/A03_no_logic.yaml'))
        require_equivalent(old, beta_config(self.full, 0))
        import torch
        logits = {t: torch.tensor([-.5, 1.0], requires_grad=True) for t in ['thermo', 'dynamic', 'joint']}
        labels = torch.tensor([[0., 1., 0.], [1., 1., 1.]])
        a = build_objective(old, {})
        b = build_objective(beta_config(self.full, 0), {})
        torch.testing.assert_close(a(logits, labels)[0], b(logits, labels)[0])

    def test_equal_lc_means_three_unit_weights(self):
        cfg = resolve_experiment(self.project, load_yaml(ROOT / 'configs/experiments/core_revision/C01_equal_lc_iid.yaml'))
        self.assertEqual(cfg['weighting'], 'equal')
        self.assertEqual(cfg['fixed_weights'], {'thermo': 1., 'dynamic': 1., 'joint': 1.})
        self.assertEqual(cfg['logic_beta'], .1)
        self.assertTrue(cfg['logic_lower_bound'])
        self.assertEqual(len(build_objective(cfg, {}).log_vars), 0)

    def test_empty_literature_no_output_or_fake_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for content in ['', ','.join(FIELDS) + '\n']:
                path = root / 'template.csv'
                path.write_text(content, encoding='utf-8')
                result = analyze_literature(path, root / 'missing_predictions', root / 'missing_pool', root / 'missing_thresholds', root / 'output')
                self.assertIsNone(result)
                self.assertFalse((root / 'output').exists())

    def test_resume_contract_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'job'
            check_resume_contract(root, {'seed': 2025, 'split': 'abc'}, create=True)
            check_resume_contract(root, {'seed': 2025, 'split': 'abc'})
            with self.assertRaisesRegex(ValueError, 'changed'):
                check_resume_contract(root, {'seed': 2025, 'split': 'xyz'})

    def test_completed_analysis_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, exists = immutable_output(Path(tmp) / 'stage', 'a' * 64)
            self.assertFalse(exists)
            (path / 'science.csv').write_text('original', encoding='utf-8')
            finish_output(path, 'a' * 64)
            self.assertTrue(immutable_output(path, 'a' * 64)[1])
            new, exists = immutable_output(path, 'b' * 64)
            self.assertNotEqual(path, new)
            self.assertFalse(exists)
            self.assertEqual((path / 'science.csv').read_text(), 'original')

    def test_changed_completed_analysis_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = immutable_output(Path(tmp) / 'stage', 'c' * 64)
            (path / 'science.csv').write_text('original', encoding='utf-8')
            finish_output(path, 'c' * 64)
            (path / 'science.csv').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed or missing'):
                immutable_output(path, 'c' * 64)

    def test_full_training_failure_stops_queue_and_records_job(self):
        previous_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                jobs = [{'method': 'Equal + LC', 'seed': seed,
                         'folder': root / 'results/core_revision/training/equal_lc' / f'seed_{seed}',
                         'config': copy.deepcopy(self.full), 'contract': {'seed': seed}}
                        for seed in [2025, 2026]]
                report = {'environment': {'cuda_available': True}, 'missing_jobs': [{'seed': 2025}, {'seed': 2026}],
                          'reused_members': 0, 'data': {}}
                with patch('sys.argv', ['run_core_revision', '--mode', 'full', '--project-root', str(root)]), \
                     patch('train.run_core_revision.preflight', return_value=({}, jobs, pd.DataFrame(), report, root)), \
                     patch('train.train_gnn.train_seed', side_effect=RuntimeError('simulated training failure')) as train:
                    with self.assertRaisesRegex(RuntimeError, 'simulated training failure'):
                        run_main()
                    self.assertEqual(train.call_count, 1)
                status = json.loads((root / 'results/core_revision/_runner/status.json').read_text())
                self.assertEqual(status['status'], 'failed')
                self.assertEqual(status['failed_job']['seed'], 2025)
                self.assertFalse(jobs[1]['folder'].exists())
                os.chdir(previous_cwd)
        finally:
            os.chdir(previous_cwd)


if __name__ == '__main__':
    unittest.main()
