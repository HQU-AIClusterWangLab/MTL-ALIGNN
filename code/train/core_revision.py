"""Strict fixed-IID adapters around the project's existing evaluation functions."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train.metrics import best_mcc_threshold, ranking_metrics
from train.run_review_supplements import paired_bootstrap_difference

SEEDS = [2025, 2026, 2027, 2028, 2029]
BETAS = [0.00, 0.01, 0.05, 0.10, 0.20, 0.50]
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_REPLICATES = 2000
TASKS = ('thermo', 'dynamic', 'joint')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest_json(value):
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True,
                                    ensure_ascii=False).encode('utf-8')).hexdigest()


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def scientific_config(config):
    """Compare resolved settings, excluding only documented administrative fields.

    Defaults are taken from common.build_objective/train_gnn, not inferred from
    the test scores. beta=0 disables the entire LC branch, including lower bound.
    Unknown settings remain in the signature and therefore fail closed.
    """
    result = copy.deepcopy(config)
    for key in ('name', 'category', 'seeds', 'experiment_file', 'project', 'paths',
                'outputs', 'ranking', 'uncertainty'):
        result.pop(key, None)
    training = result.setdefault('training', {})
    for key in ('main_seeds', 'ensemble_seeds', 'ablation_seeds', 'device'):
        training.pop(key, None)
    runtime = result.setdefault('runtime', {})
    runtime.pop('skip_completed', None)
    result.setdefault('monitor_task', 'joint')
    result.setdefault('model_type', 'alignn')
    result.setdefault('shuffle_auxiliary', False)
    result.setdefault('pretrained_backbone_path', None)
    result.setdefault('weighting', 'equal')
    result.setdefault('logic_beta', 0.0)
    result.setdefault('logic_lower_bound', True)
    if float(result['logic_beta']) == 0:
        result['logic_lower_bound'] = True  # provably inactive, audited explicitly
    if result['weighting'] != 'fixed':
        fixed = result.pop('fixed_weights', {})
        if fixed and any(float(v) != 1 for v in fixed.values()):
            raise ValueError('Non-unit ignored fixed_weights must be resolved explicitly')
    result['implementation_contract'] = {
        'optimizer': 'AdamW; model weight_decay=config; objective weight_decay=0',
        'scheduler': 'ReduceLROnPlateau(max,min_lr=1e-6)',
        'loss': 'BCEWithLogits; see weighting/class_weighting/logic_beta',
        'monitor': 'validation average_precision_score for monitor_task',
    }
    return result


def config_differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            name = f'{prefix}.{key}' if prefix else key
            if key not in left or key not in right:
                differences.append(name)
            else:
                differences.extend(config_differences(left[key], right[key], name))
        return differences
    return [] if left == right else [prefix]


def require_equivalent(left, right):
    differences = config_differences(scientific_config(left), scientific_config(right))
    if differences:
        raise ValueError('Incompatible scientific configs: ' + ', '.join(differences))


def beta_config(full, beta):
    result = copy.deepcopy(full)
    result['logic_beta'] = float(beta)
    return result


def require_ids(frame, reference=None):
    if 'material_id' not in frame or frame.material_id.isna().any() or frame.material_id.duplicated().any():
        raise ValueError('material_id must be nonempty and unique')
    result = frame.copy()
    result['material_id'] = result.material_id.astype(str)
    result = result.sort_values('material_id', kind='mergesort').reset_index(drop=True)
    if reference is not None:
        reference = require_ids(reference)
        if result.material_id.tolist() != reference.material_id.tolist():
            raise ValueError('material_id mismatch across models / frozen split')
        for task in TASKS:
            col = f'y_{task}'
            if col in reference and (col not in result or not np.array_equal(result[col], reference[col])):
                raise ValueError(f'Label mismatch for {task}')
    return result


def require_labels(frame):
    y = frame[[f'y_{t}' for t in TASKS]].to_numpy()
    if not np.isin(y, [0, 1]).all() or not np.array_equal(y[:, 2], y[:, 0] * y[:, 1]):
        raise ValueError('Labels must be binary and satisfy J=T AND D')


def validation_only(frame):
    if '_partition' not in frame or set(frame['_partition']) != {'val'}:
        raise ValueError('Fitting requires explicitly tagged validation rows only')
    require_ids(frame)
    require_labels(frame)


def fit_threshold(validation, task, column):
    validation_only(validation)
    return best_mcc_threshold(validation[f'y_{task}'], validation[column])


def hard_and_thresholds(validation):
    return {task: fit_threshold(validation, task, f'p_{task}') for task in ('thermo', 'dynamic')}


def hard_and(frame, thresholds):
    return ((frame.p_thermo.to_numpy() >= thresholds['thermo']) &
            (frame.p_dynamic.to_numpy() >= thresholds['dynamic'])).astype(int)


def simple_fusions(thermo, dynamic):
    t, d = np.asarray(thermo, float), np.asarray(dynamic, float)
    if t.shape != d.shape or not np.isfinite([t, d]).all() or np.any((t < 0) | (t > 1) | (d < 0) | (d > 1)):
        raise ValueError('Fusion requires aligned finite probabilities in [0,1]')
    return {'Product': t * d, 'Minimum': np.minimum(t, d)}


def paired_ap(left, right, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    left = require_ids(left)
    right = require_ids(right, left)
    return paired_bootstrap_difference(left.y_joint, left.score, right.score,
                                       'pr_auc', replicates, seed)


def screening_curve(labels, scores, budgets=None, random_expected=False):
    y = np.asarray(labels, int)
    if not len(y) or not np.isin(y, [0, 1]).all() or y.sum() == 0:
        raise ValueError('Screening needs nonempty binary labels and positives')
    budgets = np.arange(1, 101) / 100 if budgets is None else np.asarray(budgets, float)
    if np.any((budgets <= 0) | (budgets > 1)):
        raise ValueError('Budget must be in (0,1]')
    n, positive = len(y), int(y.sum())
    order = np.argsort(-np.asarray(scores, float), kind='mergesort') if not random_expected else None
    cumulative = np.cumsum(y[order]) if order is not None else None
    rows = []
    for fraction in budgets:
        k = min(n, max(1, int(np.ceil(fraction * n - 1e-12))))
        recovered = float(k * positive / n) if random_expected else int(cumulative[k - 1])
        # Reuse the project's precision/enrichment definitions.
        metrics = {} if random_expected else ranking_metrics(y, scores, [k])
        precision = positive / n if random_expected else metrics[f'precision_at_{k}']
        rows.append({'requested_budget_fraction': fraction, 'coverage': k / n,
                     'verification_budget_fraction': k / n, 'verified_count': k,
                     'recovered_joint_positive_count': recovered,
                     'joint_positive_recall': recovered / positive,
                     'precision': precision, 'enrichment_factor': precision / (positive / n),
                     'overall_positive_prevalence': positive / n,
                     'baseline_type': 'analytic expectation' if random_expected else 'model ranking'})
    return pd.DataFrame(rows)


def target_recall_budget(labels, scores, target, random_expected=False):
    y = np.asarray(labels, int)
    if not len(y) or y.sum() == 0 or not 0 < target <= 1:
        raise ValueError('Target recall needs positives and target in (0,1]')
    if random_expected:
        k = int(np.ceil(target * len(y) - 1e-12))
    else:
        order = np.argsort(-np.asarray(scores, float), kind='mergesort')
        required = int(np.ceil(target * y.sum() - 1e-12))
        k = int(np.searchsorted(np.cumsum(y[order]), required, side='left') + 1)
    return {'target_recall': target, 'verified_count': k, 'budget_fraction': k / len(y)}


def screening_bootstrap(frame, score_columns, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    frame = require_ids(frame)
    y = frame.y_joint.to_numpy(int)
    n = len(y)
    # One material resample per replicate shared across all methods.
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in score_columns}
    points = {}
    for method, column in score_columns.items():
        score = frame[column].to_numpy(float)
        points[method] = _screening_key_metrics(y, score)
    valid = 0
    for _ in range(replicates):
        indices = rng.integers(0, n, n)
        # Stable material-ID tie order, even when bootstrap duplicates materials.
        indices.sort(kind='mergesort')
        if y[indices].sum() == 0:
            continue
        valid += 1
        for method, column in score_columns.items():
            samples[method].append(_screening_key_metrics(y[indices], frame[column].to_numpy(float)[indices]))
    if not valid:
        raise ValueError('No valid screening bootstrap samples')
    rows = []
    for method, records in samples.items():
        array = np.asarray(records)
        for j, metric in enumerate(('recall_at_10pct', 'recall_at_20pct', 'recall_at_30pct', 'budget_at_80pct_recall')):
            rows.append({'method': method, 'metric': metric, 'point_estimate': points[method][j],
                         'bootstrap_mean': array[:, j].mean(), 'ci95_low': np.quantile(array[:, j], .025),
                         'ci95_high': np.quantile(array[:, j], .975), 'replicates_requested': replicates,
                         'replicates_valid': valid, 'bootstrap_seed': seed})
    return pd.DataFrame(rows)


def _screening_key_metrics(y, score):
    order = np.argsort(-score, kind='mergesort')
    cumulative = np.cumsum(y[order])
    k = np.ceil(np.array([.1, .2, .3]) * len(y) - 1e-12).astype(int)
    recalls = cumulative[k - 1] / y.sum()
    budget = target_recall_budget(y, score, .8)['budget_fraction']
    return [*recalls, budget]


def fit_risk_detector(validation, features, threshold, seed=BOOTSTRAP_SEED):
    validation_only(validation)
    x = validation[features].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite risk features')
    errors = ((validation.p_joint >= threshold).astype(int) != validation.y_joint).astype(int)
    if errors.nunique() != 2:
        raise ValueError('Validation errors have only one class; cannot fit risk logistic regression')
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., solver='lbfgs',
                           class_weight=None, max_iter=2000, random_state=seed))
    model.fit(x, errors)
    model.validation_material_ids_ = tuple(validation.material_id)
    model.fit_partition_ = 'val'
    return model


def immutable_output(root, signature):
    """Never overwrite a prior scientific artifact, including partial analyses."""
    root = Path(root)
    marker = root / 'COMPLETED.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')).get('signature') == signature:
        verify_completed_output(root)
        return root, True
    if root.exists() and any(root.iterdir()):
        candidate = root / 'revisions' / signature[:16]
        if (candidate / 'COMPLETED.json').exists():
            verify_completed_output(candidate)
            return candidate, True
        if candidate.exists() and any(candidate.iterdir()):
            candidate = root / 'revisions' / (signature[:16] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root = candidate
    root.mkdir(parents=True, exist_ok=True)
    return root, False


def verify_completed_output(path):
    marker = json.loads((Path(path) / 'COMPLETED.json').read_text(encoding='utf-8'))
    for name, expected in marker['files_sha256'].items():
        file = Path(path) / name
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f'Completed analysis artifact changed or missing: {file}')


def finish_output(path, signature):
    files = {str(p.relative_to(path)): sha256(p) for p in Path(path).rglob('*')
             if p.is_file() and p.name != 'COMPLETED.json'}
    write_json({'signature': signature, 'timestamp': stamp(), 'files_sha256': files},
               Path(path) / 'COMPLETED.json')
