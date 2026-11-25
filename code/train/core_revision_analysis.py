"""Core evidence chain, using the existing calibration/bootstrap/risk machinery."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_curve)

from train.aggregate_ensemble import aggregate, logical_violation_rates
from train.revision_evaluation import fit_calibration, probability_matrices
from train.metrics import classification_metrics, risk_coverage_curve, uncertainty_metrics
from train.core_revision import (SEEDS, BETAS, TASKS, BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES,
    require_ids, validation_only, fit_threshold, hard_and_thresholds, hard_and,
    simple_fusions, paired_ap, screening_curve, target_recall_budget,
    screening_bootstrap, fit_risk_detector, sha256, digest_json, write_json,
    immutable_output, finish_output, stamp)

MAIN_METHODS = ('Single-task Joint', 'Equal MTL', 'Equal + LC', 'Full MTL')
COLORS = ['#4878A8', '#EE854A', '#6ACC64', '#956CB4', '#D65F5F', '#8C8C8C']


def table(records, path, columns=None):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records, columns=columns)
    frame.to_csv(path, index=False)
    return frame


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(str(stem) + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
    plt.close(fig)


def metric_row(labels, probability, threshold):
    result = classification_metrics(labels, probability, threshold)
    result['ap'] = result.pop('pr_auc')
    return result


def ensemble(members):
    seeds = sorted(members)
    val_frames = [members[s]['frames']['val'] for s in seeds]
    test_frames = [members[s]['frames']['test'] for s in seeds]
    for f in val_frames:
        validation_only(f)
    temperatures = fit_calibration(val_frames, 'independent')
    validation = aggregate(val_frames, temperatures)
    validation['_partition'] = 'val'
    thresholds = {stage: {t: fit_threshold(validation, t, f'p_{t}_{suffix}')
                         for t in temperatures}
                  for stage, suffix in [('raw', 'mean'), ('calibrated', 'calibrated')]}
    # All fit parameters are frozen before any test metric is calculated.
    test = aggregate(test_frames, temperatures)
    test['_partition'] = 'test'
    for part, frame, frames in [('val', validation, val_frames), ('test', test, test_frames)]:
        calibrated = probability_matrices(frames, 'independent', temperatures)
        for t, matrix in calibrated.items():
            frame[f'u_{t}_calibrated'] = matrix.std(axis=0, ddof=1)
    return {'val': validation, 'test': test, 'temperatures': temperatures,
            'thresholds': thresholds, 'seeds': seeds}


def analysis_manifest(report, **extra):
    return {**report, 'analysis_timestamp': stamp(), **extra}


def main_analysis(path, sources, ensembles, audit, report):
    table(audit, path / 'configuration_audit.csv')
    member_rows, summary, ensemble_rows = [], [], []
    for method in MAIN_METHODS:
        members = sources[method]
        for seed, data in sorted(members.items()):
            # Training already froze this raw threshold from validation. Reuse
            # the recorded individual-run metrics, not ensemble metrics.
            metric = dict(data['metrics']['test_joint'])
            metric['ap'] = metric.pop('pr_auc')
            member_rows.append({'method': method, 'seed': seed, 'probability_scale': 'raw',
                                'source_dir': str(data['folder']), **metric})
        subset = [r for r in member_rows if r['method'] == method]
        record = {'method': method, 'n_seeds': len(subset),
                  'status': 'complete' if sorted(members) == SEEDS else 'missing_seeds',
                  'seeds': ','.join(map(str, sorted(members))), 'probability_scale': 'raw', 'sd_ddof': 1}
        for metric in ('ap', 'roc_auc', 'mcc', 'f1', 'balanced_accuracy', 'brier', 'nll'):
            values = np.array([r[metric] for r in subset], float)
            record[f'joint_{metric}_mean'] = values.mean() if len(values) else np.nan
            record[f'joint_{metric}_sd'] = values.std(ddof=1) if len(values) > 1 else np.nan
        summary.append(record)
    predictions = path / 'ensemble_predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    for method, bundle in ensembles.items():
        name = method.lower().replace(' ', '_').replace('+', 'plus').replace('-', '_')
        for part in ('val', 'test'):
            table(bundle[part], predictions / f'{name}_{part}.csv')
        write_json({'calibration_fit_split': 'val', 'threshold_fit_split': 'val',
                    'temperatures': bundle['temperatures'], 'thresholds': bundle['thresholds'],
                    'seeds': bundle['seeds'], 'application': 'shared task temperature on each member logit, then mean'},
                   predictions / f'{name}_calibration.json')
        if method not in MAIN_METHODS:
            continue
        for stage, suffix in [('raw', 'mean'), ('calibrated', 'calibrated')]:
            test = bundle['test']
            ensemble_rows.append({'method': method, 'n_members': len(bundle['seeds']),
                                  'probability_scale': stage, **metric_row(test.y_joint,
                                  test[f'p_joint_{suffix}'], bundle['thresholds'][stage]['joint'])})
    table(member_rows, path / 'member_metrics.csv')
    table(summary, path / 'five_seed_summary.csv')
    table(ensemble_rows, path / 'ensemble_metrics.csv')
    write_json(analysis_manifest(report, individual_metric_scale='raw',
                                ensemble_metric_scales=['raw', 'calibrated'],
                                missing_seeds={m: sorted(set(SEEDS) - set(sources[m])) for m in MAIN_METHODS}),
               path / 'manifest.json')


def construction_analysis(path, ensembles, report):
    available = set(ensembles)
    if ensembles:
        ref = next(iter(ensembles.values()))
    else:
        labels = pd.read_csv(report['data']['supervised_file'], dtype={'material_id': str})
        split = pd.read_csv(report['data']['split_file'], dtype={'material_id': str})
        labels = labels.rename(columns={f'{t}_label': f'y_{t}' for t in TASKS})
        ref = {part: require_ids(labels[labels.material_id.isin(split.loc[split.split == part, 'material_id'])])
               for part in ('val', 'test')}
    frames = {p: ref[p][['material_id', *[f'y_{t}' for t in TASKS]]].copy().assign(_partition=p)
              for p in ('val', 'test')}
    thresholds, calibration, methods = {}, {}, []
    for method, source in [('Direct Joint', 'Single-task Joint'), ('Full MTL', 'Full MTL')]:
        if source in available:
            bundle = ensembles[source]
            for p in frames:
                frames[p][method] = require_ids(bundle[p], frames[p]).p_joint_calibrated
            thresholds[method] = bundle['thresholds']['calibrated']['joint']
            calibration[method] = {'temperatures': bundle['temperatures'], 'source': source}
            methods.append(method)
    and_thresholds = None
    if {'T-only', 'D-only'} <= available:
        td = {}
        for part in frames:
            td[part] = frames[part].copy()
            td[part]['p_thermo'] = require_ids(ensembles['T-only'][part], frames[part]).p_thermo_calibrated
            td[part]['p_dynamic'] = require_ids(ensembles['D-only'][part], frames[part]).p_dynamic_calibrated
            for method, score in simple_fusions(td[part].p_thermo, td[part].p_dynamic).items():
                frames[part][method] = score
        and_thresholds = hard_and_thresholds(td['val'])
        for method in ('Product', 'Minimum'):
            thresholds[method] = fit_threshold(frames['val'], 'joint', method)
            methods.append(method)
        for part in frames:
            frames[part]['Hard AND'] = hard_and(td[part], and_thresholds)
            frames[part]['p_thermo_separate'] = td[part].p_thermo
            frames[part]['p_dynamic_separate'] = td[part].p_dynamic
        calibration['separate_T_D'] = {t: ensembles[source]['temperatures'][t]
                                      for t, source in [('thermo', 'T-only'), ('dynamic', 'D-only')]}
        calibration['fusion'] = 'Product/minimum of separately calibrated T-only/D-only ensemble probabilities; no additional fusion calibration'
    write_json({'fit_split': 'val', 'joint_classification': thresholds, 'hard_and': and_thresholds}, path / 'thresholds.json')
    write_json({'fit_split': 'val', 'probability_scale': 'calibrated', **calibration}, path / 'calibration.json')
    rows = []
    for method in ('Direct Joint', 'Hard AND', 'Product', 'Minimum', 'Full MTL'):
        if method in methods:
            row = metric_row(frames['test'].y_joint, frames['test'][method], thresholds[method])
            rows.append({'method': method, 'status': 'complete', 'score_type': 'continuous', **row})
        elif method == 'Hard AND' and and_thresholds is not None:
            y, pred = frames['test'].y_joint, frames['test'][method]
            rows.append({'method': method, 'status': 'complete', 'score_type': 'binary decision only',
                         'mcc': matthews_corrcoef(y, pred), 'f1': f1_score(y, pred, zero_division=0),
                         'balanced_accuracy': balanced_accuracy_score(y, pred)})
        else:
            rows.append({'method': method, 'status': 'missing_training'})
    table(rows, path / 'joint_construction_comparison.csv')
    table(frames['test'], path / 'test_predictions.csv')
    table(frames['val'], path / 'validation_predictions.csv')
    paired = []
    if 'Full MTL' in methods:
        left = frames['test'][['material_id', 'y_joint', 'Full MTL']].rename(columns={'Full MTL': 'score'})
        for method in ('Direct Joint', 'Product', 'Minimum'):
            if method not in methods:
                continue
            right = frames['test'][['material_id', 'y_joint', method]].rename(columns={method: 'score'})
            paired.append({'comparison': f'Full MTL minus {method}', 'metric': 'AP',
                           'bootstrap_seed': BOOTSTRAP_SEED, **paired_ap(left, right)})
    if paired:
        table(paired, path / 'paired_bootstrap_full_vs_alternatives.csv')
    else:
        table([{'status': 'missing_training', 'comparison': 'Full MTL versus alternatives'}],
              path / 'paired_bootstrap_full_vs_alternatives.csv')
    write_json(analysis_manifest(report, hard_and_ranking_metrics='not defined; intentionally omitted',
                                bootstrap='IID material resample, paired by material_id; percentile CI; sign-tail p descriptive, not null-centered'),
               path / 'manifest.json')


def beta_analysis(path, sources, report):
    rows, summaries = [], []
    for beta in BETAS:
        members = sources['Full MTL'] if beta == .1 else sources[f'beta={beta:.2f}']
        members = {s: v for s, v in members.items() if s in SEEDS[:3]}
        for seed, item in sorted(members.items()):
            val, test = item['frames']['val'], item['frames']['test']
            validation_only(val)
            temps = fit_calibration([val], 'independent')
            for stage, used in [('raw', {}), ('calibrated', temps)]:
                val_m = probability_matrices([val], 'independent', used)
                test_m = probability_matrices([test], 'independent', used)
                vf = val.copy()
                vf['score'] = val_m['joint'][0]
                threshold = fit_threshold(vf, 'joint', 'score')
                violation_frame = pd.DataFrame({f'p_{t}_mean': test_m[t][0] for t in TASKS})
                rows.append({'beta': beta, 'seed': seed, 'probability_scale': stage,
                             'source_dir': str(item['folder']),
                             'joint_temperature': temps['joint'] if used else 1.,
                             **metric_row(test.y_joint, test_m['joint'][0], threshold),
                             **logical_violation_rates(violation_frame, 'mean')})
        for stage in ('raw', 'calibrated'):
            part = [r for r in rows if r['beta'] == beta and r['probability_scale'] == stage]
            row = {'beta': beta, 'probability_scale': stage, 'n_seeds': len(part),
                   'status': 'complete' if len(part) == 3 else 'missing_seeds', 'sd_ddof': 1}
            for metric in ('ap', 'mcc', 'joint_gt_thermo', 'joint_gt_dynamic', 'frechet_lower_bound', 'any_violation'):
                values = np.array([r[metric] for r in part])
                label = f'joint_{metric}' if metric in ('ap', 'mcc') else metric
                row[label + '_mean'] = values.mean() if len(values) else np.nan
                row[label + '_sd'] = values.std(ddof=1) if len(values) > 1 else np.nan
            summaries.append(row)
    table(rows, path / 'beta_sweep_member_metrics.csv')
    summary = table(summaries, path / 'beta_sweep_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    raw = summary[summary.probability_scale == 'raw']
    axes[0].errorbar(raw.beta, raw.joint_ap_mean, yerr=raw.joint_ap_sd, fmt='o-', capsize=3, color=COLORS[0])
    axes[0].set(ylabel='Joint AP (mean ± SD)', xlabel='Logic beta', ylim=(0, 1))
    for metric, label in [('joint_gt_thermo', 'pJ > pT'), ('joint_gt_dynamic', 'pJ > pD'),
                          ('frechet_lower_bound', 'pJ below lower bound'), ('any_violation', 'Any violation')]:
        axes[1].plot(raw.beta, raw[metric + '_mean'], 'o-', label=label)
    axes[1].set(ylabel='Logical violation rate (raw)', xlabel='Logic beta', ylim=(0, 1))
    axes[1].legend(fontsize=8)
    for ax in axes:
        # Keep the entire prespecified sweep visible even before missing jobs finish.
        ax.set_xlim(-.01, .51)
        ax.set_xticks([0, .1, .2, .3, .4, .5])
        ax.grid(alpha=.2)
    fig.suptitle('Prespecified beta sweep; missing configurations are gaps')
    save_figure(fig, path / 'beta_vs_ap_and_violation')
    write_json(analysis_manifest(report, primary_probability_scale='raw', tolerance=1e-10,
                                beta_selection='none; no test-based optimum selected',
                                lower_bound='max(0,pT+pD-1); nonnegative probabilities'), path / 'manifest.json')


def screening_analysis(path, construction_path, report):
    frame = pd.read_csv(construction_path / 'test_predictions.csv', dtype={'material_id': str})
    frame = require_ids(frame)
    methods = [m for m in ('Direct Joint', 'Product', 'Minimum', 'Full MTL') if m in frame]
    curves, budgets, enrichment = [], [], []
    for method in ['Random expected', *methods]:
        expected = method == 'Random expected'
        score = np.zeros(len(frame)) if expected else frame[method].to_numpy(float)
        curve = screening_curve(frame.y_joint, score, random_expected=expected).assign(method=method)
        curves.append(curve)
        for target in (.5, .8, .9):
            budgets.append({'method': method, **target_recall_budget(frame.y_joint, score, target, expected)})
        enrichment.append(curve[curve.requested_budget_fraction.isin([.1, .2, .3])])
    curve = table(pd.concat(curves, ignore_index=True), path / 'screening_curve.csv')
    table(budgets, path / 'budget_at_target_recall.csv')
    table(pd.concat(enrichment, ignore_index=True), path / 'enrichment_summary.csv')
    if methods:
        ci = screening_bootstrap(frame, {m: m for m in methods})
    else:
        ci = pd.DataFrame(columns=['method', 'metric', 'point_estimate', 'ci95_low', 'ci95_high'])
    table(ci, path / 'screening_bootstrap_ci.csv')
    for column, name, ylabel in [('joint_positive_recall', 'screening_recall_vs_budget', 'Recovered Joint-positive fraction'),
                                  ('enrichment_factor', 'enrichment_vs_budget', 'Enrichment factor')]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for method, part in curve.groupby('method', sort=False):
            ax.plot(part.verification_budget_fraction, part[column], label=method,
                    linestyle='--' if method == 'Random expected' else '-')
        ax.set(xlabel='Candidate verification budget fraction', ylabel=ylabel, xlim=(0, 1))
        ax.set_ylim(0, 1 if column == 'joint_positive_recall' else max(1, float(curve[column].max()) * 1.05))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        save_figure(fig, path / name)
    write_json(analysis_manifest(report, budget_rounding='ceil(fraction*N); reported coverage=k/N',
                                ties='stable material_id ascending; bootstrap repeats preserve this order',
                                random_baseline='analytic expectation: recall=k/N, precision=prevalence; no Monte Carlo CI',
                                interpretation='candidate counts/budget only; no measured DFT wall-time saving'),
               path / 'manifest.json')


def risk_analysis(path, bundle, report):
    if bundle is None:
        for name in ('error_detection_comparison', 'risk_coverage_comparison', 'validation_risk_model_coefficients'):
            table([{'status': 'missing Full MTL 5-member ensemble'}], path / f'{name}.csv')
        return
    frames = {}
    for part in ('val', 'test'):
        frame = bundle[part].copy()
        frame['p_joint'] = frame.p_joint_calibrated
        for t, short in [('thermo', 'uT'), ('dynamic', 'uD'), ('joint', 'uJ')]:
            frame[short] = frame[f'u_{t}_calibrated']
        frame['r_conf'] = 1 - np.maximum(frame.p_joint, 1 - frame.p_joint)
        frame['max_uT_uD'] = np.maximum(frame.uT, frame.uD)
        frames[part] = frame
    threshold = bundle['thresholds']['calibrated']['joint']
    scores = {name: frames['test'][column].to_numpy(float) for name, column in
              [('Probability uncertainty', 'r_conf'), ('Joint disagreement', 'uJ'), ('Task max disagreement', 'max_uT_uD')]}
    coefficients, failures = [], []
    for method, features in [('Learned task risk', ['uT', 'uD']), ('Combined risk', ['uT', 'uD', 'uJ', 'r_conf'])]:
        try:
            model = fit_risk_detector(frames['val'], features, threshold)
        except ValueError as exc:
            if 'only one class' not in str(exc):
                raise
            failures.append({'method': method, 'status': str(exc)})
            continue
        scaler, estimator = model.steps[0][1], model.steps[1][1]
        for feature, mean, scale, coef in zip(features, scaler.mean_, scaler.scale_, estimator.coef_[0]):
            coefficients.append({'method': method, 'feature': feature, 'validation_mean': mean,
                                 'validation_scale': scale, 'standardized_coefficient': coef,
                                 'raw_coefficient': coef / scale, 'intercept_standardized': estimator.intercept_[0],
                                 'intercept_raw': estimator.intercept_[0] - np.sum(estimator.coef_[0] * scaler.mean_ / scaler.scale_),
                                 'fit_split': 'val', 'random_state': BOOTSTRAP_SEED, 'C': 1.0,
                                 'solver': 'lbfgs', 'class_weight': 'none', 'n_validation': len(frames['val'])})
        scores[method] = model.predict_proba(frames['test'][features].to_numpy(float))[:, 1]
    table(coefficients, path / 'validation_risk_model_coefficients.csv')
    write_json({'fit_split': 'val', 'threshold_fit_split': 'val', 'joint_threshold': threshold,
                'features': {'Learned task risk': ['uT', 'uD'], 'Combined risk': ['uT', 'uD', 'uJ', 'r_conf']},
                'hyperparameters': {'C': 1., 'solver': 'lbfgs', 'class_weight': None, 'max_iter': 2000, 'random_state': BOOTSTRAP_SEED},
                'model_selection': 'none; prespecified models', 'uncertainty_scale': 'calibrated member probabilities, std ddof=1'},
               path / 'frozen_risk_models.json')
    test = frames['test']
    errors = ((test.p_joint >= threshold).astype(int) != test.y_joint).astype(int)
    comparisons, curves, roc_pr = list(failures), [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig_rc, ax_rc = plt.subplots(figsize=(6.5, 4.5))
    predictions = test[['material_id', 'y_joint', 'p_joint', 'uT', 'uD', 'uJ', 'r_conf']].copy()
    predictions['error'] = errors
    for method, score in scores.items():
        comparisons.append({'method': method, 'status': 'complete',
                            **uncertainty_metrics(test.y_joint, test.p_joint, score, threshold)})
        curve = risk_coverage_curve(test.y_joint, test.p_joint, score, threshold).assign(method=method)
        curves.append(curve)
        ax_rc.plot(curve.coverage, curve.risk, label=method)
        predictions[method] = score
        if errors.nunique() == 2:
            fpr, tpr, _ = roc_curve(errors, score)
            precision, recall, _ = precision_recall_curve(errors, score)
            axes[0].plot(fpr, tpr, label=method)
            axes[1].plot(recall, precision, label=method)
            roc_pr += [{'method': method, 'curve': 'ROC', 'x': x, 'y': y} for x, y in zip(fpr, tpr)]
            roc_pr += [{'method': method, 'curve': 'PR', 'x': x, 'y': y} for x, y in zip(recall, precision)]
    table(comparisons, path / 'error_detection_comparison.csv')
    table(pd.concat(curves, ignore_index=True), path / 'risk_coverage_comparison.csv')
    table(predictions, path / 'test_risk_predictions.csv')
    table(roc_pr, path / 'risk_roc_pr_curves.csv')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].axhline(errors.mean(), linestyle='--', color='gray')
    for ax, xlabel, ylabel in zip(axes, ['False-positive rate', 'Error recall'], ['True-positive rate', 'Error precision']):
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    save_figure(fig, path / 'risk_detection_roc_pr')
    ax_rc.set(xlabel='Retained coverage', ylabel='Classification error risk', xlim=(0, 1), ylim=(0, 1))
    ax_rc.legend(fontsize=8)
    ax_rc.grid(alpha=.2)
    save_figure(fig_rc, path / 'risk_coverage')
    write_json(analysis_manifest(report, primary_task_disagreement='max(uT,uD)',
                                probability_scale='calibrated for both confidence and all disagreements',
                                aurc='mean risk over all attainable coverages',
                                risk_at_coverage='floor(coverage*N) retained; ascending risk, material_id tie-break',
                                threshold=threshold), path / 'manifest.json')


def summary_analysis(path, stages, report):
    mapping = {'main_5seed_table': ('01_main_5seed', 'five_seed_summary.csv'),
               'joint_construction_table': ('02_joint_construction', 'joint_construction_comparison.csv'),
               'beta_sensitivity_table': ('03_beta_sweep', 'beta_sweep_summary.csv'),
               'screening_efficiency_table': ('04_screening_efficiency', 'budget_at_target_recall.csv'),
               'risk_increment_table': ('05_risk_increment', 'error_detection_comparison.csv'),
               'paired_bootstrap_table': ('02_joint_construction', 'paired_bootstrap_full_vs_alternatives.csv')}
    numbers = {}
    for name, (stage, filename) in mapping.items():
        frame = pd.read_csv(stages[stage] / filename)
        table(frame, path / f'{name}.csv')
        numbers[name] = frame.to_dict('records')
    for name, file in [('screening_bootstrap', 'screening_bootstrap_ci.csv'), ('enrichment', 'enrichment_summary.csv')]:
        numbers[name] = pd.read_csv(stages['04_screening_efficiency'] / file).to_dict('records')
    numbers.update(status='complete' if not report['missing_jobs'] else 'partial_missing_training',
                   missing_jobs=report['missing_jobs'], bootstrap_seed=BOOTSTRAP_SEED,
                   bootstrap_replicates=BOOTSTRAP_REPLICATES, individual_scale='raw',
                   ensemble_scale='calibrated', beta_primary_scale='raw')
    write_json(numbers, path / 'key_numbers.json')
    write_json(analysis_manifest(report, stage_directories=stages, summary_directory=path,
                                analysis_code_sha256={str(p): sha256(p) for p in Path(__file__).parent.glob('*core_revision*.py')}),
               path / 'result_manifest.json')
    missing = '\n'.join(f"- {j['method']}, seed={j['seed']}" for j in report['missing_jobs']) or '- None'
    readme = f'''# Core supplementary experiments on the fixed IID split

Status: {'Complete' if not report['missing_jobs'] else 'Partially complete; missing training runs are listed below'}. Only the existing iid_split.csv is used; the data were not repartitioned.

## Questions addressed by the tables

- main_5seed_table.csv: mean±SD (ddof=1) of raw-probability individual-run metrics for four methods on matching seeds. Missing seeds are marked explicitly; incomplete results are not presented as complete five-seed conclusions. Ensemble metrics are reported separately in 01_main_5seed/ensemble_metrics.csv.
- joint_construction_table.csv: compare direct Joint prediction with Hard AND, Product, and Minimum fusion of independent T-only/D-only ensembles. AP/ROC-AUC are not calculated for Hard AND.
- paired_bootstrap_table.csv: AP of Full MTL minus each alternative, strictly paired by material ID, with 2000 bootstrap samples and 95% percentile CIs. The two-sided sign-tail probability is a descriptive bootstrap quantity without multiple-testing correction; it is not a stand-alone confirmatory significance test.
- beta_sensitivity_table.csv: relationships between beta and AP/MCC/logical-violation rates. Core results use raw probabilities; calibrated results are separate. Missing points are not interpolated, and beta is not selected using test data.
- screening_efficiency_table.csv: fraction of candidates requiring validation to recover 50%/80%/90% of true positives. Full curves, 10/20/30% enrichment, and material-bootstrap CIs are in 04_screening_efficiency. This measures candidate-count budgets, not actual DFT runtime savings.
- risk_increment_table.csv: whether task-decomposed risk adds information for error detection and selective risk. All methods are retained, without a predetermined ranking conclusion.

## Reused and new results

Reused sources are recorded for every seed in configuration_audit.csv and the manifest. Scientific configurations are checked before merging M04 and Q1 Joint results; omitted monitor_task=joint values are normalized to the code default. A03 has beta=0, so the logical branch is not executed and lower_bound flag differences do not change its loss. Beta configurations retain lower_bound=true and vary beta only. Existing bootstrap, temperature, ensemble, and risk-coverage implementations are reused.

Equal+LC and unfinished beta runs require new training; other completed, eligible models are skipped. Joint construction, screening, risk, and bootstrap analyses post-process existing frozen predictions. Actual new-training sources are under core_revision/training. Currently missing jobs:

{missing}

## Leakage prevention and provenance

Temperatures, thresholds, risk normalization, and LogisticRegression are fitted on validation data only. Test data are used only for frozen evaluation and bootstrap analysis. The risk model fixes C=1, lbfgs, and random_state={BOOTSTRAP_SEED}, without test-based tuning. Main ensemble and risk analyses consistently use calibrated probabilities; member summaries and core beta results use raw probabilities in separate tables. No successful-seed selection is applied.

Historical runs did not save training-time code/data content hashes, so byte-for-byte historical code identity cannot be established retrospectively. The complete resolved configuration, environment, actual batch size, split, materials, and labels have been checked in the current audit. New results preserve run_contract and current code/data fingerprints.

## Execution and result versions

VS Code task 04 performs preflight checks, 05 runs the complete CUDA queue, and 06 performs analysis only. Repeated calls skip completed analyses with identical inputs. When seeds are added, old scientific files are retained and new versions are written to revisions/<signature>/ under each analysis directory. Consult the analysis entry in _runner/status.json and this result_manifest.json for the latest paths. Training failures stop execution immediately and record failed_job while preserving the original failure log. An empty literature template exits normally without generating literature evidence.
'''
    (path / 'README.md').write_text(readme, encoding='utf-8')


def analyze(root, sources, audit, report):
    output = root / 'results/core_revision'
    code = {str(p.relative_to(root)): sha256(p) for p in sorted((root / 'train').glob('*.py'))}
    signature = digest_json({'inputs': report['input_files_sha256'], 'data': report['data'], 'code': code,
                             'seeds': SEEDS, 'betas': BETAS, 'bootstrap_seed': BOOTSTRAP_SEED})
    names = ('01_main_5seed', '02_joint_construction', '03_beta_sweep',
             '04_screening_efficiency', '05_risk_increment', 'summary')
    plans = {name: immutable_output(output / name, signature) for name in names}
    stages = {name: path for name, (path, _) in plans.items()}
    if all(complete for _, complete in plans.values()):
        for path in stages.values():
            print(f'SKIP completed analysis: {path}', flush=True)
        return {name: str(path) for name, path in stages.items()}
    ensembles = {}
    # No ensemble is labeled as five-member unless all five common seeds exist.
    for method in (*MAIN_METHODS, 'T-only', 'D-only'):
        if sorted(sources[method]) == SEEDS:
            print(f'Calibrating {method}: validation only', flush=True)
            ensembles[method] = ensemble(sources[method])
    for name in names:
        path, complete = plans[name]
        if complete:
            print(f'SKIP completed analysis: {path}', flush=True)
            continue
        print(f'Analyzing {name}', flush=True)
        if name == '01_main_5seed':
            main_analysis(path, sources, ensembles, audit, report)
        elif name == '02_joint_construction':
            construction_analysis(path, ensembles, report)
        elif name == '03_beta_sweep':
            beta_analysis(path, sources, report)
        elif name == '04_screening_efficiency':
            screening_analysis(path, stages['02_joint_construction'], report)
        elif name == '05_risk_increment':
            risk_analysis(path, ensembles.get('Full MTL'), report)
        else:
            summary_analysis(path, stages, report)
        finish_output(path, signature)
    return {name: str(path) for name, path in stages.items()}
