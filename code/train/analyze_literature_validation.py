"""Join manually verified literature to the frozen 8803-member unknown pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train.core_revision import require_ids, sha256, digest_json, write_json, immutable_output, finish_output

FIELDS = ['material_id', 'formula', 'model_rank', 'p_thermo', 'p_dynamic', 'p_joint',
          'u_dynamic', 'u_joint', 'source_title', 'doi', 'year', 'structure_match_status',
          'structure_match_basis', 'literature_dynamic_conclusion', 'notes']
MATCH_STATUSES = {'matched', 'partial', 'uncertain', 'excluded'}
CONCLUSIONS = {'stable', 'unstable', 'ambiguous'}


def analyze_literature(template, predictions, pool, thresholds, output):
    template = Path(template)
    if not template.exists():
        raise FileNotFoundError(f'Manual template not found: {template}')
    try:
        literature = pd.read_csv(template, dtype=str).fillna('')
    except pd.errors.EmptyDataError:
        literature = pd.DataFrame(columns=FIELDS)
    if literature.empty:
        print('文献 CSV 为空：未进行分析，未生成或补造任何文献数据。')
        return None
    missing = set(FIELDS) - set(literature)
    if missing:
        raise ValueError(f'Missing literature columns: {sorted(missing)}')
    if (literature.material_id.str.strip() == '').any():
        raise ValueError('Literature material_id must not be empty')
    if not set(literature.structure_match_status) <= MATCH_STATUSES:
        raise ValueError('Invalid structure_match_status')
    if not set(literature.literature_dynamic_conclusion) <= CONCLUSIONS:
        raise ValueError('Invalid literature_dynamic_conclusion')
    if (literature.structure_match_basis.str.strip() == '').any():
        raise ValueError('Every inclusion/exclusion requires structure_match_basis')
    if ((literature.source_title.str.strip() == '') & (literature.doi.str.strip() == '')).any():
        raise ValueError('Each record needs a real source title or DOI; never auto-filled')
    pool_frame = require_ids(pd.read_csv(pool, dtype={'material_id': str}))
    if len(pool_frame) != 8803:
        raise ValueError(f'Expected the frozen 8803 unknown candidate pool, got {len(pool_frame)}')
    prediction_frame = require_ids(pd.read_csv(predictions, dtype={'material_id': str}), pool_frame[['material_id']])
    outside = set(literature.material_id) - set(pool_frame.material_id)
    if outside:
        raise ValueError(f'Literature materials outside the unknown pool: {sorted(outside)}')
    thresholds_dict = json.loads(Path(thresholds).read_text(encoding='utf-8'))
    dynamic_threshold = float(thresholds_dict['dynamic'])
    if not 0 <= dynamic_threshold <= 1:
        raise ValueError('Invalid frozen dynamic threshold')
    signature = digest_json({str(p): sha256(p) for p in map(Path, [template, predictions, pool, thresholds, __file__])})
    output, complete = immutable_output(output, signature)
    if complete:
        print(f'SKIP completed literature analysis: {output}')
        return output
    selected = prediction_frame[['material_id', 'formula', 'ehull']].copy()
    for t in ('thermo', 'dynamic', 'joint'):
        selected[f'p_{t}'] = prediction_frame[f'p_{t}_calibrated']
    for t in ('dynamic', 'joint'):
        selected[f'u_{t}'] = prediction_frame[f'p_{t}_std']
    selected['model_rank'] = selected.p_joint.rank(method='first', ascending=False).astype(int)
    # Preserve any manually entered prediction/formula snapshot under explicit names.
    overlapping = set(selected) & set(literature) - {'material_id'}
    literature = literature.rename(columns={k: f'manual_{k}' for k in overlapping})
    joined = literature.merge(selected, on='material_id', how='left', validate='many_to_one')
    joined['predicted_dynamic_stable'] = (joined.p_dynamic >= dynamic_threshold).astype(int)
    joined['eligible_for_dynamic_consistency'] = ((joined.structure_match_status == 'matched') &
                                                 joined.literature_dynamic_conclusion.isin(['stable', 'unstable']))
    consistent = joined.predicted_dynamic_stable == (joined.literature_dynamic_conclusion == 'stable').astype(int)
    joined['dynamic_consistent'] = np.where(joined.eligible_for_dynamic_consistency, consistent.astype(float), np.nan)
    joined['ehull_meets_paper_condition'] = joined.ehull <= .10
    joined['joint_validation_status'] = 'not established by phonon literature alone'
    joined.to_csv(output / 'literature_validation_joined.csv', index=False)
    summaries = []
    for (status, conclusion), group in joined.groupby(['structure_match_status', 'literature_dynamic_conclusion'], sort=True):
        summaries.append({'structure_match_status': status, 'literature_dynamic_conclusion': conclusion,
                          'n_records': len(group), 'n_materials': group.material_id.nunique(),
                          'n_assessed_records': int(group.eligible_for_dynamic_consistency.sum()),
                          'n_consistent_records': int(group.dynamic_consistent.fillna(0).sum()),
                          'dynamic_consistency_rate_record_level': group.dynamic_consistent.mean(),
                          'frozen_dynamic_threshold': dynamic_threshold})
    pd.DataFrame(summaries).to_csv(output / 'dynamic_consistency_summary.csv', index=False)
    candidates = []
    for material, group in joined.groupby('material_id', sort=True):
        matched = group[group.structure_match_status == 'matched']
        decisive = set(matched.literature_dynamic_conclusion) & {'stable', 'unstable'}
        candidates.append({'material_id': material, 'formula': group.formula.iloc[0],
                           'model_rank': group.model_rank.iloc[0], 'p_dynamic': group.p_dynamic.iloc[0],
                           'p_joint': group.p_joint.iloc[0], 'ehull': group.ehull.iloc[0],
                           'n_records': len(group), 'n_matched': len(matched),
                           'n_excluded': int((group.structure_match_status == 'excluded').sum()),
                           'conflicting_matched_conclusions': len(decisive) > 1,
                           'candidate_dynamic_conclusion': next(iter(decisive)) if len(decisive) == 1 else 'ambiguous',
                           'joint_validation_status': 'not established by phonon literature alone'})
    pd.DataFrame(candidates).to_csv(output / 'candidate_level_validation_table.csv', index=False)
    write_json({'input_files': list(map(str, [template, predictions, pool, thresholds])),
                'pool_size': 8803, 'ranking': 'all unknown materials, calibrated pJ descending; material_id tie-break',
                'uncertainty_scale': 'existing raw member std, ddof=1; separate from calibrated probability',
                'dynamic_threshold_source': str(thresholds), 'dynamic_threshold': dynamic_threshold,
                'literature_source': 'manual only; no auto-generated citations or conclusions'}, output / 'manifest.json')
    finish_output(output, signature)
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, default=root / 'manual_validation/literature_phonon_validation_template.csv')
    parser.add_argument('--predictions', type=Path, default=root / 'results/main/M10_unknown_screening/unknown_all_predictions.csv')
    parser.add_argument('--pool', type=Path, default=root / 'data/processed/unknown_pool.csv')
    parser.add_argument('--thresholds', type=Path, default=root / 'results/main/M06_full_mtl/ensemble/thresholds.json')
    parser.add_argument('--output', type=Path, default=root / 'results/core_revision/06_literature_validation')
    args = parser.parse_args()
    analyze_literature(args.template, args.predictions, args.pool, args.thresholds, args.output)


if __name__ == '__main__':
    main()
