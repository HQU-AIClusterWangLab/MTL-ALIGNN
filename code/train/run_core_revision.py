"""Audited, resumable core revision queue; never regenerates data or splits."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train.common import capture_environment, load_yaml, resolve_experiment, save_yaml
from train.core_revision import (SEEDS, BETAS, BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES,
    TASKS, beta_config, scientific_config, config_differences, require_equivalent,
    require_ids, require_labels, sha256, digest_json, write_json, stamp,
    immutable_output, finish_output)
from train.run_paper_revision import queue_lock, Tee

FIXED_PYTHON = Path('D:/ck/anaconda_env/envs/two/python.exe')
REQUIRED = ('COMPLETED.json', 'resolved_experiment.yaml', 'metrics.json',
            'val_predictions.csv', 'test_predictions.csv', 'best_checkpoint.pt',
            'environment.json', 'thresholds.json', 'summary.yaml')
CORE_METHODS = ('Single-task Joint', 'Equal MTL', 'Equal + LC', 'Full MTL')


def load_project(root):
    config = load_yaml(root / 'config.yaml')
    processed = Path(config['paths']['processed_dir'])
    if not all((processed / p).is_file() for p in ('supervised.csv', 'iid_split.csv')):
        external = Path(os.environ.get('FIRST_SCI_DATA_DIR', 'D:/ck/First sci/data'))
        choices = [external / 'processed', external]
        processed = next((p for p in choices if all((p / f).is_file() for f in ('supervised.csv', 'iid_split.csv'))), None)
        if processed is None:
            raise FileNotFoundError('No existing supervised.csv and iid_split.csv; will not generate data')
        config['paths']['processed_dir'] = str(processed)
    return config, processed


def configurations(root, project):
    def old(category, name):
        path = root / 'train/experiments' / category / f'{name}.yaml'
        cfg = load_yaml(path)
        cfg['experiment_file'] = str(path)
        return resolve_experiment(project, cfg)
    def entry(key, cfg, seeds, legacy):
        return {'method': key, 'config': cfg, 'seeds': list(seeds),
                'legacy': [root / 'results' / p for p in legacy]}
    full = old('main', 'M06_full_mtl')
    equal_lc = resolve_experiment(project, load_yaml(root / 'configs/experiments/core_revision/C01_equal_lc_iid.yaml'))
    entries = [
        entry('Single-task Joint', old('main', 'M04_single_alignn'), SEEDS,
              ['main/M04_single_alignn', 'q1/Q1_joint_only_alignn_iid_2028_2029']),
        entry('Equal MTL', old('main', 'M05_equal_mtl'), SEEDS, ['main/M05_equal_mtl']),
        entry('Equal + LC', equal_lc, SEEDS, []),
        entry('Full MTL', full, SEEDS, ['main/M06_full_mtl']),
        entry('T-only', old('q1', 'Q1_thermo_only_alignn_iid'), SEEDS, ['q1/Q1_thermo_only_alignn_iid']),
        entry('D-only', old('q1', 'Q1_dynamic_only_alignn_iid'), SEEDS, ['q1/Q1_dynamic_only_alignn_iid']),
    ]
    for beta in BETAS:
        if beta == .1:  # same audited members, never a duplicate job
            continue
        cfg = beta_config(full, beta)
        file_cfg = load_yaml(root / f'configs/experiments/core_revision/B{int(round(beta * 100)):02d}_iid.yaml')
        check = resolve_experiment(project, file_cfg)
        require_equivalent(check, cfg)
        entries.append(entry(f'beta={beta:.2f}', cfg, SEEDS[:3],
                             ['ablation/A03_no_logic'] if beta == 0 else []))
    for item in entries:
        cfg = copy.deepcopy(item['config'])
        slug = {'Single-task Joint': 'single_joint', 'Equal MTL': 'equal_mtl',
                'Equal + LC': 'equal_lc', 'Full MTL': 'full_mtl', 'T-only': 'thermo_only',
                'D-only': 'dynamic_only'}.get(item['method'], item['method'].replace('=', '_').replace('.', 'p'))
        cfg.update(name=slug, category='core_revision/training', seeds=item['seeds'])
        cfg['paths']['results_dir'] = str(root / 'results')
        item['config'] = cfg
        item['training_root'] = root / 'results/core_revision/training' / slug
    return entries


def check_data(processed):
    metadata = require_ids(pd.read_csv(processed / 'supervised.csv', dtype={'material_id': str}))
    split = require_ids(pd.read_csv(processed / 'iid_split.csv', dtype={'material_id': str}), metadata[['material_id']])
    if set(split.split) != {'train', 'val', 'test'}:
        raise ValueError('Fixed split must contain nonempty train, val, test')
    labels = metadata.rename(columns={f'{t}_label': f'y_{t}' for t in TASKS})
    require_labels(labels)
    references = {}
    for part in ('train', 'val', 'test'):
        reference = labels[labels.material_id.isin(split.loc[split.split == part, 'material_id'])]
        references[part] = require_ids(reference[['material_id', *[f'y_{t}' for t in TASKS]]])
        if any(reference[f'y_{t}'].nunique() != 2 for t in TASKS):
            raise ValueError(f'{part} must contain both classes for all tasks')
    graph_stats = []
    for text in sorted(metadata.graph_path.unique()):
        path = Path(text)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing or empty supervised graph: {path}')
        stat = path.stat()
        graph_stats.append([str(path), stat.st_size, stat.st_mtime_ns])
    signature = {'split_file': str(processed / 'iid_split.csv'),
                 'split_sha256': sha256(processed / 'iid_split.csv'),
                 'supervised_file': str(processed / 'supervised.csv'),
                 'supervised_sha256': sha256(processed / 'supervised.csv'),
                 'graph_file_count': len(graph_stats),
                 'graph_inventory_sha256': digest_json(graph_stats),
                 'graph_inventory_definition': 'path,size,mtime_ns; not graph-content hash',
                 'partitions': {p: len(f) for p, f in references.items()}}
    return references, signature


def training_code(root):
    paths = [root / 'train/train_gnn.py', root / 'train/dataset.py', root / 'train/common.py']
    paths += sorted((root / 'models').glob('*.py'))
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def runtime_signature(env):
    return {k: env.get(k) for k in ('python', 'python_executable', 'torch', 'torch_cuda', 'dgl', 'alignn', 'pandas', 'sklearn')}


def validate_member(folder, config, seed, references, data_signature, environment):
    missing = [name for name in REQUIRED if not (folder / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete COMPLETED result: {folder}: {missing}')
    old = load_yaml(folder / 'resolved_experiment.yaml')
    require_equivalent(old, config)
    if old.get('split_file') != 'iid_split.csv':
        raise ValueError(f'Only fixed iid_split.csv is eligible: {folder}')
    old_split = Path(old['paths']['processed_dir']) / old['split_file']
    if not old_split.exists() or sha256(old_split) != data_signature['split_sha256']:
        raise ValueError(f'Recorded split path cannot be verified: {folder}')
    old_metadata = Path(old['paths']['processed_dir']) / 'supervised.csv'
    if not old_metadata.exists() or sha256(old_metadata) != data_signature['supervised_sha256']:
        raise ValueError(f'Recorded supervised data path cannot be verified: {folder}')
    info = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    marker = json.loads((folder / 'COMPLETED.json').read_text(encoding='utf-8'))
    if marker.get('status') != 'completed' or int(marker['seed']) != seed or int(info['seed']) != seed:
        raise ValueError(f'Invalid completion/seed metadata: {folder}')
    if any(int(record['batch_size']) != int(config['training']['batch_size']) for record in (marker, info)):
        raise ValueError(f'Actual batch size differs from scientific configuration: {folder}')
    if info.get('device', '').split(':')[0] != 'cuda':
        raise ValueError(f'CUDA training required: {folder}')
    stored_thresholds = json.loads((folder / 'thresholds.json').read_text(encoding='utf-8'))
    for task in config['heads']:
        value = stored_thresholds[task]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid stored threshold: {folder}/{task}')
        if any(info[f'{part}_{task}']['threshold'] != value for part in ('val', 'test')):
            raise ValueError(f'Stored threshold metadata disagree: {folder}/{task}')
    old_env = json.loads((folder / 'environment.json').read_text(encoding='utf-8'))
    if runtime_signature(old_env) != runtime_signature(environment):
        raise ValueError(f'Scientific runtime mismatch: {folder}')
    frames = {}
    for part in ('val', 'test'):
        frame = require_ids(pd.read_csv(folder / f'{part}_predictions.csv', dtype={'material_id': str}), references[part])
        require_labels(frame)
        for task in config['heads']:
            p = frame[f'p_{task}'].to_numpy(float)
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError(f'Invalid probability: {folder}/{part}/{task}')
        logits = frame.filter(regex='^logit_').to_numpy(float)
        if not np.isfinite(logits).all():
            raise ValueError(f'Nonfinite logits: {folder}/{part}')
        frame['_partition'] = part
        frames[part] = frame
    if set(frames['val'].material_id) & set(frames['test'].material_id):
        raise ValueError('Validation/test overlap')
    return frames, info


def preflight(root):
    project, processed = load_project(root)
    environment = capture_environment()
    references, data_signature = check_data(processed)
    entries = configurations(root, project)
    code = training_code(root)
    rows, jobs, sources, inputs = [], [], {}, {}
    for item in entries:
        method, config = item['method'], item['config']
        sources[method] = {}
        for seed in item['seeds']:
            candidates = [p / f'seed_{seed}' for p in item['legacy']]
            fresh = item['training_root'] / f'seed_{seed}'
            candidates.append(fresh)
            contract = {'schema': 1, 'seed': seed, 'scientific_config': scientific_config(config),
                        'data': data_signature, 'code': code, 'runtime': runtime_signature(environment)}
            for folder in candidates:
                if not (folder / 'COMPLETED.json').exists():
                    continue
                old = load_yaml(folder / 'resolved_experiment.yaml')
                differences = config_differences(scientific_config(old), scientific_config(config))
                row = {'method': method, 'seed': seed, 'source_dir': str(folder),
                       'compatible': not bool(differences), 'differences': ';'.join(differences),
                       'scientific_config_sha256': digest_json(scientific_config(old)),
                       'expected_config_sha256': digest_json(scientific_config(config)),
                       'normalization': 'names/paths/seed lists administrative; omitted monitor_task=joint; beta=0 lower-bound flag inactive',
                       'old_logic_lower_bound': old.get('logic_lower_bound'),
                       'expected_logic_lower_bound': config.get('logic_lower_bound'),
                       'historical_provenance': 'legacy resolved config/runtime; historical code/data hash unavailable' if folder != fresh else 'run_contract.json'}
                if differences:
                    row['decision'] = 'rejected; new compatible job required'
                    rows.append(row)
                    if folder == fresh:
                        raise ValueError(f'Incompatible core completion: {folder}: {differences}')
                    continue
                frames, info = validate_member(folder, config, seed, references, data_signature, environment)
                if folder == fresh:
                    check_resume_contract(folder, contract)
                if seed in sources[method]:
                    row['decision'] = 'compatible duplicate; first declared source retained'
                else:
                    sources[method][seed] = {'folder': folder, 'frames': frames, 'metrics': info}
                    row['decision'] = 'reuse' if folder != fresh else 'completed new training'
                rows.append(row)
                for name in REQUIRED:
                    if name != 'best_checkpoint.pt':
                        inputs[str(folder / name)] = sha256(folder / name)
            if seed not in sources[method]:
                check_resume_contract(fresh, contract)
                jobs.append({'method': method, 'seed': seed, 'folder': fresh,
                             'config': config, 'contract': contract})
                rows.append({'method': method, 'seed': seed, 'source_dir': str(fresh),
                             'compatible': True, 'decision': 'resume' if (fresh / 'last_checkpoint.pt').exists() else 'missing',
                             'expected_config_sha256': digest_json(scientific_config(config))})
    audit = pd.DataFrame(rows)
    report = {'timestamp': stamp(), 'environment': environment, 'data': data_signature,
              'code_sha256': code, 'input_files_sha256': inputs,
              'input_result_dirs': sorted({str(v['folder']) for members in sources.values() for v in members.values()}),
              'seeds': SEEDS, 'beta_seeds': SEEDS[:3], 'betas': BETAS,
              'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': BOOTSTRAP_REPLICATES,
              'reused_members': sum(len(v) for v in sources.values()),
              'missing_jobs': [{k: str(j[k]) if k == 'folder' else j[k] for k in ('method', 'seed', 'folder')} for j in jobs],
              'legacy_limitation': 'Historical code/data content hashes were not recorded. Resolved configs, runtime, current split, IDs, labels, actual batch size are checked; historical byte-identical training code cannot be proved.',
              'policy': {'calibration': 'validation only', 'threshold': 'validation only',
                         'risk_fit': 'validation only', 'beta': 'prespecified sweep; no test selection',
                         'test': 'frozen predictions for final evaluation/bootstrap; never tuning',
                         'training': 'CUDA, fixed batch size; no OOM fallback changing scientific config'}}
    return sources, jobs, audit, report, processed


def check_resume_contract(folder, contract, create=False):
    path = folder / 'run_contract.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != contract:
            raise ValueError(f'Resume input/config/code/runtime changed: {folder}')
    elif folder.exists() and any(folder.iterdir()):
        raise ValueError(f'Untracked training output; preserve and inspect before continuing: {folder}')
    elif create:
        write_json(contract, path)


def run_tests(root, output):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([str(FIXED_PYTHON), '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'],
                            cwd=root, env=env, text=True, encoding='utf-8', errors='replace',
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.write_text(result.stdout, encoding='utf-8')
    print(result.stdout[-1600:], flush=True)
    if result.returncode:
        raise RuntimeError(f'CPU unit tests failed; see {output}')
    return {'exit_code': 0, 'log': str(output), 'cuda_visible_devices': ''}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'analyze', 'full'], default='preflight')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    if Path(sys.executable).resolve() != FIXED_PYTHON.resolve():
        raise RuntimeError(f'Use the fixed interpreter: {FIXED_PYTHON}')
    os.chdir(root)
    output = root / 'results/core_revision'
    runner = output / '_runner'
    runner.mkdir(parents=True, exist_ok=True)
    tag = stamp().replace(':', '').replace('+', '_')
    with queue_lock(runner / 'queue.lock'), (runner / f'{args.mode}_{tag}.log').open('a', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            status = {'mode': args.mode, 'status': 'preflight', 'started': stamp(), 'pid': os.getpid()}
            write_json(status, runner / 'status.json')
            try:
                sources, jobs, audit, report, processed = preflight(root)
                report_path = runner / f'preflight_{tag}.json'
                write_json(report, report_path)
                audit.to_csv(runner / f'configuration_audit_{tag}.csv', index=False)
                status.update(missing_jobs=report['missing_jobs'], preflight=str(report_path), reused_members=report['reused_members'])
                write_json(status, runner / 'status.json')
                print(f"Preflight: {report['reused_members']} reusable members, {len(jobs)} missing jobs; CUDA={report['environment']['cuda_available']}", flush=True)
                for j in jobs:
                    print(f"  {j['method']} seed={j['seed']} -> {j['folder']}", flush=True)
                if args.mode == 'full':
                    if not report['environment']['cuda_available']:
                        raise RuntimeError('CUDA required for full training')
                    from train.train_gnn import train_seed
                    for index, job in enumerate(jobs, 1):
                        status.update(status='training', job_index=index,
                                      current_job={'method': job['method'], 'seed': job['seed'], 'folder': str(job['folder'])})
                        write_json(status, runner / 'status.json')
                        check_resume_contract(job['folder'], job['contract'], create=True)
                        cfg = copy.deepcopy(job['config'])
                        cfg['training']['device'] = 'cuda:0'
                        config_file = job['folder'] / 'requested_experiment.yaml'
                        if not config_file.exists():
                            save_yaml(cfg, config_file)
                        print(f"[{index}/{len(jobs)}] {job['method']} seed={job['seed']}", flush=True)
                        # Existing trainer restores optimizer/scheduler/RNG from last_checkpoint.pt.
                        # Calling train_seed directly prevents OOM fallback from changing batch size.
                        train_seed(cfg, job['seed'], job['folder'], int(cfg['training']['batch_size']), force=False)
                        validate_member(job['folder'], cfg, job['seed'], check_data(processed)[0], report['data'], report['environment'])
                        status.pop('current_job', None)
                        status.pop('job_index', None)
                    sources, jobs, audit, report, processed = preflight(root)
                    if jobs:
                        raise RuntimeError('Training queue ended with missing qualified members')
                    status['missing_jobs'] = []
                if args.mode in ('analyze', 'full'):
                    from train.core_revision_analysis import analyze
                    status.update(status='analyzing')
                    write_json(status, runner / 'status.json')
                    status['analysis'] = analyze(root, sources, audit, report)
                    from train.analyze_literature_validation import analyze_literature
                    literature = analyze_literature(root / 'manual_validation/literature_phonon_validation_template.csv',
                        root / 'results/main/M10_unknown_screening/unknown_all_predictions.csv',
                        processed / 'unknown_pool.csv', root / 'results/main/M06_full_mtl/ensemble/thresholds.json',
                        output / '06_literature_validation')
                    status['literature'] = str(literature) if literature else 'empty template; no fabricated data'
                    status.update(status='testing')
                    write_json(status, runner / 'status.json')
                    status['unit_tests'] = run_tests(root, runner / f'cpu_tests_{tag}.log')
                status.update(status='completed' if not jobs or args.mode == 'preflight' else 'partial_missing_training',
                              finished=stamp())
                write_json(status, runner / 'status.json')
                write_json(status, runner / f'status_{tag}.json')
                print(f"Final status: {status['status']}; {runner / 'status.json'}", flush=True)
            except BaseException as exc:
                phase = status.get('status')
                status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc), finished=stamp())
                if phase == 'training' and 'current_job' in status:
                    status['failed_job'] = status['current_job']
                else:
                    status['failed_stage'] = phase
                write_json(status, runner / 'status.json')
                write_json(status, runner / f'failed_{tag}.json')
                raise


if __name__ == '__main__':
    main()
