"""Path and regenerable-cache support for the compact reproducibility package."""
from pathlib import Path
import csv
import importlib
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def relocate(value):
    if isinstance(value, dict):
        return {key: relocate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if not isinstance(value, str):
        return value
    normalized = value.replace('\\', '/')
    for prefix in ('D:/ck/MTL-ALIGNN', 'D:/ck/two'):
        if normalized.casefold() == prefix.casefold():
            return ROOT.as_posix()
        if normalized.casefold().startswith(prefix.casefold() + '/'):
            relative = normalized[len(prefix) + 1:]
            break
    else:
        return value
    first = relative.split('/')[0]
    if first in {'train', 'scripts', 'src', 'tests'}:
        return (ROOT / 'code' / relative).as_posix()
    if first == 'config.yaml':
        return (ROOT / 'configs' / 'config.yaml').as_posix()
    if first == 'pretrained':
        return (ROOT / 'models' / relative).as_posix()
    return (ROOT / relative).as_posix()


def weights_directory(experiment):
    experiment = Path(experiment).resolve()
    try:
        relative = experiment.relative_to((ROOT / 'results').resolve())
    except ValueError:
        return experiment
    candidate = ROOT / 'models/experiments' / relative
    return candidate if candidate.is_dir() else experiment


def graph_cache_status(subset):
    name = {'supervised': 'supervised.csv', 'unknown': 'unknown_pool.csv'}[subset]
    total = missing = 0
    with (ROOT / 'data/processed' / name).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            total += 1
            if not Path(relocate(row['graph_path'])).is_file():
                missing += 1
    return {'subset': subset, 'rows': total, 'missing_graphs': missing,
            'regeneration_required': bool(missing)}


def ensure_graph_cache(subset, prepare=False):
    status = graph_cache_status(subset)
    if not status['missing_graphs']:
        return
    if not prepare:
        raise SystemExit(
            f"Graph caches are omitted from this package ({status['missing_graphs']} missing). "
            "Run this command again with --prepare-graphs to regenerate them from data/c2db.db. "
            "Regeneration writes local cache files; it does not retrain the model."
        )
    module = importlib.import_module('data.build_graphs')
    previous = sys.argv
    try:
        sys.argv = ['data.build_graphs', '--subset', subset, '--verify-existing']
        module.main()
    finally:
        sys.argv = previous
    if graph_cache_status(subset)['missing_graphs']:
        raise RuntimeError('Graph regeneration left missing cache files.')


def configure():
    """Redirect saved configuration and graph paths in memory."""
    code = ROOT / 'code'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    os.chdir(code)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    import pandas as pd
    from train import common
    from data import utils
    original_load_yaml = common.load_yaml
    original_load_config = utils.load_config
    original_read_csv = pd.read_csv

    def load_yaml(path):
        return relocate(original_load_yaml(path))

    def load_config(path='config.yaml'):
        return relocate(original_load_config(path))

    def read_csv(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        if hasattr(frame, 'columns') and 'graph_path' in frame.columns:
            frame['graph_path'] = frame['graph_path'].map(relocate)
        return frame

    common.load_yaml = load_yaml
    utils.load_config = load_config
    pd.read_csv = read_csv
