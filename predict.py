"""Screen the Unknown pool using the retained five-member final model."""
from pathlib import Path
import argparse
import importlib
import json
import sys


def main():
    root = Path(__file__).resolve().parent
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / 'code'))
    from package_runtime import configure, graph_cache_status, ensure_graph_cache, weights_directory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', default='results/main/M06_full_mtl')
    parser.add_argument('--output-dir', default='results/reproduced/unknown_screening')
    parser.add_argument('--prepare-graphs', action='store_true', help='Regenerate omitted Unknown-pool graph caches before inference.')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Inspect model inputs and cache status without inference or cache generation.')
    args = parser.parse_args()
    experiment = Path(args.experiment_dir)
    output = Path(args.output_dir)
    if not experiment.is_absolute():
        experiment = root / experiment
    if not output.is_absolute():
        output = root / output
    weights = weights_directory(experiment)
    required = [experiment / 'resolved_experiment.yaml', experiment / 'ensemble/temperatures.json',
                experiment / 'ranking/ranking_configuration.json', root / 'data/processed/unknown_pool.csv',
                root / 'data/c2db.db']
    required += [weights / f'seed_{seed}/best_checkpoint.pt' for seed in range(2025, 2030)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error('Missing required inputs: ' + ', '.join(missing))
    if args.dry_run:
        print(json.dumps({'module': 'train.screen_unknown', 'experiment_dir': str(experiment),
                          'weights_dir': str(weights), 'output_dir': str(output),
                          'checked_inputs': len(required), 'graph_cache': graph_cache_status('unknown')}, indent=2))
        return
    configure()
    ensure_graph_cache('unknown', prepare=args.prepare_graphs)
    module = importlib.import_module('train.screen_unknown')
    original_discover = module.discover_member_dirs

    def discover_member_dirs(full_dir, experiment_config):
        return original_discover(weights_directory(full_dir), experiment_config)

    module.discover_member_dirs = discover_member_dirs
    sys.argv = ['train.screen_unknown', '--full-experiment-dir', str(experiment), '--output-dir', str(output)]
    if args.force:
        sys.argv.append('--force')
    module.main()


if __name__ == '__main__':
    main()
