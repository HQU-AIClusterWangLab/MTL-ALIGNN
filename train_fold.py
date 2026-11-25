"""Train an existing experiment using its fixed split and independent seeds."""
from pathlib import Path
import argparse
import json
import runpy
import sys


def main():
    root = Path(__file__).resolve().parent
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / 'code'))
    from package_runtime import configure, graph_cache_status, ensure_graph_cache
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='configs/experiments/main/M06_full_mtl.yaml')
    parser.add_argument('--seed', type=int, help='An independent random seed, not a cross-validation fold.')
    parser.add_argument('--backend', choices=['gnn', 'tabular'], default='gnn')
    parser.add_argument('--prepare-graphs', action='store_true', help='Regenerate omitted supervised graph caches before GNN training.')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    experiment = Path(args.experiment)
    if not experiment.is_absolute():
        experiment = root / experiment
    if not experiment.is_file():
        parser.error(f'Experiment configuration does not exist: {experiment}')
    module = 'train.train_' + args.backend
    if args.dry_run:
        print(json.dumps({'module': module, 'experiment': str(experiment), 'seed': args.seed,
                          'project_config': str(root / 'configs/config.yaml'),
                          'output_root': str(root / 'results/reproduced'),
                          'graph_cache': graph_cache_status('supervised') if args.backend == 'gnn' else None}, indent=2))
        return
    configure()
    if args.backend == 'gnn':
        ensure_graph_cache('supervised', prepare=args.prepare_graphs)
    sys.argv = [module, '--experiment', str(experiment)]
    if args.seed is not None:
        sys.argv += ['--seed', str(args.seed)]
    if args.force:
        sys.argv.append('--force')
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
