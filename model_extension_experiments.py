"""Freeze cohort/splits and write or execute the paired extension experiment matrix."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from protein_hdf5_dataset import ProteinGraphHDF5Dataset
from train_gate import split_dataset_by_target, save_target_split_manifest

BASE_NODES = 'aa_type,chain,interface_nodes,rsasa_i,interface_node_degree'
BASE_EDGES = 'interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing'
APBS = 'apbs_pair_mean,apbs_pair_absdiff,apbs_pair_product,apbs_pair_missing'


def variants(without_apbs=False):
    result = [dict(name='seed_replicate_base', architecture='gat', apbs='none',
                   pooling='all', esm=False, objective='ae')]
    for apbs, pooling, esm in itertools.product(('none',) if without_apbs else ('none', 'plain', 'area'), ('all', 'interface'), (False, True)):
        label = 'baseline' if (apbs, pooling, esm) == ('none', 'all', False) else f'gat_{apbs}_{pooling}_esm{int(esm)}'
        result.append(dict(name=label, architecture='gat', apbs=apbs, pooling=pooling, esm=esm, objective='ae'))
    for architecture, objective in (('egnn', 'ae'), ('egnn', 'supervised'), ('gat', 'supervised')):
        for combined in (False, True):
            result.append(dict(name=f'{architecture}_{objective}_{"combined" if combined else "baseline"}',
                architecture=architecture, objective=objective, apbs='plain' if combined and not without_apbs else 'none',
                pooling='interface' if combined else 'all', esm=combined))
    return result


def prepare(args):
    without_apbs = getattr(args, 'without_apbs', False)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seed list must not contain duplicates')
    args.output.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output / 'matrix.json'
    if matrix_path.exists():
        raise FileExistsError(f'{matrix_path} already exists; use a new experiment directory')
    dataset = ProteinGraphHDF5Dataset(args.data, node_features=BASE_NODES.split(','),
        edge_features=(BASE_EDGES if without_apbs else BASE_EDGES + ',' + APBS + ',apbs_pair_area_product').split(','),
        require_target=True, skip_invalid_files=False, optional_node_features_dir=args.optional_node_features_dir,
        use_esm=True, require_pos=True)
    exclusions = {}
    if args.targets_file:
        exclusions_path = args.targets_file.parent / 'excluded_targets.json'
        if exclusions_path.exists():
            exclusions = json.loads(exclusions_path.read_text())
            if any(record.get('reason') != 'empty_source_subset' or record.get('source_entry_count') != 0
                   for record in exclusions.values()):
                raise ValueError('Invalid empty-target exclusion record')
        requested = {line.strip() for line in args.targets_file.read_text().splitlines() if line.strip()}
        if requested != {path.stem for path in dataset.paths}:
            raise ValueError('Prepared data targets differ from the requested subset; use a dedicated output directory')
    # Ensure optional CSV availability never silently changes the frozen cohort.
    expected = set()
    audits = {}
    for path in dataset.paths:
        audit_path = args.data / f'{path.stem}.audit.json'
        audit = json.loads(audit_path.read_text())
        if without_apbs and audit.get('apbs_enabled') is not False:
            raise ValueError(f'{audit_path}: prepare with --without-apbs so APBS availability does not filter the cohort')
        audits[path.stem] = {'accepted': len(audit['accepted']), 'rejected': len(audit['rejected'])}
        expected.update((path.stem, name) for name in audit['accepted'])
    actual = {(key.path.stem, key.group_name) for key in dataset.samples}
    if actual != expected:
        raise ValueError(f'Prepared cohort differs from usable training cohort: {len(expected - actual)} missing, {len(actual - expected)} extra. Check rSASA CSVs.')
    esm_dims = set()
    for graph in dataset:
        if not graph.interface_mask.any() or not all(np.isfinite(t.numpy()).all() for t in (graph.x, graph.edge_attr, graph.pos, graph.esm, graph.y)):
            raise ValueError(f'Invalid graph: {graph.graph_name}')
        esm_dims.add(graph.esm.shape[1])
    if len(esm_dims) != 1:
        raise ValueError(f'Inconsistent embedding widths: {esm_dims}')
    split_path = args.output / 'target_splits.json'
    if args.prior_split:
        old = json.loads(args.prior_split.read_text())
        paths_by_target = {p.stem: p for p in dataset.paths}
        prior_targets = [Path(p).stem for split in old['splits'].values() for p in split['paths']]
        if len(prior_targets) != len(set(prior_targets)):
            raise ValueError('Prior split contains repeated targets')
        unknown = set(prior_targets) - set(paths_by_target) - set(exclusions)
        if unknown:
            raise ValueError(f'Prior split has unavailable targets without empty-subset exclusions: {sorted(unknown)}')
        split_paths = {name: [paths_by_target[Path(p).stem] for p in old['splits'][name]['paths']
                             if Path(p).stem not in exclusions]
                       for name in ('train', 'val', 'test')}
        targets = [p.stem for paths in split_paths.values() for p in paths]
        if len(set(targets)) != len(targets) or set(targets) != set(paths_by_target):
            raise ValueError('Prior split must cover each prepared target exactly once')
        split_indices = {name: [i for i, key in enumerate(dataset.samples) if key.path in paths]
                         for name, paths in split_paths.items()}
    else:
        split_indices, split_paths = split_dataset_by_target(dataset.samples, args.test_fraction, args.val_fraction, args.seed)
    if any(not indices for indices in split_indices.values()):
        raise ValueError('Empty split')
    save_target_split_manifest(split_path, split_paths, split_indices, dataset.samples, SimpleNamespace(**{**vars(args), "data": str(args.data)}))
    common = ['--data', str(args.data.resolve()), '--optional-node-features-dir', str(args.optional_node_features_dir.resolve()),
        '--split-manifest', str(split_path.resolve()), '--strict-hdf5', '--device', args.device,
        '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--num-workers', str(args.num_workers),
        '--worker-start-method', 'spawn', '--lr', '3e-4', '--lr-schedule', 'cosine', '--warmup-steps', '1000',
        '--edge-recon-features', 'interface_edges,ca_dist', '--edge-stats-seed', '7',
        '--loss-weight-mode', 'fixed', '--checkpoint-metric', 'target_mse',
        '--validation-only-during-training', '--residual_connections']
    runs = []
    for config in variants(without_apbs):
        for seed in args.seeds:
            original_base = config['name'] == 'seed_replicate_base'
            nodes = 'aa_type,chain,interface_nodes,rsasa_i' if original_base else BASE_NODES
            edges = 'interface_edges,ca_dist' if original_base else BASE_EDGES
            standardized = '' if original_base else 'voronoi_contact_area'
            if config['apbs'] != 'none':
                edges += ',' + APBS
                standardized += ',apbs_pair_mean,apbs_pair_absdiff,apbs_pair_product'
                if config['apbs'] == 'area':
                    edges += ',apbs_pair_area_product'
                    standardized += ',apbs_pair_area_product'
            output = args.output / f"{config['name']}_seed{seed}"
            command = common + ['--architecture', config['architecture'], '--pooling', config['pooling'],
                '--node-features', nodes, '--edge-features', edges,
                '--seed', str(seed), '--output-dir', str(output.resolve())]
            if not original_base:
                command += ['--standardize-edge-features', standardized,
                            '--edge-feature-transforms', 'voronoi_contact_area=log1p']
            if config['esm']:
                command += ['--use-esm']
            weights = ('1', '1', '0.1', '1') if config['objective'] == 'ae' else ('0', '0', '0', '1')
            for flag, weight in zip(('node', 'edge-attr', 'edge-presence', 'target'), weights):
                command += [f'--{flag}-weight', weight]
            runs.append(dict(config=config, seed=seed, output=str(output.resolve()), argv=command))
    matrix_path.write_text(json.dumps({'runs': runs, 'cohort': audits, 'excluded_targets': exclusions,
        'apbs_enabled': not without_apbs,
        'primary_metric': 'target-macro DockQ MSE', 'split': str(split_path.resolve()),
        'cohort_keys': sorted([list(key) for key in actual])}, indent=2) + '\n')
    print(f'{len(runs)} runs written to {matrix_path}; {len(dataset)} graphs; cohort attrition: {audits}')


def training_command(run, cpus_per_task=None):
    """Cap loader processes to the Slurm budget, reserving one CPU for main."""
    argv = list(run['argv'])
    worker_index = argv.index('--num-workers') + 1
    requested = int(argv[worker_index])
    if requested < 0:
        raise ValueError('num-workers cannot be negative')
    if cpus_per_task is not None:
        if cpus_per_task < 1:
            raise ValueError('cpus-per-task must be positive')
        workers = min(requested, cpus_per_task - 1)
        argv[worker_index] = str(workers)
        print(f'CPU allocation: {cpus_per_task}; loader processes: {workers}; '
              'one main process; numerical-library threads set by cluster script', flush=True)
    return [sys.executable, str(Path(__file__).with_name('train_gate.py'))] + argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    make = sub.add_parser('prepare')
    make.add_argument('--data', type=Path, required=True)
    make.add_argument('--without-apbs', action='store_true', help='Create the 11-configuration non-APBS matrix')
    make.add_argument('--output', type=Path, required=True)
    make.add_argument('--optional-node-features-dir', type=Path, required=True)
    make.add_argument('--targets-file', type=Path)
    make.add_argument('--prior-split', type=Path)
    make.add_argument('--seed', type=int, default=7)
    make.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    make.add_argument('--val-fraction', type=float, default=0.15)
    make.add_argument('--test-fraction', type=float, default=0.15)
    make.add_argument('--epochs', type=int, default=50)
    make.add_argument('--batch-size', type=int, default=16)
    make.add_argument('--num-workers', type=int, default=7, help='Loader processes; default reserves one of eight CPUs for the main process')
    make.add_argument('--device', default='cuda')
    run = sub.add_parser('run')
    run.add_argument('--matrix', type=Path, required=True)
    run.add_argument('--cpus-per-task', type=int, help='Actual Slurm CPU allocation; caps loader workers')
    run.add_argument('--task', type=int, required=True, help='One-based Slurm array index')
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args)
    else:
        runs = json.loads(args.matrix.read_text())['runs']
        if not 1 <= args.task <= len(runs):
            raise ValueError('Task index outside matrix')
        subprocess.run(training_command(runs[args.task - 1], args.cpus_per_task), check=True)


if __name__ == '__main__':
    main()
