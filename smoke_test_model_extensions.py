"""Exercise all 19 configurations on nine real graphs, with one CPU process.

SOFTWARE TEST ONLY: uses stored APBS means without certifying area weighting.
Outputs and metrics must not be used for scientific model comparisons.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from add_interface_node_degree import calculate_interface_node_degree
from model_extension_experiments import prepare, training_command
from prepare_model_extensions import (read_residues, validate_nodes, load_embeddings,
                                      apbs_features, contact_indices)
from voronoi_edge_features.common import infer_relative_pdb_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh smoke output directory')
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    print(f'Device: {args.device}; 1 main process, 0 loader workers, 1 numerical thread', flush=True)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    data = args.output / 'software_only_data'
    data.mkdir(parents=True)
    (args.output / 'SOFTWARE_TEST_ONLY.txt').write_text(__doc__ + '\n')
    for target in ('1acb', '1avx', '1ay7'):
        with h5py.File(args.bundle / 'subset_hdf5' / f'{target}.hdf5') as source, \
             h5py.File(args.bundle / 'apbs_model_data' / f'{target}_apbs_surface.hdf5') as apbs, \
             h5py.File(data / f'{target}.hdf5', 'w') as out:
            ordered = sorted(source, key=lambda name: (float(source[name]['target_scores/DockQ'][()]), name))
            names = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
            audit = {'accepted': [], 'rejected': {}, 'software_smoke_only': True,
                     'apbs_weighting': 'unverified: numerical execution only'}
            for name in names:
                graph = source[name]
                relative, _ = infer_relative_pdb_path(target, name)
                residues, pos = read_residues(args.bundle / 'pdb' / relative)
                validate_nodes(graph, residues)
                esm, provenance = load_embeddings(args.bundle / 'SS_embeds', target, name, residues,
                                                  '{target}_all.fasta', '{target}.{chain}.pt', 33)
                key = name.removesuffix('_corrected')
                if key not in apbs:
                    key += '_corrected'
                features = apbs_features(apbs[key], graph, residues)
                source.copy(name, out)
                copied = out[name]
                copied.attrs['software_smoke_only'] = True
                copied.attrs['esm_provenance'] = json.dumps(provenance)
                for key, value in {'pos': pos, 'esm2': esm}.items():
                    if key in copied:
                        del copied[key]
                    copied.create_dataset(key, data=value, compression='gzip')
                for key, value in features.items():
                    copied['edge_features'].create_dataset(key, data=value)
                degree_key = 'node_features/interface_node_degree'
                if degree_key in copied:
                    del copied[degree_key]
                copied[degree_key] = calculate_interface_node_degree(
                    graph['node_features/interface_nodes'][()], contact_indices(graph, len(residues)))
                audit['accepted'].append(name)
            out.attrs['software_smoke_only'] = True
        (data / f'{target}.audit.json').write_text(json.dumps(audit, indent=2))
    runs = args.output / 'runs'
    prepare(SimpleNamespace(data=data, output=runs, seeds=[7], seed=7, targets_file=None,
        prior_split=None, optional_node_features_dir=args.bundle / 'rsasa_i',
        test_fraction=.15, val_fraction=.15, epochs=2, batch_size=2, num_workers=0, device=args.device))
    matrix = json.loads((runs / 'matrix.json').read_text())
    results = []
    for i, run in enumerate(matrix['runs'], 1):
        name = run['config']['name']
        log = args.output / f'{name}.log'
        print(f'[{i}/19] {name}', flush=True)
        with log.open('w') as f:
            result = subprocess.run(training_command(run), stdout=f, stderr=subprocess.STDOUT)
        entry = {'name': name, 'returncode': result.returncode, 'log': str(log)}
        if result.returncode == 0:
            with (Path(run['output']) / 'test_predictions.csv').open() as f:
                rows = list(csv.DictReader(f))
            entry['finite_predictions'] = len(rows) == 3 and all(
                np.isfinite(float(r['predicted_target'])) for r in rows)
            with (Path(run['output']) / 'loss_history.csv').open() as f:
                history = list(csv.DictReader(f))
            entry['finite_losses'] = len(history) == 2 and all(
                np.isfinite(float(r[k])) for r in history for k in ('train_loss', 'val_loss'))
        results.append(entry)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    if any(r['returncode'] or not r.get('finite_predictions') or not r.get('finite_losses') for r in results):
        raise SystemExit('Smoke failures; see results.json and per-configuration logs')
    for reference in ('baseline', 'seed_replicate_base'):
        subprocess.run([sys.executable, 'compare_model_extensions.py', '--matrix', str(runs / 'matrix.json'),
            '--reference', reference, '--output', str(args.output / f'comparison_{reference}'),
            '--bootstrap', '100'], check=True)
    print(f'All 19 {args.device} software smoke runs passed; these metrics are not scientific comparisons.', flush=True)


if __name__ == '__main__':
    main()
