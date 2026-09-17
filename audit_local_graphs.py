"""Serial, read-only audit of every graph in a local HDF5 directory.

Checks training-facing shapes, finiteness, indexing, labels and feature coverage.
This does not verify the physical correctness of Voronoi/APBS computations.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from add_interface_node_degree import normalize_contacts


def inspect_graph(g):
    issues = []
    result = dict(base_graph_ok=False, voronoi_ready=False, mask_ready=False,
                  interface_degree_present=False, interface_pool_ready=False)
    try:
        n = len(g['node_features/aa_type'])
        if not n:
            raise ValueError('zero nodes')
        for name, width in [('aa_type', 20), ('chain', 1), ('interface_nodes', 1)]:
            v = g[f'node_features/{name}'][()]
            if width == 1 and v.ndim == 1:
                v = v[:, None]
            if v.shape != (n, width) or not np.isfinite(v).all():
                raise ValueError(f'invalid node feature {name}')
        mask = g['node_features/interface_nodes'][()].reshape(-1)
        if not np.isin(mask, [0, 1]).all():
            raise ValueError('nonbinary interface_nodes')
        result['interface_pool_ready'] = bool(mask.any())
        ref = g['node_reference'][()]
        if ref.shape != (n, 3) or not np.array_equal(ref[:, 0].astype(int), np.arange(n)):
            raise ValueError('invalid node_reference indices/shape')
        edges = normalize_contacts(g['edge_features/contacts'][()], n)
        for name in ('interface_edges', 'ca_dist'):
            v = g[f'edge_features/{name}'][()]
            if v.ndim == 1:
                v = v[:, None]
            if v.shape != (len(edges), 1) or not np.isfinite(v).all():
                raise ValueError(f'invalid edge feature {name}')
            if name == 'ca_dist' and (v < 0).any():
                raise ValueError('negative ca_dist')
            if name == 'interface_edges' and not np.isin(v, [0, 1]).all():
                raise ValueError('nonbinary interface_edges')
        score = np.asarray(g['target_scores/DockQ'][()])
        if score.size != 1 or not np.isfinite(score).all() or not (0 <= float(score.item()) <= 1):
            raise ValueError('invalid DockQ')
        result['base_graph_ok'] = True
        for name, key in [('voronoi_contact_area', 'voronoi_ready'), ('voronoi_contact_missing', 'mask_ready')]:
            if name not in g['edge_features']:
                issues.append(f'missing {name}')
                continue
            v = g['edge_features'][name][()]
            if v.ndim == 1:
                v = v[:, None]
            valid = v.shape == (len(edges), 1) and np.isfinite(v).all() and not (v < 0).any()
            if name.endswith('missing'):
                valid = valid and np.isin(v, [0, 1]).all()
            result[key] = bool(valid)
            if not valid:
                issues.append(f'invalid {name}')
        result['interface_degree_present'] = 'interface_node_degree' in g['node_features']
        if not result['interface_pool_ready']:
            issues.append('no interface nodes')
    except (KeyError, ValueError, TypeError, OSError) as exc:
        issues.append(str(exc))
    result['issues'] = '; '.join(issues)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/scratch/ppi_autoencoder_code/processed_graph_data'))
    p.add_argument('--output', type=Path, default=Path('local_data_audit'))
    args = p.parse_args()
    paths = sorted(set(args.data.glob('*.hdf5')) | set(args.data.glob('*.h5')))
    if not paths:
        raise ValueError('No graph files')
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, total = [], Counter()
    fields = ['target', 'graph', *inspect_graph({}).keys()]
    with (args.output / 'graphs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for i, path in enumerate(paths, 1):
            counts = Counter()
            error = ''
            try:
                with h5py.File(path, 'r') as handle:
                    for name in handle:
                        result = inspect_graph(handle[name])
                        writer.writerow(dict(target=path.stem, graph=name, **result))
                        counts['graphs'] += 1
                        for key, value in result.items():
                            if key != 'issues':
                                counts[key] += int(value)
            except OSError as exc:
                error = str(exc)
            total.update(counts)
            summaries.append(dict(target=path.stem, **dict(counts), file_error=error))
            stream.flush()
            print(f'[{i}/{len(paths)}] {path.stem}: {dict(counts)} {error}', flush=True)
    summary = {'data': str(args.data), 'files': len(paths), 'totals': dict(total), 'targets': summaries}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'files': len(paths), **dict(total)}), flush=True)


if __name__ == '__main__':
    main()
