"""Read-only, serial alignment audit of all graphs in a downloaded bundle.

This checks numerical compatibility, not the provenance of APBS area weighting.
Outputs belong in a separate audit directory; no input is modified.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from audit_local_graphs import inspect_graph
from prepare_model_extensions import (read_residues, validate_nodes, load_embeddings,
                                      apbs_features, read_embedding, read_fasta)
from voronoi_edge_features.common import infer_relative_pdb_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bundle / 'bundle_manifest.json').read_text())
    totals = Counter()
    reports = []
    for i, (target, names) in enumerate(manifest['targets'].items(), 1):
        counts, failures, provenance = Counter(), {}, Counter()
        with (args.bundle / 'rsasa_i' / f'{target}_avg_rSASA_i.csv').open() as f:
            rsasa = {}
            for row in csv.DictReader(f):
                key = row['Decoy'].strip()
                if not key or not row['avg_rsasa_i'].strip():
                    continue
                if key in rsasa:
                    raise ValueError(f'{target}: duplicate rSASA entry {key}')
                rsasa[key] = float(row['avg_rsasa_i'])
        with h5py.File(args.bundle / 'subset_hdf5' / f'{target}.hdf5') as graphs, \
             h5py.File(args.bundle / 'apbs_model_data' / f'{target}_apbs_surface.hdf5') as apbs:
            if set(graphs) != set(names):
                raise ValueError(f'{target}: graph keys differ from manifest')
            for name in names:
                counts['graphs'] += 1
                try:
                    graph = graphs[name]
                    basic = inspect_graph(graph)
                    if not all(basic[k] for k in ('base_graph_ok', 'voronoi_ready', 'mask_ready', 'interface_pool_ready')):
                        raise ValueError(f'Graph features: {basic}')
                    if name not in rsasa or not np.isfinite(rsasa[name]):
                        raise ValueError('Missing/nonfinite rSASA')
                    path, _ = infer_relative_pdb_path(target, name)
                    residues, pos = read_residues(args.bundle / 'pdb' / path)
                    validate_nodes(graph, residues)
                    if not np.isfinite(pos).all():
                        raise ValueError('Nonfinite coordinates')
                    esm, _ = load_embeddings(args.bundle / 'SS_embeds', target, name, residues,
                                              '{target}_all.fasta', '{target}.{chain}.pt', 33)
                    if esm.shape != (len(residues), 1280):
                        raise ValueError(f'Unexpected ESM shape {esm.shape}')
                    key = name.removesuffix('_corrected')
                    if key not in apbs:
                        key += '_corrected'
                    group = apbs[key]
                    if group.attrs.get('potential_units', apbs.attrs.get('potential_units')) != 'kT/e':
                        raise ValueError('Unexpected APBS units')
                    features = apbs_features(group, graph, residues)
                    weighted = bool(group.attrs.get('residue_statistics_area_weighted',
                                                    apbs.attrs.get('residue_statistics_area_weighted', False)))
                    provenance['recorded' if weighted else 'unverified'] += 1
                    counts['apbs_missing_edges'] += int(features['apbs_pair_missing'].sum())
                    counts['edges'] += len(features['apbs_pair_missing'])
                    counts['compatible'] += 1
                except (KeyError, ValueError, OSError, RuntimeError) as exc:
                    failures[name] = str(exc)
                    counts['failed'] += 1
        report = {'target': target, 'counts': dict(counts), 'apbs_weighting': dict(provenance), 'failures': failures}
        (args.output / f'{target}.json').write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
        totals.update(counts)
        read_embedding.cache_clear()
        read_fasta.cache_clear()
        print(f'[{i}/{len(manifest["targets"])}] {target}: {dict(counts)}', flush=True)
    summary = {'bundle': str(args.bundle), 'totals': dict(totals), 'targets': reports,
               'scope': 'Numerical/alignment checks; unmarked APBS area weighting remains unverified'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(totals)), flush=True)
    if totals['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
