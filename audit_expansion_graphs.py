"""Model lists for the target-expansion graphs: the models of one target usable by the rank model.

A model is accepted if it passes full_pooling_validation.validate_graph (the audit behind
gate_run/full_pooling_repaired/model_lists: finite node/edge features of the right length, non-empty interface,
DockQ in [0, 1], interface_node_degree consistent with the contacts) and also has a finite rsasa_i_node per node and
a finite v_es per edge, the two rank-model inputs that audit predates. The expansion graphs carry rSASA as the
per-node rsasa_i_node, so validate_graph's per-model rSASA CSV check is not used.

    python audit_expansion_graphs.py --graph /path/to/graphs/1brb.hdf5 --output-dir gate_run/dataset_size_comparison
writes <output-dir>/model_lists/<pdb>.txt (accepted names) and <output-dir>/audit/<pdb>.json (counts, rejections).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from full_pooling_validation import validate_graph, write_json


def check_rank_inputs(g) -> None:
    n = len(g['node_features/interface_nodes'])
    rsasa = np.asarray(g['node_features/rsasa_i_node'][()])
    if len(rsasa) != n or not np.isfinite(rsasa).all():
        raise ValueError('Invalid node feature rsasa_i_node')
    e = len(g['edge_features/contacts'])
    v_es = np.asarray(g['edge_features/v_es'][()])
    if len(v_es) != e or not np.isfinite(v_es).all():
        raise ValueError('Invalid edge feature v_es')


def audit(graph: Path) -> dict:
    accepted, rejected = [], {}
    with h5py.File(graph, 'r') as f:
        for name in sorted(f):
            try:
                validate_graph(f[name], 0.0)  # rSASA is checked per node below, not from a CSV
                check_rank_inputs(f[name])
                accepted.append(name)
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                rejected[name] = str(exc).strip('"')
    return dict(target=graph.stem, graph=str(graph.resolve()), raw_models=len(accepted) + len(rejected),
                accepted=accepted, rejected=rejected, rejection_counts=dict(Counter(rejected.values())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.graph)
    lists = args.output_dir / 'model_lists'
    lists.mkdir(parents=True, exist_ok=True)
    (lists / f'{args.graph.stem}.txt').write_text(''.join(n + '\n' for n in result['accepted']))
    write_json(args.output_dir / 'audit' / f'{args.graph.stem}.json', result)
    print(f'{args.graph.stem}: {len(result["accepted"])}/{result["raw_models"]} accepted; '
          f'rejections {result["rejection_counts"] or "none"}', flush=True)


if __name__ == '__main__':
    main()
