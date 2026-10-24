"""Write 5 difficulty-stratified and 5 random train/val/test target splits for the
expanded 350-target cohort (146 original + 204 target-expansion batch 1).

* Difficulty = Fig. 4-style average Spearman rho of the raw ZRank2, PyDock and Rosetta
  scores with DockQ on each target's sampled set (lower = easier).
* The 45 batch-1 targets flagged training_only (they share a 30% chain cluster with an
  original target) are in train in every fold.
* The remaining targets are grouped by shared 30% chain clusters (connected components),
  and whole groups are assigned, so no val/test target has a homolog in another split.
* Each scheme is a 5-fold partition: every eligible target is in test exactly once.
  Validation (~10% of all targets) is drawn from the non-test eligible targets.

Manifests use the train_gate.py --split-manifest format; paths are <data-dir>/<pdb>.hdf5.
Sample counts are not filled in (train_gate.py only reads the paths).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXPANSION = Path('/scratch/ppi_target_expansion')
sys.path.insert(0, str(EXPANSION))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'gate_run' / 'ch3_comparison'))
from raw_score_parsers import PARSERS, parse_dockq  # noqa: E402
from select_targets import existing_clusters  # noqa: E402

N_FOLDS = 5
N_BINS = 5


def original_rho(cohort: set[str]) -> pd.DataFrame:
    """Same procedure as build_scoring_progress_notebook.py (Fig. 7a cell)."""
    rows = []
    for f in sorted(Path('/scratch/ppi_scoring_data/all_score_csvs').glob('*_sampled_scores.csv')):
        p = f.name[:4]
        if p not in cohort:
            continue
        d = pd.read_csv(f)
        rows.append({'target': p, **{m: spearmanr(d.DockQ, d[col]).statistic
                                     for m, col in (('ZRank2', 'ZRank'), ('PyDock', 'PyDock'), ('Rosetta', 'Rosetta'))}})
    bm55 = Path('/scratch/ppi_scoring_data/bm_55_scoring')
    files = {'ZRank2': 'zrank_files_sampled_{p}.txt.zr.out', 'PyDock': 'pydock_scores_{p}_sampled.txt',
             'Rosetta': 'rosetta_scores_sampled_{p}.txt'}
    for d in sorted(bm55.glob('scores_sampled_*')):
        p = d.name.split('_')[-1]
        dq = parse_dockq(d / f'dockq_scores_{p}_sampled.txt')
        r = {'target': p}
        for m, f in files.items():
            s = PARSERS[m](d / f.format(p=p))
            keys = [k for k in dq if k in s]
            r[m] = spearmanr([dq[k] for k in keys], [s[k] for k in keys]).statistic
        rows.append(r)
    return pd.DataFrame(rows)


def expansion_rho() -> pd.DataFrame:
    rows = []
    for f in sorted((EXPANSION / 'results' / 'scores').glob('*_sampled_scores.csv')):
        d = pd.read_csv(f, index_col=0)
        d = d[d.set == 'sampled']
        rows.append({'target': f.name[:4], **{m: spearmanr(d.DockQ, d[col]).statistic
                                              for m, col in (('ZRank2', 'ZRank'), ('PyDock', 'PyDock'),
                                                             ('Rosetta', 'Rosetta'))}})
    return pd.DataFrame(rows)


def homology_groups(original: list[str], batch: pd.DataFrame) -> dict[str, int]:
    """Groups over the original targets and the non-training_only batch-1 targets."""
    clusters = existing_clusters(original, EXPANSION / 'rcsb_existing_cache.json')
    for r in batch[~batch.training_only].itertuples():
        clusters[r.pdb] = {r.cluster_1, r.cluster_2}
    graph = nx.Graph()
    graph.add_nodes_from(clusters)
    members: dict[str, list[str]] = {}
    for pdb, cs in clusters.items():
        for c in cs:
            members.setdefault(c, []).append(pdb)
    for pdbs in members.values():
        nx.add_path(graph, pdbs)
    comps = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])
    return {pdb: i for i, comp in enumerate(comps) for pdb in comp}


def random_partition(groups: pd.Series, n: int, rng: random.Random) -> list[list[str]]:
    """Shuffle groups, then place each (largest first) into the currently smallest part."""
    by_group = {g: list(ts) for g, ts in groups.groupby(groups).groups.items()}
    order = list(by_group)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(by_group[g]))  # stable: random order within equal sizes
    parts: list[list[str]] = [[] for _ in range(n)]
    for g in order:
        min(parts, key=len).extend(by_group[g])
    return parts


def stratified_partition(targets: pd.DataFrame, n: int, rng: random.Random) -> list[list[str]]:
    """Groups in random order (largest first), each into the part that keeps per-bin and total
    counts most even (sum of squares of the counts after adding it)."""
    by_group = {g: list(ts) for g, ts in targets.groupby('group').groups.items()}
    order = list(by_group)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(by_group[g]))
    parts: list[list[str]] = [[] for _ in range(n)]
    counts = np.zeros((n, N_BINS + 1))  # per-bin counts, then total
    for g in order:
        add = np.zeros(N_BINS + 1)
        np.add.at(add, targets.bin[by_group[g]].to_numpy(), 1)
        add[-1] = len(by_group[g])
        cost = (((counts + add) ** 2) - counts ** 2).sum(1)
        best = rng.choice(np.flatnonzero(cost == cost.min()))
        parts[best].extend(by_group[g])
        counts[best] += add
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'gate_run' / 'expanded_target_splits')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    original = (EXPANSION / 'existing_146.txt').read_text().split()
    batch = pd.read_csv(EXPANSION / 'targets_batch1.csv')

    rho = pd.concat([original_rho(set(original)), expansion_rho()], ignore_index=True).set_index('target')
    assert len(rho) == 350 and set(rho.index) == set(original) | set(batch.pdb), 'not the 350-target cohort'
    rho['difficulty'] = rho[['ZRank2', 'PyDock', 'Rosetta']].mean(1)
    rho['source'] = np.where(rho.index.isin(original), 'original_146', 'batch1')
    rho['training_only'] = rho.index.isin(batch.pdb[batch.training_only])
    rho['group'] = pd.Series(homology_groups(original, batch))
    rho['bin'] = pd.qcut(rho.difficulty, N_BINS, labels=False)  # quintiles over all 350
    assert rho.difficulty.notna().all() and rho.group.notna().eq(~rho.training_only).all()

    eligible = rho[~rho.training_only]
    fixed_train = sorted(rho.index[rho.training_only])
    n_val = int(round(len(rho) * args.val_fraction))

    args.out.mkdir(parents=True, exist_ok=True)
    assignment = rho[['source', 'training_only', 'group', 'difficulty', 'bin', 'ZRank2', 'PyDock', 'Rosetta']].copy()
    summary = []
    for scheme in ('stratified', 'random'):
        rng = random.Random(f'{scheme}-{args.seed}')
        if scheme == 'stratified':
            test_folds = stratified_partition(eligible, N_FOLDS, rng)
        else:
            test_folds = random_partition(eligible.group, N_FOLDS, rng)
        for k, test in enumerate(test_folds):
            rest = eligible.drop(test)
            n_parts = max(2, int(round(len(rest) / n_val)))
            if scheme == 'stratified':
                val = stratified_partition(rest, n_parts, rng)[0]
            else:
                val = random_partition(rest.group, n_parts, rng)[0]
            train = sorted(set(rest.index) - set(val)) + fixed_train
            splits = {'train': sorted(train), 'val': sorted(val), 'test': sorted(test)}

            assert not (set(splits['train']) & set(splits['val'])) and not (set(splits['train']) & set(splits['test']))
            assert not (set(splits['val']) & set(splits['test']))
            assert set().union(*splits.values()) == set(rho.index)
            g = {s: set(rho.group[ts].dropna()) for s, ts in splits.items()}
            assert not (g['train'] & g['val']) and not (g['train'] & g['test']) and not (g['val'] & g['test'])

            name = f'{scheme}_fold{k}'
            payload = {
                'data_path': args.data_dir,
                'scheme': scheme,
                'fold': k,
                'seed': args.seed,
                'difficulty': 'mean Spearman rho(raw ZRank2/PyDock/Rosetta score, DockQ), sampled set',
                'training_only_targets': fixed_train,
                'splits': {s: {'target_count': len(ts), 'targets': ts,
                               'paths': [f'{args.data_dir.rstrip("/")}/{t}.hdf5' for t in ts]}
                           for s, ts in splits.items()},
            }
            (args.out / f'{name}.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='ascii')
            assignment[name] = pd.Series({t: s for s, ts in splits.items() for t in ts})
            for s, ts in splits.items():
                d = rho.loc[ts]
                summary.append({'manifest': name, 'split': s, 'targets': len(ts),
                                'batch1': int((d.source == 'batch1').sum()),
                                'difficulty_mean': d.difficulty.mean(), 'difficulty_std': d.difficulty.std(),
                                **{f'bin{b}': int((d.bin == b).sum()) for b in range(N_BINS)}})

    assignment.sort_index().to_csv(args.out / 'target_assignments.csv')
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out / 'split_summary.csv', index=False)
    print(summary.round(3).to_string(index=False))
    print(f'wrote {2 * N_FOLDS} manifests to {args.out}')


if __name__ == '__main__':
    main()
