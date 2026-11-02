"""Homolog-free train/val/test split from a stratified fold of gate_run/expanded_target_splits.

Homolog families stay in training only: any validation or test target that shares a 30% sequence-identity chain cluster
(RCSB) with a training target moves to training, repeated until no validation or test target has a homolog in training.
Nothing is dropped, so all 350 targets stay in the cohort.

    python make_homolog_free_split.py --fold 4

writes gate_run/homolog_free_fold<k>/target_splits.json (paths are filled in on the cluster by the submit script) and
moved_to_train.csv (each moved target, its old split and the training targets that caused the move).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPANSION = Path('/scratch/ppi_target_expansion')
sys.path.insert(0, str(EXPANSION))
from select_targets import existing_clusters  # noqa: E402


def load_clusters() -> dict[str, set]:
    original = (EXPANSION / 'existing_146.txt').read_text().split()
    clusters = existing_clusters(original, EXPANSION / 'rcsb_existing_cache.json')
    for r in pd.read_csv(EXPANSION / 'targets_batch1.csv').itertuples():
        clusters[r.pdb] = {r.cluster_1, r.cluster_2}
    return clusters


def homolog_free(train: set, val: set, test: set, clusters: dict[str, set]):
    """Move val/test targets sharing a chain cluster with train into train, until none remain."""
    train, moved = set(train), {}
    while True:
        in_train = set().union(*(clusters[t] for t in train))
        hits = {t for t in (val | test) - set(moved) if clusters[t] & in_train}
        if not hits:
            break
        for t in hits:
            moved[t] = sorted(p for p in train if clusters[p] & clusters[t])
        train |= hits
    return train, val - set(moved), test - set(moved), moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fold', type=int, default=4)
    parser.add_argument('--splits', type=Path, default=ROOT / 'gate_run' / 'expanded_target_splits')
    args = parser.parse_args()

    base = json.loads((args.splits / f'stratified_fold{args.fold}.json').read_text())
    tr, va, te = (set(base['splits'][s]['targets']) for s in ('train', 'val', 'test'))
    clusters = load_clusters()
    if not all(clusters[t] for t in tr | va | te):
        raise SystemExit('A target has no 30% chain cluster; homology cannot be checked for it')
    train, val, test, moved = homolog_free(tr, va, te, clusters)

    in_train = set().union(*(clusters[t] for t in train))
    assert not any(clusters[t] & in_train for t in val | test), 'homolog left in val or test'
    assert not train & val and not train & test and not val & test
    assert train | val | test == tr | va | te and len(train | val | test) == 350

    out = ROOT / 'gate_run' / f'homolog_free_fold{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    payload = {'base_manifest': f'expanded_target_splits/stratified_fold{args.fold}.json', 'homology': 'RCSB 30% chain clusters',
               'splits': {s: {'target_count': len(ts), 'targets': sorted(ts)} for s, ts in (('train', train), ('val', val), ('test', test))}}
    (out / 'target_splits.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='ascii')
    pd.DataFrame([dict(target=t, moved_from='test' if t in te else 'val', homologs_in_train=' '.join(p))
                  for t, p in sorted(moved.items())]).to_csv(out / 'moved_to_train.csv', index=False)

    source = pd.read_csv(args.splits / 'target_assignments.csv', index_col=0)
    for name, ts in (('train', train), ('val', val), ('test', test)):
        d = source.loc[sorted(ts)]
        print(f'{name}: {len(ts)} targets ({(d.source == "batch1").sum()} new, {(d.source != "batch1").sum()} original), '
              f'mean difficulty {d.difficulty.mean():.3f}, bins {d.bin.value_counts().sort_index().to_dict()}')
    print(f'moved to train: {len(moved)} ({len(moved.keys() & te)} test, {len(moved.keys() & va)} val)')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
