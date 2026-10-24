"""Paired folds that measure how much one-chain homologs in train inflate test performance.

Built from gate_run/expanded_target_splits/stratified_fold{k}.json. In every fold, val and
test are unchanged and the two arms differ only in which training_only targets (batch-1
targets sharing one 30% chain cluster with an original target) are removed from train:

* no_homologs:     drop every training_only target linked to a test target of this fold.
* matched_control: drop the same number of training_only targets NOT linked to any test
                   target (random, seeded), so train size is identical.

Across the 5 folds every linked original target is tested once. The effect of leakage is
the per-target difference (matched_control - no_homologs) on linked test targets, against
the same difference on unlinked test targets (which should be ~0).
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BATCH = Path('/scratch/ppi_target_expansion/targets_batch1.csv')
N_FOLDS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--splits', type=Path, default=ROOT / 'gate_run' / 'expanded_target_splits')
    parser.add_argument('--out', type=Path, default=ROOT / 'gate_run' / 'homolog_leakage_splits')
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    batch = pd.read_csv(BATCH)
    links = {r.pdb: set(r.shares_one_chain_with.split(','))
             for r in batch[batch.training_only].itertuples()}
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    design = []
    for k in range(N_FOLDS):
        base = json.loads((args.splits / f'stratified_fold{k}.json').read_text())
        splits = {s: v['targets'] for s, v in base['splits'].items()}
        test = set(splits['test'])
        linked = sorted(t for t, originals in links.items() if originals & test)
        unlinked = sorted(set(links) - set(linked))
        assert len(unlinked) >= len(linked), f'fold {k}: not enough unlinked training_only targets'
        control = sorted(rng.sample(unlinked, len(linked)))
        linked_test = sorted(test & set().union(*(links[t] for t in linked)))

        for arm, dropped in (('no_homologs', linked), ('matched_control', control)):
            train = sorted(set(splits['train']) - set(dropped))
            arm_splits = {'train': train, 'val': splits['val'], 'test': splits['test']}
            payload = {
                'data_path': base['data_path'],
                'scheme': f'homolog_leakage_{arm}',
                'fold': k,
                'seed': args.seed,
                'base_manifest': f'stratified_fold{k}.json',
                'dropped_training_only': dropped,
                'linked_test_targets': linked_test,
                'splits': {s: {'target_count': len(ts), 'targets': ts,
                               'paths': [f'{base["data_path"].rstrip("/")}/{t}.hdf5' for t in ts]}
                           for s, ts in arm_splits.items()},
            }
            (args.out / f'fold{k}_{arm}.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='ascii')

        design.append({'fold': k, 'train_targets': len(splits['train']) - len(linked),
                       'n_dropped': len(linked), 'n_linked_test': len(linked_test),
                       'n_test': len(test), 'no_homologs_dropped': ','.join(linked),
                       'matched_control_dropped': ','.join(control), 'linked_test': ','.join(linked_test)})

    design = pd.DataFrame(design)
    design.to_csv(args.out / 'leakage_design.csv', index=False)
    print(design[['fold', 'train_targets', 'n_dropped', 'n_linked_test', 'n_test']].to_string(index=False))
    print(f'wrote {2 * N_FOLDS} manifests to {args.out}')


if __name__ == '__main__':
    main()
