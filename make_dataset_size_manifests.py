"""Split manifests for the dataset-size comparison (original 146 vs expanded 350 targets).

Both arms use one stratified fold of gate_run/expanded_target_splits with identical val and test targets; they
differ only in train:

* expanded:      the fold's full train split (original + batch-1 targets, incl. the 45 training_only ones)
* original_only: the original-146 targets of that train split

train_gate.py requires the dataset directory to hold exactly the manifest's targets, so each arm gets its own
directory of symlinks (made by cluster/submit_dataset_size_comparison.sh); the manifest paths point there.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLUSTER_EXPERIMENT = '/gpfs/boyd/pi/pi_co54/jas485/PPI-graph_autoencoder/gate_run/dataset_size_comparison'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--splits', type=Path, default=ROOT / 'gate_run' / 'expanded_target_splits')
    parser.add_argument('--out', type=Path, default=ROOT / 'gate_run' / 'dataset_size_comparison')
    parser.add_argument('--experiment-dir', default=CLUSTER_EXPERIMENT, help='experiment directory on the cluster')
    args = parser.parse_args()

    base = json.loads((args.splits / f'stratified_fold{args.fold}.json').read_text())
    source = pd.read_csv(args.splits / 'target_assignments.csv', index_col=0).source
    splits = {s: v['targets'] for s, v in base['splits'].items()}
    arms = {
        'expanded': splits,
        'original_only': {**splits, 'train': [t for t in splits['train'] if source[t] == 'original_146']},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for arm, arm_splits in arms.items():
        data_dir = f'{args.experiment_dir}/data_{arm}'
        payload = {
            'data_path': data_dir,
            'arm': arm,
            'base_manifest': f'expanded_target_splits/stratified_fold{args.fold}.json',
            'splits': {s: {'target_count': len(ts),
                           'original_146_count': int((source[ts] == 'original_146').sum()),
                           'targets': ts, 'paths': [f'{data_dir}/{t}.hdf5' for t in ts]}
                       for s, ts in arm_splits.items()},
        }
        (args.out / f'target_splits_{arm}.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='ascii')
        (args.out / f'targets_{arm}.txt').write_text(
            ''.join(f'{t}\n' for t in sorted(set().union(*arm_splits.values()))))
        print(arm, {s: f'{len(ts)} ({(source[ts] == "original_146").sum()} original)' for s, ts in arm_splits.items()})


if __name__ == '__main__':
    main()
