"""Refresh the portable checkpoint metadata used by analyze_seed_replicates.ipynb.

Run with the project's PyTorch environment. Hashes bind metadata to the exact
checkpoint and CSV inputs; the analysis notebook itself does not need PyTorch.
"""
import hashlib
import json
from pathlib import Path
import torch


def main():
    directory = Path(__file__).resolve().parent
    records = {}
    for path in sorted((directory / 'seed_replicates').glob('*/gate_model.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        records[path.parent.name] = {key: checkpoint.get(key) for key in
            ('args', 'best_epoch', 'best_metric', 'best_metric_value', 'node_features',
             'edge_features', 'edge_recon_features', 'edge_feature_transforms')}
        records[path.parent.name]['sha256'] = {
            name: hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
            for name in ('test_predictions.csv', 'loss_history.csv', 'gate_model.pt')}
    if not records:
        raise FileNotFoundError('No seed-replicate checkpoints found')
    (directory / 'seed_replicates_metadata.json').write_text(json.dumps(records, indent=2, default=str) + '\n')
    print(f'Extracted metadata for {len(records)} checkpoints')


if __name__ == '__main__':
    main()
