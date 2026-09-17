"""Paired target-level comparisons, with seeds averaged within target.

Bootstrap targets, not decoys or seed replicates. Positive improvement means
lower MSE than the full-feature baseline. All predictions must have identical
(target, graph) keys and labels. Missing runs are an error.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def read_predictions(path):
    result = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (row['target_id'], row['graph_name'])
            if key in result:
                raise ValueError(f'{path}: duplicate prediction {key}')
            result[key] = (float(row['true_target']), float(row['predicted_target']))
    if not result or not np.isfinite(list(result.values())).all():
        raise ValueError(f'{path}: empty or nonfinite predictions')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', default='baseline', help='Configuration used as the paired control')
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    if args.bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap draws')
    matrix = json.loads(args.matrix.read_text())
    runs = matrix['runs']
    predictions = {(run['config']['name'], run['seed']): read_predictions(Path(run['output']) / 'test_predictions.csv') for run in runs}
    reference = next(iter(predictions.values()))
    for label, values in predictions.items():
        if values.keys() != reference.keys() or any(values[key][0] != reference[key][0] for key in reference):
            raise ValueError(f'{label}: test cohort/labels differ')
    targets = sorted({key[0] for key in reference})
    target_keys = {target: sorted(key for key in reference if key[0] == target) for target in targets}
    metrics = {}
    for run, values in predictions.items():
        per_target = []
        for target in targets:
            pairs = np.array([values[key] for key in target_keys[target]])
            truth, pred = pairs.T
            rho = float(spearmanr(truth, pred).statistic) if len(truth) > 1 and np.std(truth) > 0 and np.std(pred) > 0 else float('nan')
            # Deterministic graph-name tie breaking, identical across runs.
            selected = int(np.argmax(pred))
            per_target.append((float(np.mean((pred - truth)**2)), rho,
                               float(truth.max() - truth[selected])))
        metrics[run] = np.array(per_target)
    seeds = sorted(seed for name, seed in predictions if name == args.reference)
    if not seeds:
        raise ValueError(f'Reference configuration absent: {args.reference}')
    for name in {name for name, seed in predictions}:
        if sorted(seed for config, seed in predictions if config == name) != seeds:
            raise ValueError(f'Seeds differ from reference for {name}')
    baseline = np.stack([metrics[args.reference, seed] for seed in seeds])
    rng = np.random.default_rng(args.seed)
    boot_indices = rng.integers(len(targets), size=(args.bootstrap, len(targets)))
    rows = []
    target_rows = []
    for name in dict.fromkeys(run['config']['name'] for run in runs):
        values = np.stack([metrics[name, seed] for seed in seeds])
        delta = (baseline[:, :, 0] - values[:, :, 0]).mean(axis=0)
        distribution = delta[boot_indices].mean(axis=1)
        low, high = np.quantile(distribution, [0.025, 0.975])
        # Familywise intervals across planned nonbaseline comparisons.
        comparisons = len({run['config']['name'] for run in runs}) - 1
        adj_low, adj_high = np.quantile(distribution, [0.025 / comparisons, 1 - 0.025 / comparisons])
        finite_rho = values[:, :, 1][np.isfinite(values[:, :, 1])]
        rows.append(dict(config=name, reference=args.reference, seeds=len(seeds), targets=len(targets),
            macro_mse=float(values[:, :, 0].mean()),
            seed_macro_mse_sd=float(values[:, :, 0].mean(axis=1).std(ddof=1)) if len(seeds) > 1 else float('nan'),
            macro_spearman=float(finite_rho.mean()) if finite_rho.size else float('nan'),
            defined_spearman_count=int(finite_rho.size),
            top1_regret=float(values[:, :, 2].mean()),
            mse_improvement=float(delta.mean()), ci95_low=float(low), ci95_high=float(high),
            familywise_low=float(adj_low), familywise_high=float(adj_high)))
        target_rows.extend(dict(config=name, target=target, mse_improvement=float(d)) for target, d in zip(targets, delta))
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in [('comparison.csv', rows), ('paired_targets.csv', target_rows)]:
        with (args.output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    print(f'Wrote paired comparisons to {args.output}; bootstrap uncertainty is across targets conditional on these seeds/split.')


if __name__ == '__main__':
    main()
