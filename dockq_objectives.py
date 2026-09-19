"""Training-only DockQ range weighting and graph-weighted validation reports."""
from collections import defaultdict
import numpy as np
import h5py
import torch

EDGES = np.array([0., .2, .4, .6, .8, 1.])


def bin_indices(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('DockQ labels must be finite and within [0, 1]')
    return np.searchsorted(EDGES[1:-1].astype(np.float32), values.astype(np.float32), side='right')


def fit_range_weights(labels, cap=3., exponent=.5):
    if cap < 1 or not np.isfinite(cap):
        raise ValueError('Weight cap must be finite and >= 1')
    if not np.isfinite(exponent) or not 0 <= exponent <= 1:
        raise ValueError("Weight exponent must be in [0, 1]")
    bins = bin_indices(labels)
    if len(bins) == 0:
        raise ValueError('Cannot fit weights to an empty training split')
    counts = np.bincount(bins, minlength=5)
    # Cap the ratio to the most common bin BEFORE mean-one normalization.
    raw = (counts.max() / np.maximum(counts, 1)) ** exponent
    raw = np.minimum(raw, cap)
    weights = raw / np.mean(raw[bins])
    return dict(edges=EDGES.tolist(), counts=counts.tolist(), weights=weights.tolist(),
                exponent=exponent, max_weight_ratio=cap, normalization='mean training example weight = 1',
                source='training labels only')


def read_training_labels(dataset, indices):
    by_file = defaultdict(list)
    for index in indices:
        sample = dataset.samples[index]
        by_file[sample.path].append(sample.group_name)
    labels = []
    for path, names in sorted(by_file.items()):
        with h5py.File(path, 'r') as handle:
            labels.extend(float(handle[name]['target_scores'][dataset.target_name][()]) for name in names)
    bin_indices(labels)
    return labels


def weighted_dockq_mse(prediction, target, weights=None):
    prediction, target = prediction.view(-1), target.view(-1)
    squared = (prediction - target).square()
    if weights is None:
        return squared.mean()
    boundaries = target.new_tensor(EDGES[1:-1])
    bins = torch.bucketize(target.contiguous(), boundaries, right=True)
    # Do not normalize per batch: that would undo weighting in homogeneous batches.
    return (squared * target.new_tensor(weights)[bins]).mean()


def prediction_report(truth, prediction, targets):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    bins = bin_indices(truth)
    error = prediction-truth
    if not np.isfinite(prediction).all() or not len(truth):
        raise ValueError('Predictions must be finite and nonempty')
    by_target = defaultdict(list)
    for target, sq in zip(targets, error**2):
        by_target[str(target)].append(float(sq))
    ranges=[]
    for b in range(5):
        mask = bins==b
        ranges.append(dict(bin=b, low=float(EDGES[b]), high=float(EDGES[b+1]), n=int(mask.sum()),
            mse=float(np.mean(error[mask]**2)) if mask.any() else None,
            bias=float(np.mean(error[mask])) if mask.any() else None))
    return dict(n=len(truth), targets=len(by_target), pooled_mse=float(np.mean(error**2)),
                macro_target_mse=float(np.mean([np.mean(v) for v in by_target.values()])),
                macro_bin_mse=float(np.mean([r['mse'] for r in ranges if r['n']])),
                occupied_bins=sum(r['n']>0 for r in ranges), bins=ranges)
