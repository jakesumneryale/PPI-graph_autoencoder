"""Training-only DockQ objectives (range/hinge weighting, CAPRI head, within-target ranking)
and graph-weighted validation reports."""
from collections import defaultdict
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# CAPRI quality classes on DockQ: incorrect < 0.23 <= acceptable < 0.49 <= medium < 0.80 <= high.
CAPRI_EDGES = (0.23, 0.49, 0.80)
CAPRI_CLASSES = ('incorrect', 'acceptable', 'medium', 'high')


def capri_class(target):
    """Integer CAPRI class (0-3) for DockQ labels, as a tensor."""
    return torch.bucketize(target.contiguous().view(-1), target.new_tensor(CAPRI_EDGES), right=True)


class CapriHead(nn.Module):
    """Predicts the CAPRI class of a decoy from its graph embedding (softmax over 4 classes).

    Kept outside the GATE model so model checkpoints and the reconstruction evaluator are unchanged.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, len(CAPRI_CLASSES)))

    def forward(self, graph_z):
        return self.net(graph_z)


def hinge_weights(values, beta, start=0.23):
    """1 below ``start``, rising linearly to ``beta`` at DockQ = 1."""
    if not np.isfinite(beta) or beta < 1:
        raise ValueError('Hinge beta must be finite and >= 1')
    if not 0 <= start < 1:
        raise ValueError('Hinge start must be in [0, 1)')
    ramp = (values - start) / (1 - start)
    ramp = ramp.clamp(0, 1) if torch.is_tensor(ramp) else np.clip(ramp, 0, 1)
    return 1 + (beta - 1) * ramp


def fit_hinge_normalizer(labels, beta, start=0.23):
    """Mean raw hinge weight over training labels; dividing by it keeps the mean training weight at 1,
    so raising beta shifts emphasis toward high DockQ without raising DockQ's overall loss weight."""
    labels = np.asarray(labels, dtype=float)
    bin_indices(labels)
    return dict(beta=float(beta), start=float(start), normalizer=float(np.mean(hinge_weights(labels, beta, start))),
                normalization='mean training example weight = 1 (on true labels)', source='training labels only')


def hinge_dockq_mse(prediction, target, beta, start, normalizer, use_prediction=False):
    """Hinge-weighted DockQ MSE. With ``use_prediction`` the weight uses max(true, predicted), so a
    poor decoy predicted high is penalized like a good one; the prediction enters the weight detached."""
    prediction, target = prediction.view(-1), target.view(-1)
    basis = torch.maximum(target, prediction.detach().clamp(0, 1)) if use_prediction else target
    return ((prediction - target).square() * hinge_weights(basis, beta, start)).mean() / normalizer


def pairwise_rank_loss(prediction, target, groups, start=0.23, temperature=0.1):
    """Within-target pairwise logistic ranking loss.

    Uses pairs of decoys from the same target (``groups``: one id per graph) where at least one decoy is
    at or above ``start``; each pair is weighted by its DockQ difference, so separating a high decoy from a
    medium one counts more than separating two near-equal ones. Returns (loss, number of pairs).
    """
    prediction, target, groups = prediction.view(-1), target.view(-1), groups.view(-1)
    i, j = torch.triu_indices(len(target), len(target), offset=1, device=target.device)
    diff = target[i] - target[j]
    keep = (groups[i] == groups[j]) & (diff != 0) & (torch.maximum(target[i], target[j]) >= start)
    if not keep.any():
        return prediction.sum() * 0.0, 0
    i, j, diff = i[keep], j[keep], diff[keep]
    margin = torch.sign(diff) * (prediction[i] - prediction[j]) / temperature
    return (diff.abs() * F.softplus(-margin)).sum() / diff.abs().sum(), int(keep.sum())


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
