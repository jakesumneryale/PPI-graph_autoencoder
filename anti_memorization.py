"""Training-time defences against per-target memorization of DockQ.

With ~105 training targets and ~1,200 decoys each, the network can learn each
training target's native contacts and score decoys by overlap with them. That
fits training DockQ almost perfectly and transfers to no unseen target. The
pieces here attack that shortcut from different directions:

- ``TargetAdversary``: a target-identity classifier behind gradient reversal,
  pushing the graph embedding the DockQ head reads to be target-invariant.
- ``augment_batch``: denoising-style input corruption (edge dropout, node
  feature masking, C-alpha distance jitter) so exact contact patterns are an
  unreliable cue. Reconstruction targets stay the clean graph.
- ``TargetBalancedSampler``: a fixed number of decoys per training target per
  epoch, redrawn every epoch.
- ``ModelEMA``: an exponential moving average of the weights, used for
  validation and checkpointing.

The "encoder never sees DockQ" arm is a model flag
(``GraphAttentionAutoencoder.detach_quality_input``), not part of this module.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


def gradient_reversal(x: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Identity forward; multiplies the incoming gradient by ``-coefficient``."""
    return _GradientReversal.apply(x, float(coefficient))


def dann_coefficient(progress: float, gamma: float = 10.0) -> float:
    """Ganin & Lempitsky ramp from 0 to 1 over training ``progress`` in [0, 1].

    Starting at zero lets the DockQ head and the adversary form before the
    reversed gradient reaches the encoder at full strength.
    """
    progress = min(max(progress, 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


class TargetAdversary(nn.Module):
    """Predicts which training target a graph embedding came from."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_targets: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_targets < 2:
            raise ValueError("A target adversary needs at least two training targets.")
        self.num_targets = num_targets
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, graph_z: torch.Tensor, coefficient: float) -> torch.Tensor:
        return self.net(gradient_reversal(graph_z, coefficient))


def adversary_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Cross-entropy and accuracy. Chance accuracy is ``1 / num_targets``."""
    loss = F.cross_entropy(logits, labels)
    accuracy = float((logits.argmax(dim=1) == labels).float().mean().detach().cpu())
    return loss, accuracy


def augment_batch(
    batch,
    *,
    edge_dropout: float = 0.0,
    node_feature_mask: float = 0.0,
    ca_dist_noise: float = 0.0,
    ca_dist_column: int | None = None,
):
    """Return a corrupted copy of ``batch`` for the encoder; ``batch`` is untouched.

    - ``edge_dropout``: drop each contact with this probability. Both directions
      of an undirected contact are dropped together when the reverse edge exists.
    - ``node_feature_mask``: zero the whole feature row of this fraction of nodes.
    - ``ca_dist_noise``: add N(0, sigma^2) Angstrom noise to the C-alpha distance
      column, clamped at zero.
    """
    for name, value in (("edge_dropout", edge_dropout), ("node_feature_mask", node_feature_mask)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must lie in [0, 1)")
    if ca_dist_noise < 0:
        raise ValueError("ca_dist_noise must be nonnegative")
    if ca_dist_noise > 0 and ca_dist_column is None:
        raise ValueError("ca_dist_noise requires the ca_dist edge column")

    out = batch.clone()
    if node_feature_mask > 0:
        keep = torch.rand(out.x.size(0), device=out.x.device) >= node_feature_mask
        out.x = out.x.float() * keep.unsqueeze(1)

    edge_attr = getattr(out, "edge_attr", None)
    if ca_dist_noise > 0 and edge_attr is not None and edge_attr.numel() > 0:
        edge_attr = edge_attr.float().clone()
        noisy = edge_attr[:, ca_dist_column] + ca_dist_noise * torch.randn_like(edge_attr[:, ca_dist_column])
        edge_attr[:, ca_dist_column] = noisy.clamp_min(0.0)
        out.edge_attr = edge_attr

    if edge_dropout > 0 and out.edge_index.size(1) > 0:
        src, dst = out.edge_index
        num_nodes = out.x.size(0)
        # One coin per unordered pair, so a contact stored in both directions
        # is either kept or dropped as a whole.
        low, high = torch.minimum(src, dst), torch.maximum(src, dst)
        pair_id = low * num_nodes + high
        unique_pairs, inverse = torch.unique(pair_id, return_inverse=True)
        keep_pair = torch.rand(unique_pairs.numel(), device=pair_id.device) >= edge_dropout
        keep = keep_pair[inverse]
        out.edge_index = out.edge_index[:, keep]
        if getattr(out, "edge_attr", None) is not None and out.edge_attr.size(0) == keep.numel():
            out.edge_attr = out.edge_attr[keep]
    return out


class TargetBalancedSampler(Sampler[int]):
    """Each epoch yields ``per_target`` dataset positions from every target, shuffled.

    ``groups`` lists, per target, the positions (in the dataset the loader
    indexes) of that target's decoys. Targets with fewer decoys than
    ``per_target`` contribute all of them. Draws are without replacement within
    an epoch and change every epoch.
    """

    def __init__(self, groups: list[list[int]], per_target: int, seed: int) -> None:
        if per_target < 1:
            raise ValueError("per_target must be positive")
        if not groups or any(not group for group in groups):
            raise ValueError("Every target group must be nonempty")
        self.groups = [list(group) for group in groups]
        self.per_target = per_target
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return sum(min(len(group), self.per_target) for group in self.groups)

    def __iter__(self):
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        self.epoch += 1
        chosen: list[int] = []
        for group in self.groups:
            chosen.extend(rng.sample(group, min(len(group), self.per_target)))
        rng.shuffle(chosen)
        return iter(chosen)


class ModelEMA:
    """Exponential moving average of floating-point parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie in (0, 1)")
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Standard warmup so early averages are not dominated by the random init.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for name, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.shadow[name].copy_(value)

    @contextmanager
    def applied(self, model: nn.Module):
        """Temporarily load the averaged weights into ``model``."""
        backup = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)
