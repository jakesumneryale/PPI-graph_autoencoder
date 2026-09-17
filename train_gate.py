"""Train the GATE model on PPI graph HDF5 files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

try:
    from torch_geometric.utils import batched_negative_sampling
except ImportError:  # pragma: no cover - compatibility fallback for older PyG.
    batched_negative_sampling = None

from GATE_model import GraphAttentionAutoencoder
from EGNN_model import build_graph_model
from evaluate_gate_reconstruction import evaluate_checkpoint, write_csv
from protein_hdf5_dataset import (
    DEFAULT_EDGE_FEATURES,
    DEFAULT_NODE_FEATURES,
    HDF5GraphKey,
    ProteinGraphHDF5Dataset,
    apply_cluster_path_defaults,
    compute_edge_feature_stats,
)


LOSS_METRIC_NAMES = ("loss", "node_mse", "edge_attr_mse", "edge_presence_bce", "target_mse")
LOSS_TERM_NAMES = ("node_mse", "edge_attr_mse", "edge_presence_bce", "target_mse")
NODE_FEATURE_SET_CHOICES = {
    "all": DEFAULT_NODE_FEATURES,
    "no-aa-identity": ("chain", "interface_nodes"),
}


def parse_feature_transforms(raw: str | None) -> dict[str, str]:
    """Parse ``--edge-feature-transforms`` (``NAME=TRANSFORM,...``)."""
    if not raw:
        return {}
    transforms: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected NAME=TRANSFORM in --edge-feature-transforms, got {item!r}.")
        name, transform = item.split("=", 1)
        transforms[name.strip()] = transform.strip()
    return transforms


def resolve_edge_recon_features(args, edge_features: tuple[str, ...]) -> tuple[str, ...]:
    if args.edge_recon_features is None:
        return edge_features
    requested = tuple(item.strip() for item in args.edge_recon_features.split(",") if item.strip())
    unknown = [name for name in requested if name not in edge_features]
    if unknown:
        raise ValueError(
            f"--edge-recon-features contains feature(s) not in --edge-features: {unknown}"
        )
    return requested


def edge_recon_column_indices(
    edge_feature_slices: dict[str, slice], edge_recon_features: tuple[str, ...]
) -> list[int]:
    columns: list[int] = []
    for name in edge_recon_features:
        span = edge_feature_slices[name]
        columns.extend(range(span.start, span.stop))
    return columns


def parse_feature_list(raw: str | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return defaults
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_worker_start_method(args) -> str:
    if args.worker_start_method == "auto":
        return "spawn" if args.cluster else "default"
    return args.worker_start_method


def make_dataloader_kwargs(args, device: torch.device, shuffle: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
    }
    if device.type == "cuda":
        kwargs["pin_memory"] = True

    worker_start_method = resolve_worker_start_method(args)
    if args.num_workers > 0:
        if worker_start_method != "default":
            kwargs["multiprocessing_context"] = worker_start_method
        kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor is not None:
            kwargs["prefetch_factor"] = args.prefetch_factor

    return kwargs


def resolve_node_features(args) -> tuple[str, ...]:
    if args.node_features is not None:
        base_features = list(parse_feature_list(args.node_features, DEFAULT_NODE_FEATURES))
    else:
        base_features = list(NODE_FEATURE_SET_CHOICES[args.node_feature_set])

    if args.use_rsasa_i and "rsasa_i" not in base_features:
        base_features.append("rsasa_i")
    if args.use_drsasa and "drsasa" not in base_features:
        base_features.append("drsasa")
    return tuple(base_features)


def bounded_split_count(total_items: int, fraction: float, min_remaining: int) -> int:
    if fraction <= 0 or total_items <= min_remaining:
        return 0

    proposed = int(round(total_items * fraction))
    proposed = max(1, proposed)
    return min(proposed, total_items - min_remaining)


def target_name_from_path(path: Path) -> str:
    return path.stem


def split_dataset_by_target(
    samples: list[HDF5GraphKey],
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, list[int]], dict[str, list[Path]]]:
    unique_paths = sorted({sample.path for sample in samples})
    if len(unique_paths) < 3:
        raise ValueError("Need at least 3 target files to create train/val/test splits.")

    shuffled_paths = unique_paths[:]
    random.Random(seed).shuffle(shuffled_paths)

    test_count = bounded_split_count(len(shuffled_paths), test_fraction, min_remaining=2)
    remaining_paths = shuffled_paths[test_count:]
    test_paths = shuffled_paths[:test_count]

    val_count = bounded_split_count(len(remaining_paths), val_fraction, min_remaining=1)
    val_paths = remaining_paths[:val_count]
    train_paths = remaining_paths[val_count:]

    split_paths = {
        "train": sorted(train_paths),
        "val": sorted(val_paths),
        "test": sorted(test_paths),
    }
    path_to_split = {
        path: split_name
        for split_name, split_path_list in split_paths.items()
        for path in split_path_list
    }
    split_indices = {"train": [], "val": [], "test": []}
    for index, sample in enumerate(samples):
        split_indices[path_to_split[sample.path]].append(index)

    for split_name, indices in split_indices.items():
        if not indices:
            raise ValueError(f"{split_name} split is empty; adjust the split fractions.")

    return split_indices, split_paths


def load_target_split_manifest(
    manifest_path: Path,
    samples: list[HDF5GraphKey],
) -> tuple[dict[str, list[int]], dict[str, list[Path]]]:
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    unique_paths = sorted({sample.path for sample in samples})
    available_paths = {path.resolve() for path in unique_paths}

    split_paths: dict[str, list[Path]] = {}
    for split_name in ("train", "val", "test"):
        raw_paths = payload["splits"][split_name]["paths"]
        split_paths[split_name] = [Path(raw_path).resolve() for raw_path in raw_paths]

    manifest_paths = {
        path
        for split_path_list in split_paths.values()
        for path in split_path_list
    }
    if manifest_paths != available_paths:
        missing_from_manifest = sorted(str(path) for path in available_paths - manifest_paths)
        missing_from_dataset = sorted(str(path) for path in manifest_paths - available_paths)
        details = []
        if missing_from_manifest:
            details.append(f"missing from manifest: {missing_from_manifest[:5]}")
        if missing_from_dataset:
            details.append(f"missing from dataset: {missing_from_dataset[:5]}")
        raise ValueError(
            f"Split manifest {manifest_path} does not match the current dataset target files. "
            + "; ".join(details)
        )

    path_to_split = {
        path: split_name
        for split_name, split_path_list in split_paths.items()
        for path in split_path_list
    }
    split_indices = {"train": [], "val": [], "test": []}
    for index, sample in enumerate(samples):
        split_indices[path_to_split[sample.path.resolve()]].append(index)

    for split_name, indices in split_indices.items():
        if not indices:
            raise ValueError(f"{split_name} split from {manifest_path} is empty.")

    return split_indices, split_paths


def summarize_split(indices: list[int], samples: list[HDF5GraphKey]) -> dict[str, int]:
    target_count = len({samples[index].path for index in indices})
    return {"targets": target_count, "samples": len(indices)}


def save_target_split_manifest(
    output_path: Path,
    split_paths: dict[str, list[Path]],
    split_indices: dict[str, list[int]],
    samples: list[HDF5GraphKey],
    args,
) -> None:
    payload = {
        "data_path": args.data,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "val_fraction": args.val_fraction,
        "splits": {},
    }
    for split_name, paths in split_paths.items():
        payload["splits"][split_name] = {
            "target_count": len(paths),
            "sample_count": len(split_indices[split_name]),
            "targets": [target_name_from_path(path) for path in paths],
            "paths": [str(path) for path in paths],
        }

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def make_history_fieldnames() -> list[str]:
    fieldnames = ["epoch"]
    for split_name in ("train", "val", "test"):
        fieldnames.extend(f"{split_name}_{metric_name}" for metric_name in LOSS_METRIC_NAMES)
    fieldnames.extend(f"weight_{term_name}" for term_name in LOSS_TERM_NAMES)
    fieldnames.extend(f"train_weighted_{term_name}" for term_name in LOSS_TERM_NAMES)
    return fieldnames


def flatten_history_row(epoch: int, metrics_by_split: dict[str, dict[str, float]]) -> dict[str, float | int]:
    row: dict[str, float | int] = {"epoch": epoch}
    for split_name, metrics in metrics_by_split.items():
        for metric_name in LOSS_METRIC_NAMES:
            row[f"{split_name}_{metric_name}"] = metrics.get(metric_name, 0.0)
    train_metrics = metrics_by_split.get("train", {})
    for term_name in LOSS_TERM_NAMES:
        row[f"weight_{term_name}"] = train_metrics.get(f"weight_{term_name}", 0.0)
        row[f"train_weighted_{term_name}"] = train_metrics.get(f"weighted_{term_name}", 0.0)
    return row


def collect_prediction_rows(model, loader, device, args, node_feature_set: str, node_features: tuple[str, ...]):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            quality_pred = output["quality_pred"]
            if quality_pred is None:
                raise ValueError("Model did not produce quality predictions.")

            graph_names = list(batch.graph_name)
            hdf5_paths = list(batch.hdf5_path)
            sample_indices = batch.sample_index.view(-1).detach().cpu().tolist()
            true_values = batch.y.float().view(-1).detach().cpu().tolist()
            predicted_values = quality_pred.detach().cpu().tolist()

            for sample_index, hdf5_path, graph_name, true_value, predicted_value in zip(
                sample_indices,
                hdf5_paths,
                graph_names,
                true_values,
                predicted_values,
                strict=True,
            ):
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "target_name": args.target_name,
                        "target_id": target_name_from_path(Path(hdf5_path)),
                        "hdf5_path": hdf5_path,
                        "graph_name": graph_name,
                        "true_target": float(true_value),
                        "predicted_target": float(predicted_value),
                        "node_feature_set": node_feature_set,
                        "node_features": ",".join(node_features),
                    }
                )
    return rows


def save_prediction_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_index",
        "target_name",
        "target_id",
        "hdf5_path",
        "graph_name",
        "true_target",
        "predicted_target",
        "node_feature_set",
        "node_features",
    ]
    with output_path.open("w", newline="", encoding="ascii") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_negative_edges(batch, num_neg_samples: int) -> torch.Tensor:
    if num_neg_samples <= 0:
        return batch.edge_index.new_empty((2, 0))

    if batched_negative_sampling is not None and hasattr(batch, "batch"):
        return batched_negative_sampling(
            batch.edge_index,
            batch.batch,
            num_neg_samples=num_neg_samples,
        )
    return negative_sampling(
        batch.edge_index,
        num_nodes=batch.num_nodes,
        num_neg_samples=num_neg_samples,
    )


class LossShareBalancer:
    """Turn requested loss *shares* into per-term weights.

    The four GATE terms differ by orders of magnitude: edge-attribute MSE is
    computed on raw Angstrom-scale features (~2 without Voronoi area, ~10-35 with
    it) while DockQ MSE sits around 0.08.  Using the shares directly as weights
    therefore does not produce those shares -- a 0.60 weight on a 0.08 term still
    loses to a 0.15 weight on a 10.0 term by more than an order of magnitude.

    Each term is divided by a running (bias-corrected EMA) estimate of its own
    magnitude before the share is applied, so the requested share is the fraction
    of the total loss, and therefore of the gradient signal, that the term
    actually contributes.  Statistics are only updated on training batches.
    """

    def __init__(self, shares: dict[str, float], momentum: float = 0.99, eps: float = 1e-8) -> None:
        total = sum(shares.values())
        if total <= 0.0:
            raise ValueError("Loss shares must sum to a positive value.")
        if any(value < 0.0 for value in shares.values()):
            raise ValueError("Loss shares must be non-negative.")
        self.shares = {name: value / total for name, value in shares.items()}
        self.momentum = momentum
        self.eps = eps
        self._ema = {name: 0.0 for name in self.shares}
        self._steps = 0

    def observe(self, losses: dict[str, torch.Tensor]) -> None:
        self._steps += 1
        for name in self.shares:
            magnitude = abs(float(losses[name].detach()))
            self._ema[name] = self.momentum * self._ema[name] + (1.0 - self.momentum) * magnitude

    def weights(self) -> dict[str, float]:
        if self._steps == 0:
            return dict(self.shares)
        correction = 1.0 - self.momentum ** self._steps
        return {
            name: share / max(self._ema[name] / correction, self.eps)
            for name, share in self.shares.items()
        }


def build_lr_scheduler(optimizer, args, steps_per_epoch: int):
    """Optional linear warmup into cosine decay.

    Returns ``None`` for ``--lr-schedule constant``, which is the default and
    reproduces the original fixed-rate behaviour.
    """
    if args.lr_schedule == "constant":
        return None
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = min(max(args.warmup_steps, 0), total_steps - 1)
    final_fraction = args.lr_final_fraction

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_fraction + (1.0 - final_fraction) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_loss_weights(args) -> dict[str, float]:
    return {
        "node_mse": args.node_weight,
        "edge_attr_mse": args.edge_attr_weight,
        "edge_presence_bce": args.edge_presence_weight,
        "target_mse": args.target_weight,
    }


def build_loss_balancer(args) -> "LossShareBalancer | None":
    if args.loss_weight_mode != "shares":
        return None
    shares = {
        "node_mse": args.node_share,
        "edge_attr_mse": args.edge_attr_share,
        "edge_presence_bce": args.edge_presence_share,
        "target_mse": args.target_share,
    }
    return LossShareBalancer(shares, momentum=args.loss_share_momentum)


def compute_gate_loss(model, batch, output, args, balancer=None, update_balancer: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    losses: dict[str, torch.Tensor] = {}

    losses["node_mse"] = F.mse_loss(output["node_recon"], batch.x.float())

    if (
        getattr(batch, "edge_attr", None) is not None
        and batch.edge_attr.numel() > 0
        and output["edge_recon"].size(1) > 0
    ):
        edge_targets = model.select_edge_recon_targets(batch.edge_attr.float())
        losses["edge_attr_mse"] = F.mse_loss(output["edge_recon"], edge_targets)
    else:
        losses["edge_attr_mse"] = output["node_recon"].new_tensor(0.0)

    pos_logits = output["edge_logits"]
    neg_edge_index = sample_negative_edges(batch, num_neg_samples=pos_logits.numel())
    if neg_edge_index.numel() > 0:
        neg_logits = model.decode_edge_logits(output["node_embeddings"], neg_edge_index)
        link_logits = torch.cat([pos_logits, neg_logits], dim=0)
        link_labels = torch.cat(
            [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)],
            dim=0,
        )
        losses["edge_presence_bce"] = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    else:
        losses["edge_presence_bce"] = output["node_recon"].new_tensor(0.0)

    if getattr(batch, "y", None) is not None and output["quality_pred"] is not None:
        losses["target_mse"] = F.mse_loss(output["quality_pred"], batch.y.float().view(-1))
    else:
        losses["target_mse"] = output["node_recon"].new_tensor(0.0)

    if balancer is not None:
        if update_balancer:
            balancer.observe(losses)
        weights = balancer.weights()
    else:
        weights = resolve_loss_weights(args)

    total = sum(
        (weights[name] * losses[name] for name in LOSS_TERM_NAMES),
        start=output["node_recon"].new_tensor(0.0),
    )
    metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
    metrics["loss"] = float(total.detach().cpu())
    for name in LOSS_TERM_NAMES:
        metrics[f"weight_{name}"] = weights[name]
        metrics[f"weighted_{name}"] = weights[name] * metrics[name]
    return total, metrics


def run_epoch(model, loader, optimizer, device, args, train: bool, balancer=None, scheduler=None) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            output = model(batch)
            loss, metrics = compute_gate_loss(
                model, batch, output, args, balancer=balancer, update_balancer=train
            )
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        num_batches += 1

    return {name: value / max(num_batches, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GATE on PPI graph HDF5 files.")
    parser.add_argument("--cluster", action="store_true", help="Use the cluster default data directories.")
    parser.add_argument("--data", default=None, help="HDF5 file or directory.")
    parser.add_argument(
        "--model-list-dir",
        default=None,
        help="Optional directory containing <target>.txt files of HDF5 model keys to include.",
    )
    parser.add_argument("--target-name", default="DockQ")
    parser.add_argument(
        "--node-feature-set",
        choices=tuple(NODE_FEATURE_SET_CHOICES),
        default="all",
        help="Named node feature configuration to use when --node-features is not provided.",
    )
    parser.add_argument("--node-features", default=None, help="Comma-separated node feature names.")
    parser.add_argument("--use-rsasa-i", action="store_true", help="Append the per-decoy avg rSASA_i scalar to each node.")
    parser.add_argument("--use-drsasa", action="store_true", help="Append the per-decoy avg dRSASA scalar to each node.")
    parser.add_argument(
        "--optional-node-features-dir",
        default=None,
        help="Directory containing optional rsasa/drsasa CSV files.",
    )
    parser.add_argument("--edge-features", default=None, help="Comma-separated edge feature names.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--worker-start-method",
        choices=("auto", "default", "spawn", "fork", "forkserver"),
        default="auto",
        help="DataLoader worker start method. On cluster runs, auto uses spawn to avoid HDF5/fork hangs.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs when num_workers > 0.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="PyTorch DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="'cosine' applies linear warmup then cosine decay to --lr-final-fraction of --lr.",
    )
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-final-fraction", type=float, default=0.03)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--residual-connections",
        "--residual_connections",
        action="store_true",
        help="Use four GAT layers with residual connections instead of the original two-layer encoder.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of target files reserved for test.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of the remaining non-test target files reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Optional JSON split manifest. If it exists, reuse that exact train/val/test target split.",
    )
    parser.add_argument(
        "--prepare-splits-only",
        action="store_true",
        help=(
            "Write the split manifest and exit without training. Run this once before dispatching "
            "seed replicates so they share one split instead of racing to create their own."
        ),
    )
    parser.add_argument(
        "--edge-recon-features",
        default=None,
        help=(
            "Comma-separated subset of --edge-features that the decoder is trained to reconstruct. "
            "Defaults to all of them. Features left out are still encoder inputs (they inform "
            "attention) but are not regression targets -- use this for voronoi_contact_area, whose "
            "raw-scale reconstruction otherwise takes over the loss."
        ),
    )
    parser.add_argument(
        "--edge-feature-transforms",
        default=None,
        help=(
            "Comma-separated NAME=TRANSFORM pairs applied to edge features on load, "
            "e.g. 'voronoi_contact_area=log1p'. Supported transforms: none, log1p."
        ),
    )
    parser.add_argument(
        "--standardize-edge-features",
        default=None,
        help=(
            "Comma-separated edge features to z-score after their transform, using statistics "
            "computed from the training split only. Leave features whose reconstruction metrics "
            "are reported in physical units (ca_dist) out of this list."
        ),
    )
    parser.add_argument(
        "--edge-stats-seed",
        type=int,
        default=None,
        help=(
            "Seed for sampling graphs when estimating edge-feature statistics. Defaults to --seed. "
            "Fix it across seed replicates so every run normalises inputs identically."
        ),
    )
    parser.add_argument(
        "--edge-stats-max-graphs",
        type=int,
        default=200,
        help="Number of training graphs sampled when estimating edge-feature statistics.",
    )
    parser.add_argument(
        "--loss-weight-mode",
        choices=("fixed", "shares"),
        default="fixed",
        help=(
            "'fixed' multiplies each raw loss term by its --*-weight. "
            "'shares' instead normalises every term by a running estimate of its own "
            "magnitude and then applies --*-share, so each term contributes that "
            "fraction of the total loss regardless of its natural scale."
        ),
    )
    parser.add_argument("--node-weight", type=float, default=1.0)
    parser.add_argument("--edge-attr-weight", type=float, default=1.0)
    parser.add_argument("--edge-presence-weight", type=float, default=0.1)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--node-share", type=float, default=0.15)
    parser.add_argument("--edge-attr-share", type=float, default=0.15)
    parser.add_argument("--edge-presence-share", type=float, default=0.10)
    parser.add_argument("--target-share", type=float, default=0.60)
    parser.add_argument(
        "--loss-share-momentum",
        type=float,
        default=0.99,
        help="EMA momentum for the per-term magnitude estimates used by --loss-weight-mode shares.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=LOSS_METRIC_NAMES,
        default="loss",
        help=(
            "Validation metric used to pick the saved checkpoint. The default total 'loss' is "
            "dominated by edge-attribute reconstruction; use 'target_mse' to keep the epoch that "
            "actually predicts DockQ best."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-edge-threshold", type=float, default=0.5)
    parser.add_argument("--eval-negative-ratio", type=float, default=1.0)
    parser.add_argument("--eval-log-every", type=int, default=1000)
    parser.add_argument("--strict-hdf5", action="store_true", help="Fail instead of skipping unreadable HDF5 files.")
    parser.add_argument("--architecture", choices=("gat", "egnn"), default="gat")
    parser.add_argument("--pooling", choices=("all", "interface"), default="all")
    parser.add_argument("--use-esm", action="store_true")
    parser.add_argument("--esm-projection-dim", type=int, default=64)
    parser.add_argument("--validation-only-during-training", action="store_true",
                        help="Keep the test set sealed until the best validation checkpoint is chosen.")
    args = parser.parse_args()
    apply_cluster_path_defaults(args)

    if not 0.0 <= args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0, 1).")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1).")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    node_features = resolve_node_features(args)
    edge_features = parse_feature_list(args.edge_features, DEFAULT_EDGE_FEATURES)
    worker_start_method = resolve_worker_start_method(args)

    print("Indexing dataset...")

    edge_feature_transforms = parse_feature_transforms(args.edge_feature_transforms)
    dataset = ProteinGraphHDF5Dataset(
        args.data,
        node_features=node_features,
        edge_features=edge_features,
        target_name=args.target_name,
        require_target=True,
        max_samples=args.max_samples,
        skip_invalid_files=not args.strict_hdf5,
        optional_node_features_dir=args.optional_node_features_dir,
        model_list_dir=args.model_list_dir,
        edge_feature_transforms=edge_feature_transforms,
        use_esm=args.use_esm, require_pos=args.architecture == "egnn",
    )
    print("Dataset indexed.")
    first_graph = dataset[0]

    edge_feature_slices = dataset.edge_feature_slices()
    edge_recon_features = resolve_edge_recon_features(args, edge_features)
    recon_columns = edge_recon_column_indices(edge_feature_slices, edge_recon_features)

    model = build_graph_model(
        architecture=args.architecture, pooling=args.pooling,
        esm_dim=first_graph.esm.size(1) if args.use_esm else 0,
        esm_projection_dim=args.esm_projection_dim,
        in_node_feats=first_graph.x.size(1),
        in_edge_feats=first_graph.edge_attr.size(1),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        residual_connections=args.residual_connections,
        predict_target=True,
        out_edge_feats=len(recon_columns),
    ).to(device)
    if edge_recon_features != edge_features:
        model.set_edge_recon_index(recon_columns)
        skipped = [name for name in edge_features if name not in edge_recon_features]
        print(f"Edge features reconstructed: {edge_recon_features} (input-only: {tuple(skipped)})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "gate_model.pt"
    history_path = output_dir / "loss_history.csv"
    test_predictions_path = output_dir / "test_predictions.csv"
    reconstruction_summary_path = output_dir / "reconstruction_summary.csv"
    reconstruction_by_target_path = output_dir / "reconstruction_by_target.csv"
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else output_dir / "target_splits.json"
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if args.split_manifest and split_manifest_path.exists():
        split_indices, split_paths = load_target_split_manifest(split_manifest_path, dataset.samples)
        split_manifest_status = "loaded"
    else:
        split_indices, split_paths = split_dataset_by_target(
            dataset.samples,
            test_fraction=args.test_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        save_target_split_manifest(split_manifest_path, split_paths, split_indices, dataset.samples, args)
        split_manifest_status = "saved"

    if args.prepare_splits_only:
        counts = {name: len(paths) for name, paths in split_paths.items()}
        print(
            f"Split manifest {split_manifest_status}: {split_manifest_path}\n"
            f"Targets: train={counts['train']}, val={counts['val']}, test={counts['test']}\n"
            f"Samples: train={len(split_indices['train'])}, val={len(split_indices['val'])}, "
            f"test={len(split_indices['test'])}\n"
            "Exiting before training (--prepare-splits-only)."
        )
        return

    standardize_features = tuple(
        item.strip() for item in (args.standardize_edge_features or "").split(",") if item.strip()
    )
    unknown_standardize = [name for name in standardize_features if name not in edge_features]
    if unknown_standardize:
        raise ValueError(
            f"--standardize-edge-features contains feature(s) not in --edge-features: {unknown_standardize}"
        )
    edge_feature_stats: dict[str, tuple[float, float]] = {}
    if standardize_features:
        print(f"Estimating edge-feature statistics from the training split ({len(split_paths['train'])} targets)...")
        edge_feature_stats = compute_edge_feature_stats(
            split_paths["train"],
            standardize_features,
            edge_feature_transforms,
            max_graphs=args.edge_stats_max_graphs,
            model_list_dir=args.model_list_dir,
            seed=args.seed if args.edge_stats_seed is None else args.edge_stats_seed,
        )
        absent_stats = set(standardize_features) - set(edge_feature_stats)
        if absent_stats:
            raise ValueError(f"No valid training observations to standardize: {sorted(absent_stats)}")
        # Applied in place: the single dataset object is shared by all three
        # Subsets, and the dataloaders below have not been built yet.
        dataset.edge_feature_stats = edge_feature_stats
        for name, (mean, std) in sorted(edge_feature_stats.items()):
            transform = edge_feature_transforms.get(name, "none")
            print(f"  {name}: transform={transform} mean={mean:.4f} std={std:.4f}")

    train_dataset = torch.utils.data.Subset(dataset, split_indices["train"])
    val_dataset = torch.utils.data.Subset(dataset, split_indices["val"])
    test_dataset = torch.utils.data.Subset(dataset, split_indices["test"])

    print(
        "Building dataloaders with "
        f"num_workers={args.num_workers}, worker_start_method={worker_start_method}, "
        f"pin_memory={'yes' if device.type == 'cuda' else 'no'}..."
    )
    train_loader = DataLoader(train_dataset, **make_dataloader_kwargs(args, device, shuffle=True))
    val_loader = DataLoader(val_dataset, **make_dataloader_kwargs(args, device, shuffle=False))
    test_loader = DataLoader(test_dataset, **make_dataloader_kwargs(args, device, shuffle=False))
    print("Dataloaders ready.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    if scheduler is None:
        print(f"Learning rate: {args.lr:g} (constant)")
    else:
        print(
            f"Learning rate: {args.lr:g} with {args.warmup_steps}-step warmup then cosine decay "
            f"to {args.lr * args.lr_final_fraction:g} over {len(train_loader) * args.epochs} steps"
        )

    train_summary = summarize_split(split_indices["train"], dataset.samples)
    val_summary = summarize_split(split_indices["val"], dataset.samples)
    test_summary = summarize_split(split_indices["test"], dataset.samples)

    print(f"Device: {device}")
    print(f"Data path: {args.data}")
    if dataset.skipped_files:
        print(f"Skipped unreadable HDF5 file(s): {len(dataset.skipped_files)}")
    print(
        "Targets: "
        f"total={len(split_paths['train']) + len(split_paths['val']) + len(split_paths['test'])}, "
        f"train={train_summary['targets']}, val={val_summary['targets']}, test={test_summary['targets']}"
    )
    print(
        "Samples: "
        f"total={len(dataset)}, train={train_summary['samples']}, "
        f"val={val_summary['samples']}, test={test_summary['samples']}"
    )
    print(f"Node feature set: {args.node_feature_set}")
    print(f"Node features: {dataset.node_features} ({first_graph.x.size(1)} columns)")
    print(f"Edge features: {dataset.edge_features} ({first_graph.edge_attr.size(1)} columns)")
    print(f"Loss history: {history_path}")
    print(f"Target splits ({split_manifest_status}): {split_manifest_path}")
    print(f"Test predictions: {test_predictions_path}")
    print(f"Reconstruction summary: {reconstruction_summary_path}")

    balancer = build_loss_balancer(args)
    if balancer is not None:
        share_text = ", ".join(f"{name}={share:.0%}" for name, share in balancer.shares.items())
        print(f"Loss weighting: normalised shares ({share_text})")
    else:
        weight_text = ", ".join(f"{name}={value:g}" for name, value in resolve_loss_weights(args).items())
        print(f"Loss weighting: fixed weights ({weight_text})")
    print(f"Checkpoint selected on: val_{args.checkpoint_metric}")

    best_val_metric = float("inf")
    with history_path.open("w", newline="", encoding="ascii") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=make_history_fieldnames())
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            print(f"Starting epoch {epoch:03d}...")
            train_metrics = run_epoch(
                model, train_loader, optimizer, device, args, train=True, balancer=balancer, scheduler=scheduler
            )
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, optimizer, device, args, train=False, balancer=balancer)
                test_metrics = ({name: float("nan") for name in LOSS_METRIC_NAMES}
                                if args.validation_only_during_training else
                                run_epoch(model, test_loader, optimizer, device, args, train=False, balancer=balancer))

            writer.writerow(
                flatten_history_row(
                    epoch,
                    {
                        "train": train_metrics,
                        "val": val_metrics,
                        "test": test_metrics,
                    },
                )
            )
            history_file.flush()

            current_val_metric = val_metrics[args.checkpoint_metric]
            message = (
                f"Epoch {epoch:03d} "
                f"train loss={train_metrics['loss']:.5f} "
                f"val loss={val_metrics['loss']:.5f} "
                f"test loss={test_metrics['loss']:.5f} "
                f"val target_mse={val_metrics['target_mse']:.5f}"
            )

            if current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "node_features": dataset.node_features,
                        "node_feature_set": args.node_feature_set,
                        "edge_features": dataset.edge_features,
                        "edge_recon_features": edge_recon_features,
                        "edge_recon_columns": recon_columns,
                        "edge_feature_transforms": edge_feature_transforms,
                        "edge_feature_stats": edge_feature_stats,
                        "in_node_feats": first_graph.x.size(1),
                        "esm_dim": first_graph.esm.size(1) if args.use_esm else 0,
                        "in_edge_feats": first_graph.edge_attr.size(1),
                        "best_loss": best_val_metric,
                        "best_metric": args.checkpoint_metric,
                        "best_metric_value": best_val_metric,
                        "best_epoch": epoch,
                        "target_splits": {
                            split_name: [str(path) for path in paths]
                            for split_name, paths in split_paths.items()
                        },
                    },
                    checkpoint_path,
                )
                message += " saved"

            print(message)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_prediction_rows = collect_prediction_rows(
        model,
        test_loader,
        device,
        args,
        node_feature_set=args.node_feature_set,
        node_features=dataset.node_features,
    )
    save_prediction_rows(test_predictions_path, test_prediction_rows)

    reconstruction_eval_args = SimpleNamespace(
        cluster=args.cluster,
        split_manifest=str(split_manifest_path),
        target_name=args.target_name,
        optional_node_features_dir=args.optional_node_features_dir,
        model_list_dir=args.model_list_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_start_method=args.worker_start_method,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        edge_threshold=args.eval_edge_threshold,
        negative_ratio=args.eval_negative_ratio,
        seed=args.seed,
        max_samples=None,
        strict_hdf5=args.strict_hdf5,
        log_every=args.eval_log_every,
        data=args.data,
    )
    summary_row, per_target_rows = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        model_label=output_dir.name,
        args=reconstruction_eval_args,
        device=device,
    )
    write_csv(reconstruction_summary_path, [summary_row])
    write_csv(reconstruction_by_target_path, per_target_rows)

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Reconstruction summary CSV: {reconstruction_summary_path}")
    print(f"Reconstruction by target CSV: {reconstruction_by_target_path}")


if __name__ == "__main__":
    main()
