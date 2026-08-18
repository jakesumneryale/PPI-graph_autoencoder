"""Train the GATE model on PPI graph HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

try:
    from torch_geometric.utils import batched_negative_sampling
except ImportError:  # pragma: no cover - compatibility fallback for older PyG.
    batched_negative_sampling = None

from GATE_model import GraphAttentionAutoencoder
from protein_hdf5_dataset import (
    DEFAULT_EDGE_FEATURES,
    DEFAULT_NODE_FEATURES,
    ProteinGraphHDF5Dataset,
    default_hdf5_data_path,
)


def parse_feature_list(raw: str | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return defaults
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


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


def compute_gate_loss(model, batch, output, args) -> tuple[torch.Tensor, dict[str, float]]:
    losses: dict[str, torch.Tensor] = {}

    losses["node_mse"] = F.mse_loss(output["node_recon"], batch.x.float())

    if getattr(batch, "edge_attr", None) is not None and batch.edge_attr.numel() > 0:
        losses["edge_attr_mse"] = F.mse_loss(output["edge_recon"], batch.edge_attr.float())
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

    total = (
        args.node_weight * losses["node_mse"]
        + args.edge_attr_weight * losses["edge_attr_mse"]
        + args.edge_presence_weight * losses["edge_presence_bce"]
        + args.target_weight * losses["target_mse"]
    )
    metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
    metrics["loss"] = float(total.detach().cpu())
    return total, metrics


def run_epoch(model, loader, optimizer, device, args, train: bool) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            output = model(batch)
            loss, metrics = compute_gate_loss(model, batch, output, args)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        num_batches += 1

    return {name: value / max(num_batches, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GATE on PPI graph HDF5 files.")
    parser.add_argument("--data", default=default_hdf5_data_path(), help="HDF5 file or directory.")
    parser.add_argument("--target-name", default="DockQ")
    parser.add_argument("--node-features", default=None, help="Comma-separated node feature names.")
    parser.add_argument("--edge-features", default=None, help="Comma-separated edge feature names.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--node-weight", type=float, default=1.0)
    parser.add_argument("--edge-attr-weight", type=float, default=1.0)
    parser.add_argument("--edge-presence-weight", type=float, default=0.1)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    dataset = ProteinGraphHDF5Dataset(
        args.data,
        node_features=parse_feature_list(args.node_features, DEFAULT_NODE_FEATURES),
        edge_features=parse_feature_list(args.edge_features, DEFAULT_EDGE_FEATURES),
        target_name=args.target_name,
        require_target=True,
        max_samples=args.max_samples,
    )
    first_graph = dataset[0]
    model = GraphAttentionAutoencoder(
        in_node_feats=first_graph.x.size(1),
        in_edge_feats=first_graph.edge_attr.size(1),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        predict_target=True,
    ).to(device)

    train_len = max(1, int(len(dataset) * args.train_fraction))
    train_len = min(train_len, len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(args.seed)
    if val_len > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_len, val_len],
            generator=generator,
        )
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Device: {device}")
    print(f"Samples: total={len(dataset)}, train={train_len}, val={val_len}")
    print(f"Node features: {dataset.node_features} ({first_graph.x.size(1)} columns)")
    print(f"Edge features: {dataset.edge_features} ({first_graph.edge_attr.size(1)} columns)")

    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "gate_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True)
        message = f"Epoch {epoch:03d} train loss={train_metrics['loss']:.5f}"

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, optimizer, device, args, train=False)
            message += f" val loss={val_metrics['loss']:.5f}"
            current_val_loss = val_metrics["loss"]
        else:
            current_val_loss = train_metrics["loss"]

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "node_features": dataset.node_features,
                    "edge_features": dataset.edge_features,
                    "in_node_feats": first_graph.x.size(1),
                    "in_edge_feats": first_graph.edge_attr.size(1),
                    "best_loss": best_val_loss,
                },
                checkpoint_path,
            )
            message += " saved"

        print(message)

    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
