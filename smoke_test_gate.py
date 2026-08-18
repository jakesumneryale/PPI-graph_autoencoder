"""One-batch sanity check for the GATE model and HDF5 dataloader."""

from __future__ import annotations

import argparse

import torch
from torch_geometric.loader import DataLoader

from GATE_model import GraphAttentionAutoencoder
from protein_hdf5_dataset import (
    DEFAULT_EDGE_FEATURES,
    DEFAULT_NODE_FEATURES,
    ProteinGraphHDF5Dataset,
    apply_cluster_path_defaults,
)
from train_gate import (
    choose_device,
    compute_gate_loss,
    make_dataloader_kwargs,
    parse_feature_list,
    resolve_worker_start_method,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GATE forward/backward smoke test.")
    parser.add_argument("--cluster", action="store_true", help="Use the cluster default data directories.")
    parser.add_argument("--data", default=None, help="HDF5 file or directory.")
    parser.add_argument("--target-name", default="DockQ")
    parser.add_argument("--node-features", default=None, help="Comma-separated node feature names.")
    parser.add_argument("--edge-features", default=None, help="Comma-separated edge feature names.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--worker-start-method",
        choices=("auto", "default", "spawn", "fork", "forkserver"),
        default="auto",
        help="DataLoader worker start method. On cluster runs, auto uses spawn to avoid HDF5/fork hangs.",
    )
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--node-weight", type=float, default=1.0)
    parser.add_argument("--edge-attr-weight", type=float, default=1.0)
    parser.add_argument("--edge-presence-weight", type=float, default=0.1)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--strict-hdf5", action="store_true", help="Fail instead of skipping unreadable HDF5 files.")
    args = parser.parse_args()
    apply_cluster_path_defaults(args)

    device = choose_device(args.device)
    worker_start_method = resolve_worker_start_method(args)
    dataset = ProteinGraphHDF5Dataset(
        args.data,
        node_features=parse_feature_list(args.node_features, DEFAULT_NODE_FEATURES),
        edge_features=parse_feature_list(args.edge_features, DEFAULT_EDGE_FEATURES),
        target_name=args.target_name,
        require_target=False,
        max_samples=args.max_samples,
        skip_invalid_files=not args.strict_hdf5,
    )
    first_graph = dataset[0]
    print(
        "Building smoke-test dataloader with "
        f"num_workers={args.num_workers}, worker_start_method={worker_start_method}, "
        f"pin_memory={'yes' if device.type == 'cuda' else 'no'}..."
    )
    loader = DataLoader(dataset, **make_dataloader_kwargs(args, device, shuffle=False))

    model = GraphAttentionAutoencoder(
        in_node_feats=first_graph.x.size(1),
        in_edge_feats=first_graph.edge_attr.size(1),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        predict_target=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(loader)).to(device)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss, metrics = compute_gate_loss(model, batch, output, args)
    loss.backward()
    optimizer.step()

    print(f"Device: {device}")
    print(f"Data path: {args.data}")
    if dataset.skipped_files:
        print(f"Skipped unreadable HDF5 file(s): {len(dataset.skipped_files)}")
    print(f"Dataset samples checked: {len(dataset)}")
    print(f"Batch nodes={batch.num_nodes}, edges={batch.num_edges}, graphs={batch.num_graphs}")
    print(f"x={tuple(batch.x.shape)}, edge_attr={tuple(batch.edge_attr.shape)}")
    print(f"node_recon={tuple(output['node_recon'].shape)}")
    print(f"edge_recon={tuple(output['edge_recon'].shape)}")
    print(f"edge_logits={tuple(output['edge_logits'].shape)}")
    print(f"graph_embeddings={tuple(output['graph_embeddings'].shape)}")
    print(f"quality_pred={tuple(output['quality_pred'].shape)}")
    print("losses:", ", ".join(f"{name}={value:.5f}" for name, value in metrics.items()))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
