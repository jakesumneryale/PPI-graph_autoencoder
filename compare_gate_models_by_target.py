"""Compare multiple GATE checkpoints on the shared test targets.

This script evaluates several saved checkpoints on the same held-out target
files and writes:

- one CSV per target containing graph-level rows for all models
- a summary CSV with one aggregated row per (model, target)

Each graph-level row includes reconstruction metrics plus true/predicted DockQ,
which makes the output convenient for downstream plotting and analysis.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from GATE_model import GraphAttentionAutoencoder
from evaluate_gate_reconstruction import (
    NODE_FEATURE_DIMS,
    EDGE_FEATURE_DIMS,
    build_feature_slices,
    canonicalize_positive_edges,
    choose_device,
    finalize_metrics,
    make_dataloader_kwargs,
    make_metric_accumulator,
    resolve_test_paths,
    sample_unique_negative_edges,
    score_undirected_edge_pairs,
    update_edge_classification_metrics,
    update_feature_metrics,
    write_csv,
)
from protein_hdf5_dataset import (
    ProteinGraphHDF5Dataset,
    apply_cluster_path_defaults,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple GATE checkpoints target-by-target.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "checkpoints/gate_all_features/gate_model.pt",
            "checkpoints/gate_no_aa_identity/gate_model.pt",
            "checkpoints/gate_with_rsasa_drsasa_20ep/gate_model.pt",
        ],
        help="Checkpoint paths to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for --checkpoints. Defaults to each checkpoint parent directory name.",
    )
    parser.add_argument("--cluster", action="store_true", help="Use the cluster default data directories.")
    parser.add_argument("--data", default=None, help="HDF5 file or directory.")
    parser.add_argument(
        "--split-manifest",
        default="checkpoints/gate_feature_compare_split.json",
        help="Optional split manifest. When present, uses that test-target list for every model.",
    )
    parser.add_argument("--target-name", default="DockQ")
    parser.add_argument(
        "--optional-node-features-dir",
        default=None,
        help="Directory containing optional rsasa/drsasa CSV files.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--worker-start-method",
        choices=("auto", "default", "spawn", "fork", "forkserver"),
        default="auto",
        help="DataLoader worker start method. On cluster runs, auto uses spawn to avoid HDF5/fork hangs.",
    )
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--limit-targets",
        type=int,
        default=None,
        help="Optional number of test target files to keep, for quick smoke tests.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional total graph limit per model, for quick smoke tests.",
    )
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--output-dir", default="checkpoints/model_target_compare")
    parser.add_argument("--strict-hdf5", action="store_true")
    args = parser.parse_args()
    apply_cluster_path_defaults(args)

    if args.labels is not None and len(args.labels) != len(args.checkpoints):
        raise ValueError("--labels must match the number of --checkpoints.")
    if args.negative_ratio <= 0:
        raise ValueError("--negative-ratio must be positive.")
    if not 0.0 <= args.edge_threshold <= 1.0:
        raise ValueError("--edge-threshold must be in [0, 1].")
    if args.limit_targets is not None and args.limit_targets <= 0:
        raise ValueError("--limit-targets must be positive when provided.")
    return args


def make_target_accumulator() -> dict[str, float]:
    accumulator = make_metric_accumulator()
    accumulator.update(
        {
            "dockq_true_sum": 0.0,
            "dockq_pred_sum": 0.0,
            "dockq_abs_error_sum": 0.0,
            "dockq_sq_error_sum": 0.0,
            "dockq_count": 0.0,
        }
    )
    return accumulator


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> GraphAttentionAutoencoder:
    checkpoint_args = checkpoint.get("args", {})
    model = GraphAttentionAutoencoder(
        in_node_feats=checkpoint["in_node_feats"],
        in_edge_feats=checkpoint["in_edge_feats"],
        hidden_dim=checkpoint_args["hidden_dim"],
        latent_dim=checkpoint_args["latent_dim"],
        gat_heads=checkpoint_args["gat_heads"],
        dropout=checkpoint_args["dropout"],
        predict_target=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def metric_row_for_graph(
    *,
    model_label: str,
    checkpoint_path: Path,
    node_features: tuple[str, ...],
    edge_features: tuple[str, ...],
    node_feature_slices: dict[str, slice],
    edge_feature_slices: dict[str, slice],
    true_nodes: torch.Tensor,
    pred_nodes: torch.Tensor,
    true_edges: torch.Tensor,
    pred_edges: torch.Tensor,
    positive_probabilities: torch.Tensor,
    negative_probabilities: torch.Tensor,
    edge_threshold: float,
    target_id: str,
    graph_name: str,
    hdf5_path: str,
    sample_index: int,
    true_dockq: float,
    predicted_dockq: float,
) -> tuple[dict[str, object], dict[str, float]]:
    accumulator = make_target_accumulator()
    accumulator["graphs"] += 1.0
    accumulator["nodes"] += float(true_nodes.size(0))
    accumulator["edges"] += float(true_edges.size(0))
    accumulator["sampled_non_edges"] += float(negative_probabilities.numel())
    update_feature_metrics(
        accumulator,
        node_features=node_features,
        edge_features=edge_features,
        node_feature_slices=node_feature_slices,
        edge_feature_slices=edge_feature_slices,
        true_nodes=true_nodes,
        pred_nodes=pred_nodes,
        true_edges=true_edges,
        pred_edges=pred_edges,
    )
    update_edge_classification_metrics(
        accumulator,
        positive_probabilities=positive_probabilities,
        negative_probabilities=negative_probabilities,
        edge_threshold=edge_threshold,
    )
    dockq_error = predicted_dockq - true_dockq
    accumulator["dockq_true_sum"] += true_dockq
    accumulator["dockq_pred_sum"] += predicted_dockq
    accumulator["dockq_abs_error_sum"] += abs(dockq_error)
    accumulator["dockq_sq_error_sum"] += dockq_error**2
    accumulator["dockq_count"] += 1.0

    row = finalize_metrics(accumulator, model_label=model_label)
    row.update(
        {
            "checkpoint": str(checkpoint_path),
            "target_id": target_id,
            "graph_name": graph_name,
            "sample_index": sample_index,
            "hdf5_path": hdf5_path,
            "node_features": ",".join(node_features),
            "edge_features": ",".join(edge_features),
            "true_dockq": true_dockq,
            "predicted_dockq": predicted_dockq,
            "dockq_abs_error": abs(dockq_error),
            "dockq_squared_error": dockq_error**2,
        }
    )
    return row, accumulator


def summarize_target_rows(
    *,
    accumulator: dict[str, float],
    model_label: str,
    checkpoint_path: Path,
    target_id: str,
    node_features: tuple[str, ...],
    edge_features: tuple[str, ...],
) -> dict[str, object]:
    row = finalize_metrics(accumulator, model_label=model_label)
    dockq_count = accumulator["dockq_count"]
    row.update(
        {
            "checkpoint": str(checkpoint_path),
            "target_id": target_id,
            "node_features": ",".join(node_features),
            "edge_features": ",".join(edge_features),
            "mean_true_dockq": None if dockq_count == 0 else accumulator["dockq_true_sum"] / dockq_count,
            "mean_predicted_dockq": None if dockq_count == 0 else accumulator["dockq_pred_sum"] / dockq_count,
            "dockq_mae": None if dockq_count == 0 else accumulator["dockq_abs_error_sum"] / dockq_count,
            "dockq_rmse": None if dockq_count == 0 else np.sqrt(accumulator["dockq_sq_error_sum"] / dockq_count),
        }
    )
    return row


def update_target_accumulator(target_accumulator: dict[str, float], graph_accumulator: dict[str, float]) -> None:
    for key, value in graph_accumulator.items():
        target_accumulator[key] = target_accumulator.get(key, 0.0) + value


def evaluate_model_by_target(
    checkpoint_path: Path,
    model_label: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    node_features = tuple(checkpoint["node_features"])
    edge_features = tuple(checkpoint["edge_features"])
    test_paths = resolve_test_paths(checkpoint, args.split_manifest)
    if args.limit_targets is not None:
        test_paths = test_paths[: args.limit_targets]

    dataset = ProteinGraphHDF5Dataset(
        test_paths,
        node_features=node_features,
        edge_features=edge_features,
        target_name=args.target_name,
        require_target=True,
        max_samples=args.max_samples,
        skip_invalid_files=not args.strict_hdf5,
        optional_node_features_dir=checkpoint_args.get("optional_node_features_dir", args.optional_node_features_dir),
    )
    print(
        "Building comparison dataloader with "
        f"num_workers={args.num_workers}, worker_start_method={args.worker_start_method if args.worker_start_method != 'auto' else ('spawn' if args.cluster else 'default')}, "
        f"pin_memory={'yes' if device.type == 'cuda' else 'no'}..."
    )
    loader = DataLoader(dataset, **make_dataloader_kwargs(args, device))
    model = build_model_from_checkpoint(checkpoint, device=device)
    node_feature_slices = build_feature_slices(node_features, NODE_FEATURE_DIMS)
    edge_feature_slices = build_feature_slices(edge_features, EDGE_FEATURE_DIMS)
    rng = np.random.default_rng(args.seed)
    rows_by_target: dict[str, list[dict[str, object]]] = {}
    accumulators_by_target: dict[str, dict[str, float]] = {}
    processed_graphs = 0

    print(
        f"Comparing {model_label} on {len(dataset)} graph(s) from {len(test_paths)} test target file(s) "
        f"using node features {node_features}."
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            ptr = batch.ptr.detach().cpu()
            hdf5_paths = list(batch.hdf5_path)
            graph_names = list(batch.graph_name)
            sample_indices = batch.sample_index.view(-1).detach().cpu().tolist()
            true_dockq_values = batch.y.float().view(-1).detach().cpu().tolist()
            predicted_dockq_values = output["quality_pred"].detach().cpu().tolist()

            for graph_index in range(len(hdf5_paths)):
                node_start = int(ptr[graph_index].item())
                node_end = int(ptr[graph_index + 1].item())
                node_mask = slice(node_start, node_end)
                edge_mask = (batch.edge_index[0] >= node_start) & (batch.edge_index[0] < node_end)

                local_nodes = batch.x[node_mask]
                local_node_recon = output["node_recon"][node_mask]
                local_node_embeddings = output["node_embeddings"][node_mask]
                local_edge_index = (batch.edge_index[:, edge_mask] - node_start).long()
                local_edge_attr = batch.edge_attr[edge_mask] if getattr(batch, "edge_attr", None) is not None else None
                local_edge_recon = output["edge_recon"][edge_mask] if local_edge_attr is not None else None
                local_edge_logits = output["edge_logits"][edge_mask]

                (
                    unique_edge_index,
                    unique_edge_attr,
                    unique_edge_recon,
                    unique_edge_logits,
                ) = canonicalize_positive_edges(
                    num_nodes=node_end - node_start,
                    edge_index=local_edge_index,
                    edge_attr=local_edge_attr,
                    edge_recon=local_edge_recon,
                    edge_logits=local_edge_logits,
                )

                positive_probabilities = torch.sigmoid(unique_edge_logits)
                num_negative_edges = int(round(unique_edge_index.size(1) * args.negative_ratio))
                negative_edge_index = sample_unique_negative_edges(
                    num_nodes=node_end - node_start,
                    positive_edge_index=unique_edge_index.cpu(),
                    num_samples=num_negative_edges,
                    rng=rng,
                )
                negative_probabilities = torch.empty((0,), device=device)
                if negative_edge_index.numel() > 0:
                    negative_probabilities = torch.sigmoid(
                        score_undirected_edge_pairs(
                            model,
                            local_node_embeddings,
                            negative_edge_index.to(device),
                        )
                    )

                target_id = Path(hdf5_paths[graph_index]).stem
                row, graph_accumulator = metric_row_for_graph(
                    model_label=model_label,
                    checkpoint_path=checkpoint_path,
                    node_features=node_features,
                    edge_features=edge_features,
                    node_feature_slices=node_feature_slices,
                    edge_feature_slices=edge_feature_slices,
                    true_nodes=local_nodes,
                    pred_nodes=local_node_recon,
                    true_edges=unique_edge_attr if unique_edge_attr is not None else local_nodes.new_empty((0, 0)),
                    pred_edges=unique_edge_recon if unique_edge_recon is not None else local_nodes.new_empty((0, 0)),
                    positive_probabilities=positive_probabilities,
                    negative_probabilities=negative_probabilities,
                    edge_threshold=args.edge_threshold,
                    target_id=target_id,
                    graph_name=graph_names[graph_index],
                    hdf5_path=hdf5_paths[graph_index],
                    sample_index=int(sample_indices[graph_index]),
                    true_dockq=float(true_dockq_values[graph_index]),
                    predicted_dockq=float(predicted_dockq_values[graph_index]),
                )

                rows_by_target.setdefault(target_id, []).append(row)
                target_accumulator = accumulators_by_target.setdefault(target_id, make_target_accumulator())
                update_target_accumulator(target_accumulator, graph_accumulator)

                processed_graphs += 1
                if args.log_every and processed_graphs % args.log_every == 0:
                    print(
                        f"  {model_label}: processed {processed_graphs}/{len(dataset)} graphs "
                        f"({100.0 * processed_graphs / max(len(dataset), 1):.1f}%)"
                    )

    summary_rows = []
    for target_id in sorted(accumulators_by_target):
        summary_rows.append(
            summarize_target_rows(
                accumulator=accumulators_by_target[target_id],
                model_label=model_label,
                checkpoint_path=checkpoint_path,
                target_id=target_id,
                node_features=node_features,
                edge_features=edge_features,
            )
        )
    return rows_by_target, summary_rows


def sanitize_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    labels = args.labels or [path.parent.name or path.stem for path in checkpoint_paths]

    output_dir = Path(args.output_dir)
    target_output_dir = output_dir / "targets"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_output_dir.mkdir(parents=True, exist_ok=True)

    all_rows_by_target: dict[str, list[dict[str, object]]] = {}
    all_summary_rows: list[dict[str, object]] = []
    for checkpoint_path, label in zip(checkpoint_paths, labels, strict=True):
        model_rows_by_target, model_summary_rows = evaluate_model_by_target(
            checkpoint_path=checkpoint_path,
            model_label=label,
            args=args,
            device=device,
        )
        for target_id, rows in model_rows_by_target.items():
            all_rows_by_target.setdefault(target_id, []).extend(rows)
        all_summary_rows.extend(model_summary_rows)

    for target_id, rows in sorted(all_rows_by_target.items()):
        target_path = target_output_dir / f"{sanitize_filename(target_id)}.csv"
        rows.sort(key=lambda row: (str(row["model"]), str(row["graph_name"])))
        write_csv(target_path, rows)
        print(f"Wrote target CSV: {target_path}")

    summary_path = output_dir / "model_target_summary.csv"
    all_summary_rows.sort(key=lambda row: (str(row["target_id"]), str(row["model"])))
    write_csv(summary_path, all_summary_rows)
    print(f"Wrote summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
