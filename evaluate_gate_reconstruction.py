"""Evaluate graph reconstruction quality for saved GATE checkpoints.

This script reloads one or more trained checkpoints, runs them on the saved
test split, and reports reconstruction metrics that are easier to interpret
than raw loss alone:

- amino-acid identity top-1 accuracy / misclassification rate
- binary node/edge feature accuracy
- C-alpha distance error in angstroms
- edge recovery / sampled non-edge rejection rates

Outputs are written as CSV files so results are easy to compare or plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from GATE_model import GraphAttentionAutoencoder
from protein_hdf5_dataset import (
    ProteinGraphHDF5Dataset,
    apply_cluster_path_defaults,
)


NODE_FEATURE_DIMS = {
    "aa_type": 20,
    "chain": 1,
    "interface_nodes": 1,
    "rsasa_i": 1,
    "drsasa": 1,
}
EDGE_FEATURE_DIMS = {
    "interface_edges": 1,
    "ca_dist": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GATE reconstruction quality on the test split.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "checkpoints/gate_all_features/gate_model.pt",
            "checkpoints/gate_no_aa_identity/gate_model.pt",
        ],
        help="One or more checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels for --checkpoints. Defaults to each checkpoint parent directory name.",
    )
    parser.add_argument("--cluster", action="store_true", help="Use the cluster default data directories.")
    parser.add_argument("--data", default=None, help="HDF5 file or directory.")
    parser.add_argument(
        "--split-manifest",
        default="checkpoints/gate_feature_compare_split.json",
        help="Optional split manifest. When present, overrides the checkpoint's embedded split.",
    )
    parser.add_argument("--target-name", default="DockQ")
    parser.add_argument(
        "--optional-node-features-dir",
        default=None,
        help="Directory containing optional rsasa/drsasa CSV files when a checkpoint used those features.",
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
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="How many sampled non-edges to score per true edge when estimating edge classification metrics.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick smoke tests.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print progress after this many graphs. Use 0 to disable.",
    )
    parser.add_argument("--output-dir", default="checkpoints/reconstruction_eval")
    parser.add_argument("--strict-hdf5", action="store_true", help="Fail instead of skipping unreadable HDF5 files.")
    args = parser.parse_args()
    apply_cluster_path_defaults(args)

    if args.labels is not None and len(args.labels) != len(args.checkpoints):
        raise ValueError("--labels must match the number of --checkpoints.")
    if args.negative_ratio <= 0:
        raise ValueError("--negative-ratio must be positive.")
    if not 0.0 <= args.edge_threshold <= 1.0:
        raise ValueError("--edge-threshold must be in [0, 1].")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_worker_start_method(args) -> str:
    if getattr(args, "worker_start_method", "auto") == "auto":
        return "spawn" if getattr(args, "cluster", False) else "default"
    return args.worker_start_method


def make_dataloader_kwargs(args, device: torch.device) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if device.type == "cuda":
        kwargs["pin_memory"] = True

    worker_start_method = resolve_worker_start_method(args)
    if args.num_workers > 0:
        if worker_start_method != "default":
            kwargs["multiprocessing_context"] = worker_start_method
        kwargs["persistent_workers"] = getattr(args, "persistent_workers", False)
        if getattr(args, "prefetch_factor", None) is not None:
            kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def load_split_manifest(manifest_path: Path) -> dict[str, list[str]]:
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    return {
        split_name: list(payload["splits"][split_name]["paths"])
        for split_name in ("train", "val", "test")
    }


def resolve_test_paths(checkpoint: dict, split_manifest: str | None) -> list[str]:
    if split_manifest:
        manifest_path = Path(split_manifest)
        if manifest_path.exists():
            return load_split_manifest(manifest_path)["test"]

    target_splits = checkpoint.get("target_splits")
    if target_splits and target_splits.get("test"):
        return list(target_splits["test"])
    raise ValueError("No test split found in the checkpoint and no usable --split-manifest was provided.")


def build_feature_slices(feature_names: Iterable[str], dim_lookup: dict[str, int]) -> dict[str, slice]:
    feature_slices: dict[str, slice] = {}
    start = 0
    for name in feature_names:
        if name not in dim_lookup:
            raise KeyError(f"Unsupported feature name: {name}")
        width = dim_lookup[name]
        feature_slices[name] = slice(start, start + width)
        start += width
    return feature_slices


def make_metric_accumulator() -> dict[str, float]:
    return {
        "graphs": 0.0,
        "nodes": 0.0,
        "edges": 0.0,
        "sampled_non_edges": 0.0,
        "node_sq_error_sum": 0.0,
        "node_value_count": 0.0,
        "edge_sq_error_sum": 0.0,
        "edge_value_count": 0.0,
        "aa_correct": 0.0,
        "aa_total": 0.0,
        "chain_correct": 0.0,
        "chain_total": 0.0,
        "interface_node_correct": 0.0,
        "interface_node_total": 0.0,
        "interface_edge_correct": 0.0,
        "interface_edge_total": 0.0,
        "rsasa_i_abs_error_sum": 0.0,
        "rsasa_i_sq_error_sum": 0.0,
        "rsasa_i_signed_error_sum": 0.0,
        "rsasa_i_total": 0.0,
        "drsasa_abs_error_sum": 0.0,
        "drsasa_sq_error_sum": 0.0,
        "drsasa_signed_error_sum": 0.0,
        "drsasa_total": 0.0,
        "ca_dist_abs_error_sum": 0.0,
        "ca_dist_sq_error_sum": 0.0,
        "ca_dist_signed_error_sum": 0.0,
        "ca_dist_total": 0.0,
        "edge_tp": 0.0,
        "edge_fp": 0.0,
        "edge_tn": 0.0,
        "edge_fn": 0.0,
    }


def canonicalize_positive_edges(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    edge_recon: torch.Tensor | None,
    edge_logits: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    src = edge_index[0]
    dst = edge_index[1]
    undirected_src = torch.minimum(src, dst)
    undirected_dst = torch.maximum(src, dst)
    pair_keys = undirected_src * num_nodes + undirected_dst
    unique_keys, inverse = torch.unique(pair_keys, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=unique_keys.numel()).to(torch.float32)

    unique_edge_index = torch.stack(
        [torch.div(unique_keys, num_nodes, rounding_mode="floor"), unique_keys % num_nodes],
        dim=0,
    ).long()

    def group_mean(values: torch.Tensor | None) -> torch.Tensor | None:
        if values is None:
            return None
        if values.ndim == 1:
            values = values[:, None]
        grouped = torch.zeros((unique_keys.numel(), values.size(1)), device=values.device, dtype=values.dtype)
        grouped.index_add_(0, inverse, values)
        grouped = grouped / counts.to(values.device).unsqueeze(1)
        return grouped

    grouped_edge_attr = group_mean(edge_attr)
    grouped_edge_recon = group_mean(edge_recon)
    grouped_edge_logits = group_mean(edge_logits)
    if grouped_edge_logits is not None:
        grouped_edge_logits = grouped_edge_logits.view(-1)

    return unique_edge_index, grouped_edge_attr, grouped_edge_recon, grouped_edge_logits


def sample_unique_negative_edges(
    num_nodes: int,
    positive_edge_index: torch.Tensor,
    num_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    if num_samples <= 0 or num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)

    positive_keys = set((positive_edge_index[0] * num_nodes + positive_edge_index[1]).tolist())
    max_pairs = (num_nodes * (num_nodes - 1)) // 2
    available_negatives = max_pairs - len(positive_keys)
    num_samples = min(num_samples, max(available_negatives, 0))
    if num_samples == 0:
        return torch.empty((2, 0), dtype=torch.long)

    negative_keys: set[int] = set()
    batch_size = max(64, num_samples * 4)
    while len(negative_keys) < num_samples:
        sampled_src = rng.integers(0, num_nodes, size=batch_size, endpoint=False)
        sampled_dst = rng.integers(0, num_nodes, size=batch_size, endpoint=False)
        valid_mask = sampled_src != sampled_dst
        sampled_src = sampled_src[valid_mask]
        sampled_dst = sampled_dst[valid_mask]
        undirected_src = np.minimum(sampled_src, sampled_dst)
        undirected_dst = np.maximum(sampled_src, sampled_dst)
        sampled_keys = undirected_src * num_nodes + undirected_dst
        for key in sampled_keys.tolist():
            if key in positive_keys or key in negative_keys:
                continue
            negative_keys.add(key)
            if len(negative_keys) >= num_samples:
                break

    negative_key_array = np.fromiter(negative_keys, dtype=np.int64)
    neg_src = negative_key_array // num_nodes
    neg_dst = negative_key_array % num_nodes
    return torch.from_numpy(np.vstack([neg_src, neg_dst])).long()


def score_undirected_edge_pairs(
    model: GraphAttentionAutoencoder,
    node_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((0,), dtype=node_embeddings.dtype, device=node_embeddings.device)
    forward_scores = model.decode_edge_logits(node_embeddings, edge_index)
    reverse_scores = model.decode_edge_logits(node_embeddings, edge_index.flip(0))
    return 0.5 * (forward_scores + reverse_scores)


def update_binary_metric(
    accumulator: dict[str, float],
    true_values: torch.Tensor,
    predicted_values: torch.Tensor,
    field_prefix: str,
) -> None:
    predicted_labels = (predicted_values >= 0.5).to(torch.float32)
    correct = (predicted_labels == true_values).sum().item()
    accumulator[f"{field_prefix}_correct"] += float(correct)
    accumulator[f"{field_prefix}_total"] += float(true_values.numel())


def update_feature_metrics(
    accumulator: dict[str, float],
    node_features: tuple[str, ...],
    edge_features: tuple[str, ...],
    node_feature_slices: dict[str, slice],
    edge_feature_slices: dict[str, slice],
    true_nodes: torch.Tensor,
    pred_nodes: torch.Tensor,
    true_edges: torch.Tensor,
    pred_edges: torch.Tensor,
) -> None:
    accumulator["node_sq_error_sum"] += float(torch.sum((pred_nodes - true_nodes) ** 2).item())
    accumulator["node_value_count"] += float(true_nodes.numel())
    if true_edges.numel() > 0:
        accumulator["edge_sq_error_sum"] += float(torch.sum((pred_edges - true_edges) ** 2).item())
        accumulator["edge_value_count"] += float(true_edges.numel())

    if "aa_type" in node_features:
        aa_slice = node_feature_slices["aa_type"]
        true_aa = true_nodes[:, aa_slice].argmax(dim=1)
        pred_aa = pred_nodes[:, aa_slice].argmax(dim=1)
        accumulator["aa_correct"] += float((pred_aa == true_aa).sum().item())
        accumulator["aa_total"] += float(true_aa.numel())

    if "chain" in node_features:
        chain_slice = node_feature_slices["chain"]
        update_binary_metric(
            accumulator,
            true_nodes[:, chain_slice].view(-1),
            pred_nodes[:, chain_slice].view(-1),
            "chain",
        )

    if "interface_nodes" in node_features:
        interface_node_slice = node_feature_slices["interface_nodes"]
        update_binary_metric(
            accumulator,
            true_nodes[:, interface_node_slice].view(-1),
            pred_nodes[:, interface_node_slice].view(-1),
            "interface_node",
        )

    for feature_name in ("rsasa_i", "drsasa"):
        if feature_name in node_features:
            feature_slice = node_feature_slices[feature_name]
            feature_error = pred_nodes[:, feature_slice].view(-1) - true_nodes[:, feature_slice].view(-1)
            accumulator[f"{feature_name}_abs_error_sum"] += float(torch.sum(torch.abs(feature_error)).item())
            accumulator[f"{feature_name}_sq_error_sum"] += float(torch.sum(feature_error**2).item())
            accumulator[f"{feature_name}_signed_error_sum"] += float(torch.sum(feature_error).item())
            accumulator[f"{feature_name}_total"] += float(feature_error.numel())

    if "interface_edges" in edge_features and true_edges.numel() > 0:
        interface_edge_slice = edge_feature_slices["interface_edges"]
        update_binary_metric(
            accumulator,
            true_edges[:, interface_edge_slice].view(-1),
            pred_edges[:, interface_edge_slice].view(-1),
            "interface_edge",
        )

    if "ca_dist" in edge_features and true_edges.numel() > 0:
        ca_dist_slice = edge_feature_slices["ca_dist"]
        distance_error = pred_edges[:, ca_dist_slice].view(-1) - true_edges[:, ca_dist_slice].view(-1)
        accumulator["ca_dist_abs_error_sum"] += float(torch.sum(torch.abs(distance_error)).item())
        accumulator["ca_dist_sq_error_sum"] += float(torch.sum(distance_error**2).item())
        accumulator["ca_dist_signed_error_sum"] += float(torch.sum(distance_error).item())
        accumulator["ca_dist_total"] += float(distance_error.numel())


def update_edge_classification_metrics(
    accumulator: dict[str, float],
    positive_probabilities: torch.Tensor,
    negative_probabilities: torch.Tensor,
    edge_threshold: float,
) -> None:
    positive_predictions = positive_probabilities >= edge_threshold
    accumulator["edge_tp"] += float(positive_predictions.sum().item())
    accumulator["edge_fn"] += float((~positive_predictions).sum().item())

    if negative_probabilities.numel() > 0:
        negative_predictions = negative_probabilities >= edge_threshold
        accumulator["edge_fp"] += float(negative_predictions.sum().item())
        accumulator["edge_tn"] += float((~negative_predictions).sum().item())


def graph_target_id(hdf5_path: str) -> str:
    return Path(hdf5_path).stem


def finalize_metrics(accumulator: dict[str, float], model_label: str) -> dict[str, object]:
    def safe_divide(numerator: float, denominator: float) -> float | None:
        if denominator == 0:
            return None
        return numerator / denominator

    precision = safe_divide(accumulator["edge_tp"], accumulator["edge_tp"] + accumulator["edge_fp"])
    recall = safe_divide(accumulator["edge_tp"], accumulator["edge_tp"] + accumulator["edge_fn"])
    accuracy = safe_divide(
        accumulator["edge_tp"] + accumulator["edge_tn"],
        accumulator["edge_tp"] + accumulator["edge_tn"] + accumulator["edge_fp"] + accumulator["edge_fn"],
    )
    specificity = safe_divide(accumulator["edge_tn"], accumulator["edge_tn"] + accumulator["edge_fp"])
    f1_score = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1_score = 2 * precision * recall / (precision + recall)

    row: dict[str, object] = {
        "model": model_label,
        "graphs": int(accumulator["graphs"]),
        "nodes": int(accumulator["nodes"]),
        "unique_edges": int(accumulator["edges"]),
        "sampled_non_edges": int(accumulator["sampled_non_edges"]),
        "node_feature_mse": safe_divide(accumulator["node_sq_error_sum"], accumulator["node_value_count"]),
        "edge_feature_mse": safe_divide(accumulator["edge_sq_error_sum"], accumulator["edge_value_count"]),
        "aa_identity_accuracy_pct": None,
        "aa_identity_misclassified_pct": None,
        "chain_accuracy_pct": None,
        "interface_node_accuracy_pct": None,
        "interface_edge_accuracy_pct": None,
        "rsasa_i_mae": None,
        "rsasa_i_rmse": None,
        "rsasa_i_bias": None,
        "drsasa_mae": None,
        "drsasa_rmse": None,
        "drsasa_bias": None,
        "ca_dist_mae_angstrom": None,
        "ca_dist_rmse_angstrom": None,
        "ca_dist_bias_angstrom": None,
        "edge_recall_pct": None if recall is None else 100.0 * recall,
        "edge_precision_pct": None if precision is None else 100.0 * precision,
        "edge_specificity_pct": None if specificity is None else 100.0 * specificity,
        "sampled_edge_accuracy_pct": None if accuracy is None else 100.0 * accuracy,
        "edge_f1_pct": None if f1_score is None else 100.0 * f1_score,
    }

    aa_accuracy = safe_divide(accumulator["aa_correct"], accumulator["aa_total"])
    if aa_accuracy is not None:
        row["aa_identity_accuracy_pct"] = 100.0 * aa_accuracy
        row["aa_identity_misclassified_pct"] = 100.0 * (1.0 - aa_accuracy)

    for prefix, field_name in (
        ("chain", "chain_accuracy_pct"),
        ("interface_node", "interface_node_accuracy_pct"),
        ("interface_edge", "interface_edge_accuracy_pct"),
    ):
        accuracy_value = safe_divide(accumulator[f"{prefix}_correct"], accumulator[f"{prefix}_total"])
        if accuracy_value is not None:
            row[field_name] = 100.0 * accuracy_value

    for feature_name in ("rsasa_i", "drsasa"):
        feature_total = accumulator[f"{feature_name}_total"]
        if feature_total > 0:
            row[f"{feature_name}_mae"] = accumulator[f"{feature_name}_abs_error_sum"] / feature_total
            row[f"{feature_name}_rmse"] = math.sqrt(accumulator[f"{feature_name}_sq_error_sum"] / feature_total)
            row[f"{feature_name}_bias"] = accumulator[f"{feature_name}_signed_error_sum"] / feature_total

    if accumulator["ca_dist_total"] > 0:
        row["ca_dist_mae_angstrom"] = accumulator["ca_dist_abs_error_sum"] / accumulator["ca_dist_total"]
        row["ca_dist_rmse_angstrom"] = math.sqrt(accumulator["ca_dist_sq_error_sum"] / accumulator["ca_dist_total"])
        row["ca_dist_bias_angstrom"] = accumulator["ca_dist_signed_error_sum"] / accumulator["ca_dist_total"]

    return row


def format_metric(value: object, precision: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}{suffix}"
    if isinstance(value, float):
        return f"{value:.{precision}f}{suffix}"
    return f"{value}{suffix}"


def evaluate_checkpoint(
    checkpoint_path: Path,
    model_label: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    node_features = tuple(checkpoint["node_features"])
    edge_features = tuple(checkpoint["edge_features"])
    test_paths = resolve_test_paths(checkpoint, args.split_manifest)

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
        "Building evaluation dataloader with "
        f"num_workers={args.num_workers}, worker_start_method={resolve_worker_start_method(args)}, "
        f"pin_memory={'yes' if device.type == 'cuda' else 'no'}..."
    )
    loader = DataLoader(dataset, **make_dataloader_kwargs(args, device))

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

    node_feature_slices = build_feature_slices(node_features, NODE_FEATURE_DIMS)
    edge_feature_slices = build_feature_slices(edge_features, EDGE_FEATURE_DIMS)
    global_metrics = make_metric_accumulator()
    metrics_by_target: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(args.seed)
    processed_graphs = 0

    print(
        f"Evaluating {model_label} on {len(dataset)} graph(s) from {len(test_paths)} test target file(s) "
        f"using node features {node_features} and edge features {edge_features}."
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            ptr = batch.ptr.detach().cpu()
            hdf5_paths = list(batch.hdf5_path)

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

                target_id = graph_target_id(hdf5_paths[graph_index])
                target_metrics = metrics_by_target.setdefault(target_id, make_metric_accumulator())

                for accumulator in (global_metrics, target_metrics):
                    accumulator["graphs"] += 1.0
                    accumulator["nodes"] += float(local_nodes.size(0))
                    accumulator["edges"] += float(unique_edge_index.size(1))
                    accumulator["sampled_non_edges"] += float(negative_edge_index.size(1))
                    update_feature_metrics(
                        accumulator,
                        node_features=node_features,
                        edge_features=edge_features,
                        node_feature_slices=node_feature_slices,
                        edge_feature_slices=edge_feature_slices,
                        true_nodes=local_nodes,
                        pred_nodes=local_node_recon,
                        true_edges=unique_edge_attr if unique_edge_attr is not None else local_nodes.new_empty((0, 0)),
                        pred_edges=unique_edge_recon if unique_edge_recon is not None else local_nodes.new_empty((0, 0)),
                    )
                    update_edge_classification_metrics(
                        accumulator,
                        positive_probabilities=positive_probabilities,
                        negative_probabilities=negative_probabilities,
                        edge_threshold=args.edge_threshold,
                    )

                processed_graphs += 1
                if args.log_every and processed_graphs % args.log_every == 0:
                    print(
                        f"  {model_label}: processed {processed_graphs}/{len(dataset)} graphs "
                        f"({100.0 * processed_graphs / max(len(dataset), 1):.1f}%)"
                    )

    summary_row = finalize_metrics(global_metrics, model_label=model_label)
    summary_row["checkpoint"] = str(checkpoint_path)
    summary_row["node_features"] = ",".join(node_features)
    summary_row["edge_features"] = ",".join(edge_features)
    summary_row["test_hdf5_files"] = len(test_paths)
    summary_row["skipped_hdf5_files"] = len(dataset.skipped_files)

    per_target_rows = []
    for target_id in sorted(metrics_by_target):
        row = finalize_metrics(metrics_by_target[target_id], model_label=model_label)
        row["target_id"] = target_id
        per_target_rows.append(row)

    return summary_row, per_target_rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with output_path.open("w", newline="", encoding="ascii") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_row: dict[str, object]) -> None:
    print(f"Model: {summary_row['model']}")
    print(
        "  "
        f"graphs={summary_row['graphs']} "
        f"nodes={summary_row['nodes']} "
        f"unique_edges={summary_row['unique_edges']} "
        f"sampled_non_edges={summary_row['sampled_non_edges']}"
    )
    print(
        "  "
        f"edge recall={format_metric(summary_row['edge_recall_pct'], precision=2, suffix='%')} | "
        f"edge precision={format_metric(summary_row['edge_precision_pct'], precision=2, suffix='%')} | "
        f"sampled edge accuracy={format_metric(summary_row['sampled_edge_accuracy_pct'], precision=2, suffix='%')}"
    )
    if summary_row["aa_identity_misclassified_pct"] is not None:
        print(
            "  "
            f"AA misclassified={format_metric(summary_row['aa_identity_misclassified_pct'], precision=2, suffix='%')} | "
            f"AA accuracy={format_metric(summary_row['aa_identity_accuracy_pct'], precision=2, suffix='%')}"
        )
    print(
        "  "
        f"chain accuracy={format_metric(summary_row['chain_accuracy_pct'], precision=2, suffix='%')} | "
        f"interface node accuracy={format_metric(summary_row['interface_node_accuracy_pct'], precision=2, suffix='%')} | "
        f"interface edge accuracy={format_metric(summary_row['interface_edge_accuracy_pct'], precision=2, suffix='%')}"
    )
    print(
        "  "
        f"CA distance MAE={format_metric(summary_row['ca_dist_mae_angstrom'], precision=3, suffix=' A')} | "
        f"CA distance RMSE={format_metric(summary_row['ca_dist_rmse_angstrom'], precision=3, suffix=' A')}"
    )
    if summary_row["rsasa_i_mae"] is not None or summary_row["drsasa_mae"] is not None:
        print(
            "  "
            f"rSASA_i MAE={format_metric(summary_row['rsasa_i_mae'], precision=4)} | "
            f"dRSASA MAE={format_metric(summary_row['drsasa_mae'], precision=4)}"
        )
    print(
        "  "
        f"node MSE={format_metric(summary_row['node_feature_mse'], precision=6)} | "
        f"edge MSE={format_metric(summary_row['edge_feature_mse'], precision=6)}"
    )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [Path(path) for path in args.checkpoints]
    labels = args.labels or [path.parent.name or path.stem for path in checkpoint_paths]

    summary_rows = []
    per_target_rows = []
    for checkpoint_path, label in zip(checkpoint_paths, labels, strict=True):
        summary_row, model_target_rows = evaluate_checkpoint(checkpoint_path, label, args, device)
        summary_rows.append(summary_row)
        per_target_rows.extend(model_target_rows)
        print_summary(summary_row)

    summary_path = output_dir / "reconstruction_summary.csv"
    per_target_path = output_dir / "reconstruction_by_target.csv"
    write_csv(summary_path, summary_rows)
    write_csv(per_target_path, per_target_rows)

    print(f"Summary CSV: {summary_path}")
    print(f"Per-target CSV: {per_target_path}")


if __name__ == "__main__":
    main()
