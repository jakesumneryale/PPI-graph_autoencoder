"""Interpretability analysis for a trained GATE checkpoint (dissertation TODO,
Sec. 4.4.3): attention-weight distributions on interface versus intra-chain
edges, and whether high-attention edges coincide with high Voronoi contact
area or high burial.

NOT included: correlation against |Delta phi_r| (interaction electrostatic
potential). Delta phi_r is only computed for the 84 bound native structures
(Fig. 4.3's standalone APBS pipeline), not for any of the ~150k training/test
decoy models this checkpoint was trained and evaluated on, so there is
nothing to correlate against yet -- same blocker as the "+electrostatics"
rung of the feature ladder. See Table 4.1 (V_ae: planned).

Usage:
    python gate_run/analyze_attention_weights.py \
        --checkpoint gate_run/feature_ladder_10pct/gat4_residual_voronoi_seed7/gate_model.pt \
        --output-dir gate_run/feature_ladder_10pct/attention_analysis \
        --split test --max-graphs 400
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for GATE_model et al.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, spearmanr
from torch_geometric.loader import DataLoader

from EGNN_model import build_graph_model
from GATE_model import GraphAttentionAutoencoder
from protein_hdf5_dataset import DEFAULT_OPTIONAL_NODE_FEATURES_DIR, ProteinGraphHDF5Dataset
from train_gate import load_target_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Graph HDF5 directory. Default: the --data path recorded in the checkpoint.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-graphs", type=int, default=400, help="Cap for runtime; 0 = no cap.")
    parser.add_argument("--layer", type=int, default=-1, help="Which GAT layer's attention to analyze (-1 = last).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def edge_lookup(edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict[tuple[int, int], int]:
    src, dst = edge_index
    return {(int(s), int(d)): i for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist()))}


def collect_edge_records(model: GraphAttentionAutoencoder, loader: DataLoader, layer_index: int,
                         interface_col: int, contact_area_col: int | None, rsasa_col: int | None,
                         device: str) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, _, attention = model.encode(batch, return_attention=True)
            attn_edge_index, alpha = attention[layer_index]
            alpha_mean = alpha.mean(dim=-1)  # average over attention heads

            lookup = edge_lookup(batch.edge_index, batch.edge_attr)
            attn_src, attn_dst = attn_edge_index
            for i in range(attn_edge_index.size(1)):
                s, d = int(attn_src[i]), int(attn_dst[i])
                if s == d:
                    continue  # GATConv's added self-loop, not a real contact
                original_index = lookup.get((s, d))
                if original_index is None:
                    continue  # defensive; should not happen for non-self-loop edges
                edge_features = batch.edge_attr[original_index]
                is_interface = bool(edge_features[interface_col].item() > 0.5)
                contact_area = float(edge_features[contact_area_col].item()) if contact_area_col is not None else np.nan
                burial = np.nan
                if rsasa_col is not None:
                    burial = float((batch.x[s, rsasa_col] + batch.x[d, rsasa_col]).item() / 2.0)
                rows.append({
                    "attention": float(alpha_mean[i].item()),
                    "is_interface": is_interface,
                    "contact_area": contact_area,
                    "burial": burial,
                    "graph_name": getattr(batch, "graph_name", ["?"])[0] if hasattr(batch, "graph_name") else "?",
                })
    return pd.DataFrame(rows)


def thesis_style() -> None:
    mpl.rcParams.update({
        "text.usetex": shutil.which("latex") is not None or Path("/Library/TeX/texbin/latex").exists(),
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "savefig.dpi": 300,
    })


def make_figure(df: pd.DataFrame, output_path: Path) -> None:
    thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)

    intra = df.loc[~df.is_interface, "attention"]
    interf = df.loc[df.is_interface, "attention"]
    axes[0].boxplot([intra, interf], tick_labels=["Intra-chain", "Interface"], showfliers=False, widths=0.6)
    axes[0].set_ylabel("Attention weight")
    axes[0].set_title("(a) By edge type")

    have_area = df.dropna(subset=["contact_area"])
    if len(have_area) > 10:
        axes[1].scatter(have_area["contact_area"], have_area["attention"], s=4, alpha=0.15, color="black",
                        rasterized=True)
        rho = spearmanr(have_area["contact_area"], have_area["attention"])[0]
        # edge_attr stores the standardized log1p(area) fed to the network, not
        # raw A^2 -- label it as such rather than implying physical units here.
        axes[1].set_xlabel("Voronoi contact area (standardized log1p)")
        axes[1].set_title(rf"(b) vs. contact area ($\rho_s={rho:.2f}$)")
    else:
        axes[1].text(0.5, 0.5, "no contact-area data", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_ylabel("Attention weight")

    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    node_features = tuple(checkpoint["node_features"])
    edge_features = tuple(checkpoint["edge_features"])
    architecture = checkpoint_args.get("architecture", "gat")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Architecture: {architecture}")
    print(f"Node features: {node_features}")
    print(f"Edge features: {edge_features}")
    if architecture != "gat":
        raise SystemExit(
            f"This checkpoint is architecture={architecture!r}; attention weights are only "
            "defined for the GAT encoder (architecture='gat'). EquivariantGraphAutoencoder's "
            "encode() has no attention to report."
        )
    if "interface_edges" not in edge_features:
        raise SystemExit("This checkpoint has no 'interface_edges' feature; cannot label edges by type.")

    interface_col = edge_features.index("interface_edges")
    contact_area_col = edge_features.index("voronoi_contact_area") if "voronoi_contact_area" in edge_features else None
    rsasa_col = node_features.index("rsasa_i") if "rsasa_i" in node_features else None
    if contact_area_col is None:
        print("NOTE: no voronoi_contact_area in this checkpoint's edge features; skipping contact-area correlation.")
    if rsasa_col is None:
        print("NOTE: no rsasa_i in this checkpoint's node features; skipping burial correlation.")

    dataset = ProteinGraphHDF5Dataset(
        args.data_dir or checkpoint_args["data"],
        node_features=node_features,
        edge_features=edge_features,
        target_name=checkpoint_args.get("target_name", "DockQ"),
        require_target=True,
        optional_node_features_dir=checkpoint_args.get("optional_node_features_dir", DEFAULT_OPTIONAL_NODE_FEATURES_DIR),
        model_list_dir=checkpoint_args.get("model_list_dir"),
        edge_feature_transforms=checkpoint.get("edge_feature_transforms") or {},
        edge_feature_stats=checkpoint.get("edge_feature_stats") or {},
        use_esm=checkpoint_args.get("use_esm", False),
        require_pos=False,  # guarded to architecture == "gat" above
        esm_sidecar_dir=checkpoint_args.get("esm_sidecar_dir"),
    )
    split_indices, _ = load_target_split_manifest(Path(checkpoint_args["split_manifest"]), dataset.samples) \
        if checkpoint_args.get("split_manifest") else (None, None)
    if split_indices is None:
        raise SystemExit("Checkpoint's args do not record a --split-manifest path; cannot recover the split.")
    indices = split_indices[args.split]
    if args.max_graphs and len(indices) > args.max_graphs:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(indices, size=args.max_graphs, replace=False).tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    print(f"Analyzing {len(indices)} {args.split} graphs.")

    edge_recon_features = tuple(checkpoint.get("edge_recon_features") or edge_features)
    edge_feature_slices = dataset.edge_feature_slices()
    recon_columns = [
        i for name in edge_recon_features for i in range(edge_feature_slices[name].start, edge_feature_slices[name].stop)
    ]

    model = build_graph_model(
        architecture="gat",
        pooling=checkpoint_args.get("pooling", "all"),
        esm_dim=checkpoint.get("esm_dim", 0),
        esm_projection_dim=checkpoint_args.get("esm_projection_dim", 64),
        esm_scale=checkpoint_args.get("esm_scale", 1.0),
        esm_dropout=checkpoint_args.get("esm_dropout", 0.0),
        esm_gate=checkpoint_args.get("esm_gate", False),
        in_node_feats=checkpoint["in_node_feats"],
        in_edge_feats=checkpoint["in_edge_feats"],
        hidden_dim=checkpoint_args["hidden_dim"],
        latent_dim=checkpoint_args["latent_dim"],
        gat_heads=checkpoint_args["gat_heads"],
        dropout=checkpoint_args["dropout"],
        residual_connections=checkpoint_args.get("residual_connections", False),
        predict_target=True,
        out_edge_feats=len(recon_columns),
    ).to(args.device)
    if edge_recon_features != edge_features:
        model.set_edge_recon_index(recon_columns)
    model.load_state_dict(checkpoint["model_state_dict"])

    df = collect_edge_records(model, loader, args.layer, interface_col, contact_area_col, rsasa_col, args.device)
    df.to_csv(args.output_dir / "attention_edge_records.csv", index=False)
    print(f"Collected {len(df):,} scored edges from {df.graph_name.nunique()} graphs.")

    intra = df.loc[~df.is_interface, "attention"]
    interf = df.loc[df.is_interface, "attention"]
    u_stat, p_value = mannwhitneyu(interf, intra, alternative="two-sided")
    summary_lines = [
        f"Edges scored: {len(df):,} ({df.is_interface.sum():,} interface, {(~df.is_interface).sum():,} intra-chain)",
        f"Mean attention, interface edges:    {interf.mean():.5f} (median {interf.median():.5f})",
        f"Mean attention, intra-chain edges:  {intra.mean():.5f} (median {intra.median():.5f})",
        f"Mann-Whitney U test (interface vs intra-chain): p = {p_value:.3g}",
    ]
    have_area = df.dropna(subset=["contact_area"])
    if len(have_area) > 10:
        rho, p = spearmanr(have_area["contact_area"], have_area["attention"])
        summary_lines.append(f"Spearman(attention, contact_area): rho={rho:+.3f}, p={p:.3g}, n={len(have_area)}")
    have_burial = df.dropna(subset=["burial"])
    if len(have_burial) > 10:
        rho, p = spearmanr(have_burial["burial"], have_burial["attention"])
        summary_lines.append(f"Spearman(attention, mean-endpoint burial rSASA_i): rho={rho:+.3f}, p={p:.3g}, n={len(have_burial)}")
    summary_lines.append(
        "NOT computed: correlation against |Delta phi_r| -- not available for decoy models (see module docstring)."
    )
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (args.output_dir / "summary.txt").write_text(summary + "\n")

    make_figure(df, args.output_dir / "attention_weight_distributions.pdf")
    print(f"\nWrote {args.output_dir / 'attention_weight_distributions.pdf'}")
    print(f"Wrote {args.output_dir / 'attention_edge_records.csv'}")
    print(f"Wrote {args.output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
