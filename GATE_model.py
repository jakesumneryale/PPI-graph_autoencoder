"""Graph attention autoencoder models for PPI contact graphs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool


class GraphAttentionAutoencoder(nn.Module):
    """Variable-size graph attention autoencoder for PyG ``Data`` batches.

    The HDF5 graphs in this project contain a variable number of residues and
    contacts per decoy, so reconstruction heads operate on node and edge
    embeddings instead of decoding to one fixed-size adjacency matrix.
    """

    def __init__(
        self,
        in_node_feats: int,
        in_edge_feats: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        gat_heads: int = 4,
        dropout: float = 0.1,
        predict_target: bool = True,
        residual_connections: bool = False,
        num_gat_layers: int | None = None,
        out_edge_feats: int | None = None,
        pooling: str = "all",
        esm_dim: int = 0,
        esm_projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if in_node_feats <= 0:
            raise ValueError("in_node_feats must be positive.")
        if in_edge_feats < 0:
            raise ValueError("in_edge_feats cannot be negative.")
        if num_gat_layers is None:
            num_gat_layers = 4 if residual_connections else 2
        if num_gat_layers < 2:
            raise ValueError("num_gat_layers must be at least 2.")
        if not residual_connections and num_gat_layers != 2:
            raise ValueError("More than two GAT layers require residual_connections=True.")

        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        # Edge features can be encoder inputs without being reconstruction targets:
        # Voronoi contact area informs attention, but regressing it back dominated
        # the loss, so out_edge_feats may be narrower than in_edge_feats.
        self.out_edge_feats = in_edge_feats if out_edge_feats is None else out_edge_feats
        if self.out_edge_feats < 0:
            raise ValueError("out_edge_feats cannot be negative.")
        if self.out_edge_feats > in_edge_feats:
            raise ValueError("out_edge_feats cannot exceed in_edge_feats.")
        # Non-persistent: keeps state_dict compatible with checkpoints trained
        # before input-only edge features existed.
        self.register_buffer("edge_recon_index", None, persistent=False)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.residual_connections = residual_connections
        self.num_gat_layers = num_gat_layers
        if pooling not in ("all", "interface"):
            raise ValueError("pooling must be all or interface")
        self.pooling = pooling
        self.esm_dim = esm_dim
        self.esm_projector = (nn.Sequential(nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, esm_projection_dim), nn.SiLU()) if esm_dim else None)
        encoder_dim = in_node_feats + (esm_projection_dim if esm_dim else 0)

        edge_dim = in_edge_feats or None
        self.gat1 = GATConv(
            in_channels=encoder_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim * gat_heads,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.extra_gat_layers = nn.ModuleList(
            GATConv(
                in_channels=hidden_dim * gat_heads,
                out_channels=hidden_dim,
                heads=gat_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
            )
            for _ in range(num_gat_layers - 2)
        )
        self.input_residual = (
            nn.Linear(encoder_dim, hidden_dim * gat_heads, bias=False)
            if residual_connections
            else None
        )
        self.node_projector = nn.Linear(hidden_dim * gat_heads, latent_dim)

        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_node_feats),
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_edge_feats),
        )
        self.link_decoder = nn.Sequential(
            nn.Linear(4 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.graph_projector = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.quality_head = (
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if predict_target
            else None
        )

    def set_edge_recon_index(self, column_indices) -> None:
        """Select which ``edge_attr`` columns the decoder is trained to reproduce.

        Pass ``None`` to reconstruct every input column (the default).
        """
        if column_indices is None:
            self.edge_recon_index = None
            return
        index = torch.as_tensor(column_indices, dtype=torch.long)
        if index.numel() != self.out_edge_feats:
            raise ValueError(
                f"edge_recon_index has {index.numel()} entries but out_edge_feats is {self.out_edge_feats}."
            )
        self.edge_recon_index = index

    def select_edge_recon_targets(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Reduce ground-truth ``edge_attr`` to the reconstructed columns."""
        if self.edge_recon_index is None:
            return edge_attr
        return edge_attr.index_select(1, self.edge_recon_index.to(edge_attr.device))

    def encode(self, data):
        """Encode a PyG ``Data`` or ``Batch`` object.

        Expected attributes:
            x: ``[total_nodes, in_node_feats]`` residue/node features.
            edge_index: ``[2, total_edges]`` contact edges.
            edge_attr: optional ``[total_edges, in_edge_feats]`` edge features.
            batch: optional node-to-graph assignment vector.
        """
        x = self.encoder_inputs(data)
        edge_index = data.edge_index.long()
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        if self.residual_connections:
            h = h + self.input_residual(x)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        gat_layers = (self.gat2, *self.extra_gat_layers)
        for layer_index, layer in enumerate(gat_layers):
            residual = h
            h = layer(h, edge_index, edge_attr=edge_attr)
            if self.residual_connections:
                h = h + residual
            h = F.elu(h)
            if layer_index < len(gat_layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        node_z = self.node_projector(h)
        pooled = self.pool_nodes(node_z, batch, data)
        graph_z = self.graph_projector(pooled)
        return node_z, graph_z

    def encoder_inputs(self, data):
        x = data.x.float()
        if self.esm_projector is not None:
            esm = getattr(data, "esm", None)
            if esm is None or esm.shape != (x.size(0), self.esm_dim):
                raise ValueError("Missing or incorrectly shaped per-residue ESM embeddings")
            x = torch.cat((x, self.esm_projector(esm.float())), dim=-1)
        return x

    def pool_nodes(self, node_z, batch, data):
        size = int(batch.max()) + 1
        if self.pooling == "interface":
            mask = getattr(data, "interface_mask", None)
            if mask is None or mask.numel() != node_z.size(0):
                raise ValueError("Interface pooling requires one interface_mask value per node")
            mask = mask.view(-1).bool()
            counts = torch.bincount(batch[mask], minlength=size)
            if (counts == 0).any():
                raise ValueError("Interface pooling encountered a graph with no interface nodes")
            node_z, batch = node_z[mask], batch[mask]
        return torch.cat((global_mean_pool(node_z, batch, size=size),
                          global_max_pool(node_z, batch, size=size)), dim=-1)

    def decode_edge_features(self, node_z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Reconstruct edge attributes for the provided edge list."""
        if self.out_edge_feats == 0:
            return node_z.new_empty((edge_index.size(1), 0))
        src, dst = edge_index
        edge_inputs = torch.cat([node_z[src], node_z[dst]], dim=-1)
        return self.edge_decoder(edge_inputs)

    def decode_edge_logits(self, node_z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Score whether each pair in ``edge_index`` should be connected."""
        src, dst = edge_index
        src_z = node_z[src]
        dst_z = node_z[dst]
        pair_features = torch.cat(
            [src_z, dst_z, torch.abs(src_z - dst_z), src_z * dst_z],
            dim=-1,
        )
        return self.link_decoder(pair_features).view(-1)

    def decode(self, node_z: torch.Tensor, graph_z: torch.Tensor, data):
        edge_index = data.edge_index.long()
        node_recon = self.node_decoder(node_z)
        edge_recon = self.decode_edge_features(node_z, edge_index)
        edge_logits = self.decode_edge_logits(node_z, edge_index)
        quality_pred = self.quality_head(graph_z).view(-1) if self.quality_head else None
        return {
            "node_recon": node_recon,
            "edge_recon": edge_recon,
            "edge_logits": edge_logits,
            "quality_pred": quality_pred,
        }

    def forward(self, data):
        node_z, graph_z = self.encode(data)
        decoded = self.decode(node_z, graph_z, data)
        return {
            **decoded,
            "node_embeddings": node_z,
            "graph_embeddings": graph_z,
        }


class PredictiveModel(nn.Module):
    """Small scalar regressor for graph-level latent vectors."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(z))
        return self.fc2(x).view(-1)


if __name__ == "__main__":
    from torch_geometric.data import Data

    num_nodes = 5
    num_edges = 8
    in_node_feats = 22
    in_edge_feats = 2

    x = torch.randn((num_nodes, in_node_feats))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn((num_edges, in_edge_feats))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    model = GraphAttentionAutoencoder(
        in_node_feats=in_node_feats,
        in_edge_feats=in_edge_feats,
        hidden_dim=32,
        latent_dim=16,
    )
    output = model(data)
    print("node_recon", tuple(output["node_recon"].shape))
    print("edge_recon", tuple(output["edge_recon"].shape))
    print("edge_logits", tuple(output["edge_logits"].shape))
    print("quality_pred", tuple(output["quality_pred"].shape))
    print("graph_embeddings", tuple(output["graph_embeddings"].shape))
