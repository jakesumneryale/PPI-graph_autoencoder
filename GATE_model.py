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
    ) -> None:
        super().__init__()
        if in_node_feats <= 0:
            raise ValueError("in_node_feats must be positive.")
        if in_edge_feats < 0:
            raise ValueError("in_edge_feats cannot be negative.")

        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        edge_dim = in_edge_feats or None
        self.gat1 = GATConv(
            in_channels=in_node_feats,
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
            nn.Linear(hidden_dim, in_edge_feats),
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

    def encode(self, data):
        """Encode a PyG ``Data`` or ``Batch`` object.

        Expected attributes:
            x: ``[total_nodes, in_node_feats]`` residue/node features.
            edge_index: ``[2, total_edges]`` contact edges.
            edge_attr: optional ``[total_edges, in_edge_feats]`` edge features.
            batch: optional node-to-graph assignment vector.
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = F.elu(h)

        node_z = self.node_projector(h)
        pooled = torch.cat(
            [global_mean_pool(node_z, batch), global_max_pool(node_z, batch)],
            dim=-1,
        )
        graph_z = self.graph_projector(pooled)
        return node_z, graph_z

    def decode_edge_features(self, node_z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Reconstruct edge attributes for the provided edge list."""
        if self.in_edge_feats == 0:
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
