"""EGNN scalar encoder with equivariant coordinate updates and GATE decoders.

Adapted equations from Satorras et al., ICML 2021:
https://proceedings.mlr.press/v139/satorras21a.html
Coordinates stay separate from scalar features. Contact topology is unchanged.
"""
import torch
from torch import nn
from GATE_model import GraphAttentionAutoencoder


class EquivariantLayer(nn.Module):
    def __init__(self, width, edge_dim):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * width + edge_dim + 1, width),
                                     nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.coordinate = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                        nn.Linear(width, 1, bias=False), nn.Tanh())
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, pos, edge_index, edge_attr):
        src, dst = edge_index
        delta = pos[src] - pos[dst]
        squared = delta.square().sum(-1, keepdim=True)
        message = self.message(torch.cat((h[src], h[dst], torch.log1p(squared), edge_attr), -1))
        counts = torch.bincount(dst, minlength=len(h)).clamp_min(1).to(h.dtype).unsqueeze(-1)
        aggregated = torch.zeros_like(h).index_add_(0, dst, message) / counts
        # Bounded, distance-normalized updates avoid coordinate explosions.
        displacement = delta / (1 + (squared + 1e-8).sqrt()) * self.coordinate(message)
        pos = pos + torch.zeros_like(pos).index_add_(0, dst, displacement) / counts
        h = self.norm(h + self.update(torch.cat((h, aggregated), -1)))
        return h, pos


class EquivariantGraphAutoencoder(GraphAttentionAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.hidden_dim
        encoder_dim = self.in_node_feats + (self.esm_projector[1].out_features if self.esm_dim else 0)
        # Replace the attention encoder; keep exactly the same pooling and decoders.
        del self.gat1, self.gat2, self.extra_gat_layers, self.input_residual
        self.input_projection = nn.Linear(encoder_dim, width)
        self.equivariant_layers = nn.ModuleList(
            EquivariantLayer(width, self.in_edge_feats) for _ in range(self.num_gat_layers))
        self.node_projector = nn.Linear(width, self.latent_dim)

    def encode_geometry(self, data):
        pos = getattr(data, "pos", None)
        if pos is None or pos.shape != (len(data.x), 3) or not torch.isfinite(pos).all():
            raise ValueError("EGNN requires finite residue coordinates with shape [N, 3]")
        h = self.input_projection(self.encoder_inputs(data))
        pos = pos.float()
        for layer in self.equivariant_layers:
            h, pos = layer(h, pos, data.edge_index, data.edge_attr.float())
        return self.node_projector(h), pos

    def encode(self, data):
        node_z, _ = self.encode_geometry(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(len(node_z), dtype=torch.long, device=node_z.device)
        return node_z, self.graph_projector(self.pool_nodes(node_z, batch, data))


def build_graph_model(architecture="gat", **kwargs):
    if architecture == "gat":
        return GraphAttentionAutoencoder(**kwargs)
    if architecture == "egnn":
        return EquivariantGraphAutoencoder(**kwargs)
    raise ValueError(f"Unknown architecture: {architecture}")
