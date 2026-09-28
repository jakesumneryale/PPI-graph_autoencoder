"""Tests for GraphAttentionAutoencoder.encode(..., return_attention=True)."""
from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from GATE_model import GraphAttentionAutoencoder


def make_model(residual: bool = True, heads: int = 2) -> GraphAttentionAutoencoder:
    return GraphAttentionAutoencoder(
        in_node_feats=5, in_edge_feats=3, hidden_dim=8, latent_dim=4,
        gat_heads=heads, dropout=0.0, predict_target=True, residual_connections=residual,
    ).eval()


def make_batch(num_nodes: int = 8, num_edges: int = 14, in_edge_feats: int = 3):
    torch.manual_seed(0)
    data = Data(
        x=torch.randn(num_nodes, 5),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.rand(num_edges, in_edge_feats),
        y=torch.tensor([0.5]),
    )
    return next(iter(DataLoader([data], batch_size=1))), num_nodes


class ReturnAttentionTests(unittest.TestCase):
    def test_default_behavior_unchanged(self):
        model = make_model()
        batch, _ = make_batch()
        node_z, graph_z = model.encode(batch)
        self.assertEqual(node_z.dim(), 2)
        self.assertEqual(graph_z.dim(), 2)

    def test_one_attention_tuple_per_gat_layer(self):
        model = make_model(residual=True)
        batch, _ = make_batch()
        _, _, attention = model.encode(batch, return_attention=True)
        self.assertEqual(len(attention), model.num_gat_layers)

    def test_two_layer_nonresidual_model_has_two_attention_tuples(self):
        model = GraphAttentionAutoencoder(
            in_node_feats=5, in_edge_feats=3, hidden_dim=8, latent_dim=4,
            gat_heads=2, dropout=0.0, predict_target=True, residual_connections=False,
        ).eval()
        batch, _ = make_batch()
        _, _, attention = model.encode(batch, return_attention=True)
        self.assertEqual(len(attention), 2)

    def test_alpha_shape_matches_heads_and_edges(self):
        model = make_model(heads=3)
        batch, num_nodes = make_batch()
        _, _, attention = model.encode(batch, return_attention=True)
        for edge_index, alpha in attention:
            self.assertEqual(edge_index.shape[0], 2)
            self.assertEqual(alpha.shape[0], edge_index.shape[1])
            self.assertEqual(alpha.shape[1], 3)  # gat_heads

    def test_attention_sums_to_one_per_destination_node(self):
        model = make_model()
        batch, num_nodes = make_batch()
        _, _, attention = model.encode(batch, return_attention=True)
        for edge_index, alpha in attention:
            per_head_mean = alpha.mean(dim=-1)
            totals = torch.zeros(num_nodes).scatter_add_(0, edge_index[1], per_head_mean)
            # Every node with at least one incoming edge (self-loops guarantee this)
            # must have its incoming attention weights sum to 1 (softmax per destination).
            self.assertTrue(torch.allclose(totals, torch.ones(num_nodes), atol=1e-4))

    def test_requesting_attention_does_not_change_the_latents(self):
        model = make_model()
        batch, _ = make_batch()
        node_z_a, graph_z_a = model.encode(batch)
        node_z_b, graph_z_b, _ = model.encode(batch, return_attention=True)
        self.assertTrue(torch.allclose(node_z_a, node_z_b))
        self.assertTrue(torch.allclose(graph_z_a, graph_z_b))

    def test_requesting_attention_does_not_change_forward_output(self):
        model = make_model()
        batch, _ = make_batch()
        out = model(batch)
        node_z_b, graph_z_b, _ = model.encode(batch, return_attention=True)
        self.assertTrue(torch.allclose(out["node_embeddings"], node_z_b))
        self.assertTrue(torch.allclose(out["graph_embeddings"], graph_z_b))


if __name__ == "__main__":
    unittest.main()
