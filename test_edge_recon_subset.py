"""Tests for edge features that are encoder inputs but not reconstruction targets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from GATE_model import GraphAttentionAutoencoder
from protein_hdf5_dataset import ProteinGraphHDF5Dataset
from test_edge_feature_normalization import write_graph_file
from train_gate import (
    compute_gate_loss,
    edge_recon_column_indices,
    parse_feature_transforms,
    resolve_edge_recon_features,
)


def make_model(in_edge_feats: int, out_edge_feats: int | None = None):
    return GraphAttentionAutoencoder(
        in_node_feats=21,
        in_edge_feats=in_edge_feats,
        hidden_dim=8,
        latent_dim=4,
        gat_heads=2,
        dropout=0.0,
        predict_target=True,
        out_edge_feats=out_edge_feats,
    )


def make_batch(num_nodes: int = 6, num_edges: int = 10, in_edge_feats: int = 3):
    torch.manual_seed(0)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    data = Data(
        x=torch.randn(num_nodes, 21),
        edge_index=edge_index,
        edge_attr=torch.rand(num_edges, in_edge_feats) * 10.0,
        y=torch.tensor([0.5]),
    )
    return next(iter(DataLoader([data], batch_size=1)))


ARGS = SimpleNamespace(
    node_weight=1.0, edge_attr_weight=1.0, edge_presence_weight=0.1, target_weight=1.0
)


class EdgeReconSubsetTests(unittest.TestCase):
    def test_decoder_width_follows_out_edge_feats(self):
        model = make_model(in_edge_feats=3, out_edge_feats=2)
        batch = make_batch()
        output = model(batch)
        self.assertEqual(output["edge_recon"].size(1), 2)
        self.assertEqual(batch.edge_attr.size(1), 3)

    def test_default_reconstructs_every_input_column(self):
        model = make_model(in_edge_feats=3)
        self.assertEqual(model.out_edge_feats, 3)
        self.assertIsNone(model.edge_recon_index)
        self.assertEqual(model(make_batch())["edge_recon"].size(1), 3)

    def test_loss_compares_against_the_selected_columns_only(self):
        model = make_model(in_edge_feats=3, out_edge_feats=2)
        model.set_edge_recon_index([0, 2])
        batch = make_batch()
        output = model(batch)
        _, metrics = compute_gate_loss(model, batch, output, ARGS)

        expected = torch.nn.functional.mse_loss(
            output["edge_recon"], batch.edge_attr[:, [0, 2]].float()
        )
        self.assertAlmostEqual(metrics["edge_attr_mse"], float(expected.detach()), places=5)

    def test_excluded_column_cannot_change_the_loss(self):
        """A huge raw-scale column is exactly what used to dominate edge_attr_mse."""
        model = make_model(in_edge_feats=3, out_edge_feats=2)
        model.set_edge_recon_index([0, 2])
        batch = make_batch()
        output = model(batch)
        _, before = compute_gate_loss(model, batch, output, ARGS)

        batch.edge_attr[:, 1] *= 1000.0
        _, after = compute_gate_loss(model, batch, output, ARGS)

        # edge_presence_bce resamples negative edges on every call, so it is
        # excluded here; the other three terms must be bit-for-bit unchanged.
        for term in ("node_mse", "edge_attr_mse", "target_mse"):
            self.assertAlmostEqual(before[term], after[term], places=6, msg=term)

    def test_excluded_column_still_reaches_the_encoder(self):
        model = make_model(in_edge_feats=3, out_edge_feats=2).eval()
        model.set_edge_recon_index([0, 2])
        batch = make_batch()
        with torch.no_grad():
            baseline = model(batch)["quality_pred"].clone()
            batch.edge_attr[:, 1] += 5.0
            perturbed = model(batch)["quality_pred"]
        self.assertFalse(torch.allclose(baseline, perturbed, atol=1e-6))

    def test_recon_index_is_not_persisted_in_state_dict(self):
        model = make_model(in_edge_feats=3, out_edge_feats=2)
        model.set_edge_recon_index([0, 2])
        self.assertNotIn("edge_recon_index", model.state_dict())

    def test_index_length_must_match_decoder_width(self):
        model = make_model(in_edge_feats=3, out_edge_feats=2)
        with self.assertRaises(ValueError):
            model.set_edge_recon_index([0, 1, 2])

    def test_out_edge_feats_cannot_exceed_inputs(self):
        with self.assertRaises(ValueError):
            make_model(in_edge_feats=2, out_edge_feats=3)


class ArgumentResolutionTests(unittest.TestCase):
    EDGE_FEATURES = ("interface_edges", "ca_dist", "voronoi_contact_area")

    def test_default_keeps_all_features(self):
        args = SimpleNamespace(edge_recon_features=None)
        self.assertEqual(resolve_edge_recon_features(args, self.EDGE_FEATURES), self.EDGE_FEATURES)

    def test_subset_is_parsed(self):
        args = SimpleNamespace(edge_recon_features="interface_edges,ca_dist")
        self.assertEqual(
            resolve_edge_recon_features(args, self.EDGE_FEATURES), ("interface_edges", "ca_dist")
        )

    def test_unknown_feature_is_rejected(self):
        args = SimpleNamespace(edge_recon_features="interface_edges,nonsense")
        with self.assertRaises(ValueError):
            resolve_edge_recon_features(args, self.EDGE_FEATURES)

    def test_column_indices_skip_the_excluded_feature(self):
        slices = {
            "interface_edges": slice(0, 1),
            "ca_dist": slice(1, 2),
            "voronoi_contact_area": slice(2, 3),
        }
        self.assertEqual(
            edge_recon_column_indices(slices, ("interface_edges", "ca_dist")), [0, 1]
        )

    def test_transform_spec_parsing(self):
        self.assertEqual(
            parse_feature_transforms("voronoi_contact_area=log1p"),
            {"voronoi_contact_area": "log1p"},
        )
        self.assertEqual(parse_feature_transforms(None), {})
        self.assertEqual(parse_feature_transforms(""), {})
        with self.assertRaises(ValueError):
            parse_feature_transforms("voronoi_contact_area")


class DatasetIntegrationTests(unittest.TestCase):
    def test_real_dataset_columns_line_up_with_the_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_graph_file(root / "1abc.hdf5")
            edge_features = ("ca_dist", "voronoi_contact_area", "voronoi_contact_missing")
            dataset = ProteinGraphHDF5Dataset(
                root,
                node_features=("aa_type", "chain"),
                edge_features=edge_features,
                require_target=True,
                edge_feature_transforms={"voronoi_contact_area": "log1p"},
            )
            args = SimpleNamespace(edge_recon_features="ca_dist,voronoi_contact_missing")
            recon_features = resolve_edge_recon_features(args, edge_features)
            columns = edge_recon_column_indices(dataset.edge_feature_slices(), recon_features)
            self.assertEqual(columns, [0, 2])

            model = GraphAttentionAutoencoder(
                in_node_feats=dataset[0].x.size(1),
                in_edge_feats=dataset[0].edge_attr.size(1),
                hidden_dim=8,
                latent_dim=4,
                gat_heads=2,
                dropout=0.0,
                predict_target=True,
                out_edge_feats=len(columns),
            )
            model.set_edge_recon_index(columns)
            batch = next(iter(DataLoader([dataset[0]], batch_size=1)))
            output = model(batch)
            loss, metrics = compute_gate_loss(model, batch, output, ARGS)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(output["edge_recon"].size(1), 2)
            self.assertGreater(metrics["edge_attr_mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
