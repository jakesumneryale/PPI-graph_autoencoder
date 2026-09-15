"""Tests for log1p/standardisation of edge features and input-only edge features."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from protein_hdf5_dataset import (
    EDGE_FEATURE_TRANSFORMS,
    ProteinGraphHDF5Dataset,
    apply_feature_transform,
    compute_edge_feature_stats,
    normalize_feature_block,
)


def write_graph_file(path: Path, num_graphs: int = 4, num_nodes: int = 6, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    contacts = np.array([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i < j], dtype=np.int32)
    with h5py.File(path, "w") as handle:
        for index in range(num_graphs):
            group = handle.create_group(f"complex.0_0_{index}")
            nodes = group.create_group("node_features")
            nodes.create_dataset("aa_type", data=rng.random((num_nodes, 20)).astype(np.float32))
            nodes.create_dataset("chain", data=rng.integers(0, 2, (num_nodes, 1)).astype(np.float32))
            edges = group.create_group("edge_features")
            edges.create_dataset("contacts", data=contacts)
            edges.create_dataset("ca_dist", data=rng.uniform(3, 12, (len(contacts), 1)).astype(np.float32))
            # Heavy-tailed, like real Voronoi contact areas.
            edges.create_dataset(
                "voronoi_contact_area",
                data=rng.exponential(8.0, (len(contacts), 1)).astype(np.float32),
            )
            edges.create_dataset(
                "voronoi_contact_missing",
                data=rng.integers(0, 2, (len(contacts), 1)).astype(np.float32),
            )
            scores = group.create_group("target_scores")
            scores.create_dataset("DockQ", data=np.float32(rng.random()))


class FeatureTransformTests(unittest.TestCase):
    def test_log1p_matches_numpy_and_keeps_zero_at_zero(self):
        values = np.array([[0.0], [1.0], [50.0]], dtype=np.float32)
        transformed = apply_feature_transform(values, "log1p")
        np.testing.assert_allclose(transformed, np.log1p(values), rtol=1e-6)
        self.assertEqual(transformed[0, 0], 0.0)

    def test_log1p_is_monotone(self):
        values = np.linspace(0, 200, 50, dtype=np.float32).reshape(-1, 1)
        transformed = apply_feature_transform(values, "log1p")
        self.assertTrue(np.all(np.diff(transformed.ravel()) > 0))

    def test_log1p_compresses_the_tail(self):
        values = np.array([[1.0], [100.0]], dtype=np.float32)
        raw_ratio = values[1, 0] / values[0, 0]
        transformed = apply_feature_transform(values, "log1p")
        self.assertLess(transformed[1, 0] / transformed[0, 0], raw_ratio / 10)

    def test_none_transform_is_identity(self):
        values = np.array([[2.0], [7.0]], dtype=np.float32)
        np.testing.assert_array_equal(apply_feature_transform(values, "none"), values)

    def test_negative_values_rejected_by_log1p(self):
        with self.assertRaises(ValueError):
            apply_feature_transform(np.array([[-1.0]], dtype=np.float32), "log1p")

    def test_unknown_transform_rejected(self):
        with self.assertRaises(ValueError):
            apply_feature_transform(np.array([[1.0]], dtype=np.float32), "sqrt")
        self.assertEqual(EDGE_FEATURE_TRANSFORMS, ("none", "log1p"))

    def test_standardisation_centres_and_scales(self):
        values = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        normalized = normalize_feature_block(values, "none", (2.0, 1.0))
        np.testing.assert_allclose(normalized.ravel(), [-1.0, 0.0, 1.0], rtol=1e-6)

    def test_zero_std_does_not_blow_up(self):
        values = np.array([[5.0], [5.0]], dtype=np.float32)
        normalized = normalize_feature_block(values, "none", (5.0, 0.0))
        self.assertTrue(np.all(np.isfinite(normalized)))


class DatasetTransformTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_graph_file(self.root / "1abc.hdf5")
        self.edge_features = ("ca_dist", "voronoi_contact_area", "voronoi_contact_missing")

    def tearDown(self):
        self._tmp.cleanup()

    def _dataset(self, **kwargs):
        return ProteinGraphHDF5Dataset(
            self.root,
            node_features=("aa_type", "chain"),
            edge_features=self.edge_features,
            require_target=True,
            **kwargs,
        )

    def test_untransformed_dataset_is_unchanged(self):
        data = self._dataset()[0]
        self.assertEqual(data.edge_attr.size(1), 3)
        self.assertGreater(float(data.edge_attr[:, 1].max()), 1.0)

    def test_log1p_applies_only_to_the_named_column(self):
        raw = self._dataset()[0].edge_attr
        transformed = self._dataset(
            edge_feature_transforms={"voronoi_contact_area": "log1p"}
        )[0].edge_attr
        np.testing.assert_allclose(raw[:, 0].numpy(), transformed[:, 0].numpy(), rtol=1e-6)
        np.testing.assert_allclose(raw[:, 2].numpy(), transformed[:, 2].numpy(), rtol=1e-6)
        np.testing.assert_allclose(
            np.log1p(raw[:, 1].numpy()), transformed[:, 1].numpy(), rtol=1e-5
        )

    def test_standardised_column_is_roughly_unit_scale(self):
        stats = compute_edge_feature_stats(
            self.root,
            ("voronoi_contact_area",),
            {"voronoi_contact_area": "log1p"},
            max_graphs=50,
        )
        dataset = self._dataset(
            edge_feature_transforms={"voronoi_contact_area": "log1p"},
            edge_feature_stats=stats,
        )
        column = dataset[0].edge_attr[:, 1].numpy()
        self.assertLess(abs(float(column.mean())), 1.5)
        self.assertLess(float(np.abs(column).max()), 8.0)

    def test_stats_are_computed_on_transformed_values(self):
        stats = compute_edge_feature_stats(
            self.root, ("voronoi_contact_area",), {"voronoi_contact_area": "log1p"}, max_graphs=50
        )
        raw_stats = compute_edge_feature_stats(
            self.root, ("voronoi_contact_area",), {}, max_graphs=50
        )
        self.assertLess(stats["voronoi_contact_area"][0], raw_stats["voronoi_contact_area"][0])

    def test_edge_feature_slices_match_declared_order(self):
        slices = self._dataset().edge_feature_slices()
        self.assertEqual(slices["ca_dist"], slice(0, 1))
        self.assertEqual(slices["voronoi_contact_area"], slice(1, 2))
        self.assertEqual(slices["voronoi_contact_missing"], slice(2, 3))

    def test_transform_for_unused_feature_is_rejected(self):
        with self.assertRaises(KeyError):
            self._dataset(edge_feature_transforms={"not_a_feature": "log1p"})

    def test_unsupported_transform_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            self._dataset(edge_feature_transforms={"ca_dist": "sqrt"})


if __name__ == "__main__":
    unittest.main()
