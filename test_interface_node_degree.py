from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from add_interface_node_degree import (
    FEATURE_NAME,
    calculate_interface_node_degree,
    process_file,
)


class InterfaceNodeDegreeTests(unittest.TestCase):
    def test_unique_undirected_interface_neighbors(self):
        interface_nodes = np.array([[0], [1], [1], [0]])
        contacts = np.array(
            [
                [0, 1],
                [1, 0],  # Reversed duplicate must not add another neighbor.
                [0, 2],
                [1, 2],
                [2, 3],
                [3, 3],  # Self edge is ignored.
            ]
        )
        observed = calculate_interface_node_degree(interface_nodes, contacts)
        np.testing.assert_array_equal(observed[:, 0], [2, 1, 1, 1])

    def test_one_based_contacts_and_resumable_hdf5_write(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "1abc.hdf5"
            with h5py.File(path, "w") as handle:
                model = handle.create_group("complex.0_0_0")
                nodes = model.create_group("node_features")
                edges = model.create_group("edge_features")
                nodes.create_dataset("interface_nodes", data=np.array([[1], [0], [1]]))
                edges.create_dataset("contacts", data=np.array([[1, 2], [2, 3]]))

            first = process_file(path, overwrite=False)
            self.assertEqual(first["written_models"], 1)
            self.assertEqual(first["failed_models"], 0)
            with h5py.File(path, "r") as handle:
                values = handle["complex.0_0_0/node_features"][FEATURE_NAME][()]
                np.testing.assert_array_equal(values[:, 0], [0, 2, 0])

            second = process_file(path, overwrite=False)
            self.assertEqual(second["written_models"], 0)
            self.assertEqual(second["skipped_valid_models"], 1)


if __name__ == "__main__":
    unittest.main()
