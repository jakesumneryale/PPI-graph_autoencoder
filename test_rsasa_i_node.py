from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from add_rsasa_i_node import (
    FEATURE_NAME,
    check_failure_rate,
    load_rsasa_i_node,
    process_file,
    run_lengths,
)


class CheckFailureRateTests(unittest.TestCase):
    def test_within_threshold_returns_rate_without_raising(self):
        rate = check_failure_rate(failures=5, total=1000, max_failure_rate=0.05, feature_name="x")
        self.assertAlmostEqual(rate, 0.005)

    def test_exceeding_threshold_raises(self):
        with self.assertRaises(SystemExit):
            check_failure_rate(failures=100, total=1000, max_failure_rate=0.05, feature_name="x")

    def test_zero_total_does_not_divide_by_zero(self):
        rate = check_failure_rate(failures=0, total=0, max_failure_rate=0.05, feature_name="x")
        self.assertEqual(rate, 0.0)


def write_rsasa_csv(path: Path, values: list[float], chains: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["residue_ind", "chain_id", "rSASA_i_complex"])
        for index, (value, chain) in enumerate(zip(values, chains), start=1):
            writer.writerow([index, chain, value])


class RunLengthsTests(unittest.TestCase):
    def test_groups_consecutive_equal_values(self):
        self.assertEqual(run_lengths(["E", "E", "E", "I", "I"]), [3, 2])

    def test_empty_input(self):
        self.assertEqual(run_lengths([]), [])


class LoadRsasaINodeTests(unittest.TestCase):
    def test_reads_values_in_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "complex.0_0_0.csv"
            write_rsasa_csv(csv_path, [0.1, 0.2, 0.3], ["E", "E", "I"])
            values = load_rsasa_i_node(csv_path, num_nodes=3, chain_reference=[0, 0, 1])
            np.testing.assert_allclose(values[:, 0], [0.1, 0.2, 0.3])

    def test_rejects_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "complex.0_0_0.csv"
            write_rsasa_csv(csv_path, [0.1, 0.2], ["E", "E"])
            with self.assertRaises(ValueError):
                load_rsasa_i_node(csv_path, num_nodes=3, chain_reference=[0, 0, 1])

    def test_rejects_chain_layout_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "complex.0_0_0.csv"
            write_rsasa_csv(csv_path, [0.1, 0.2, 0.3], ["E", "I", "I"])
            with self.assertRaises(ValueError):
                load_rsasa_i_node(csv_path, num_nodes=3, chain_reference=[0, 0, 1])


class ProcessFileTests(unittest.TestCase):
    def test_writes_feature_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hdf5_path = root / "1abc.hdf5"
            with h5py.File(hdf5_path, "w") as handle:
                model = handle.create_group("complex.0_0_0")
                nodes = model.create_group("node_features")
                nodes.create_dataset("chain", data=np.array([[0], [0], [1]]))

            rsasa_dir = root / "1abc_rSASA"
            write_rsasa_csv(rsasa_dir / "complex.0_0_0.csv", [0.5, 0.6, 0.7], ["E", "E", "I"])

            first = process_file(hdf5_path, root, overwrite=False)
            self.assertEqual(first["written_models"], 1)
            self.assertEqual(first["failed_models"], 0)
            with h5py.File(hdf5_path, "r") as handle:
                values = handle["complex.0_0_0/node_features"][FEATURE_NAME][()]
                np.testing.assert_allclose(values[:, 0], [0.5, 0.6, 0.7])

            second = process_file(hdf5_path, root, overwrite=False)
            self.assertEqual(second["written_models"], 0)
            self.assertEqual(second["skipped_valid_models"], 1)

    def test_missing_csv_is_recorded_as_failure_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hdf5_path = root / "1abc.hdf5"
            with h5py.File(hdf5_path, "w") as handle:
                model = handle.create_group("complex.0_0_0")
                nodes = model.create_group("node_features")
                nodes.create_dataset("chain", data=np.array([[0], [0], [1]]))
            (root / "1abc_rSASA").mkdir()

            result = process_file(hdf5_path, root, overwrite=False)
            self.assertEqual(result["failed_models"], 1)
            self.assertIn("complex.0_0_0", result["errors"])
            with h5py.File(hdf5_path, "r") as handle:
                self.assertNotIn(FEATURE_NAME, handle["complex.0_0_0/node_features"])


if __name__ == "__main__":
    unittest.main()
