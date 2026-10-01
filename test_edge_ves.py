from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from add_edge_ves import (
    FEATURE_NAME,
    check_failure_rate,
    load_ves_by_decoy,
    parse_v_es,
    process_file,
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


def write_ves_csv(path: Path, rows: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "Edge_Ves"])
        for decoy, values in rows.items():
            writer.writerow([decoy, list(values)])


class LoadVesByDecoyTests(unittest.TestCase):
    def test_reads_raw_strings_keyed_by_decoy(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "1abc_edge_ves.csv"
            write_ves_csv(csv_path, {"complex.0_0_0": [0.1, -0.2, 0.0]})
            by_decoy = load_ves_by_decoy(csv_path)
            self.assertEqual(set(by_decoy), {"complex.0_0_0"})
            parsed = parse_v_es(by_decoy["complex.0_0_0"], num_edges=3, source="test")
            np.testing.assert_allclose(parsed[:, 0], [0.1, -0.2, 0.0])


class ParseVEsTests(unittest.TestCase):
    def test_rejects_edge_count_mismatch(self):
        with self.assertRaises(ValueError):
            parse_v_es("[0.1, 0.2]", num_edges=3, source="test")

    def test_allows_negative_values(self):
        parsed = parse_v_es("[-1.5, 2.5]", num_edges=2, source="test")
        np.testing.assert_allclose(parsed[:, 0], [-1.5, 2.5])


class ProcessFileTests(unittest.TestCase):
    def test_writes_feature_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hdf5_path = root / "1abc.hdf5"
            with h5py.File(hdf5_path, "w") as handle:
                model = handle.create_group("complex.0_0_0")
                edges = model.create_group("edge_features")
                edges.create_dataset("contacts", data=np.array([[0, 1], [1, 2]]))

            write_ves_csv(root / "1abc_edge_ves.csv", {"complex.0_0_0": [0.3, -0.4]})

            first = process_file(hdf5_path, root, overwrite=False)
            self.assertEqual(first["written_models"], 1)
            self.assertEqual(first["failed_models"], 0)
            with h5py.File(hdf5_path, "r") as handle:
                values = handle["complex.0_0_0/edge_features"][FEATURE_NAME][()]
                np.testing.assert_allclose(values[:, 0], [0.3, -0.4])

            second = process_file(hdf5_path, root, overwrite=False)
            self.assertEqual(second["written_models"], 0)
            self.assertEqual(second["skipped_valid_models"], 1)

    def test_decoy_missing_from_csv_is_recorded_as_failure_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hdf5_path = root / "1abc.hdf5"
            with h5py.File(hdf5_path, "w") as handle:
                model = handle.create_group("complex.0_0_0")
                edges = model.create_group("edge_features")
                edges.create_dataset("contacts", data=np.array([[0, 1], [1, 2]]))
            write_ves_csv(root / "1abc_edge_ves.csv", {"complex.0_0_1": [0.3, -0.4]})

            result = process_file(hdf5_path, root, overwrite=False)
            self.assertEqual(result["failed_models"], 1)
            self.assertIn("complex.0_0_0", result["errors"])
            with h5py.File(hdf5_path, "r") as handle:
                self.assertNotIn(FEATURE_NAME, handle["complex.0_0_0/edge_features"])


if __name__ == "__main__":
    unittest.main()
