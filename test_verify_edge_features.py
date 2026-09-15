"""Regression coverage for preflight selection and failure reporting."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py


class VerifyEdgeFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.lists = self.root / "lists"
        self.data.mkdir()
        self.lists.mkdir()
        self.selection = self.lists / "target.txt"
        self.selection.write_text("selected\tannotation\n\n")
        with h5py.File(self.data / "target.hdf5", "w") as handle:
            edges = handle.create_group("selected/edge_features")
            edges.create_dataset("voronoi_contact_area", data=[1.0])
            edges.create_dataset("voronoi_contact_missing", data=[0.0])
            handle.create_group("excluded/edge_features")
            handle.create_group("failed_model")

    def run_check(self, filtered=True):
        command = [
            sys.executable, str(Path(__file__).with_name("verify_edge_features.py")),
            "--data", str(self.data), "--edge-features",
            "voronoi_contact_area,voronoi_contact_missing",
        ]
        if filtered:
            command.extend(["--model-list-dir", str(self.lists)])
        return subprocess.run(command, capture_output=True, text=True)

    def test_excluded_incomplete_models_do_not_block_selected_models(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Checked 1 graph group(s)", result.stdout)
        self.assertEqual(self.run_check(filtered=False).returncode, 1)

    def test_selected_missing_feature_fails(self):
        with h5py.File(self.data / "target.hdf5", "r+") as handle:
            del handle["selected/edge_features/voronoi_contact_missing"]
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("voronoi_contact_missing missing in 1 group(s)", result.stderr)

    def test_selected_missing_edge_group_fails(self):
        self.selection.write_text("failed_model\n")
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("<no edge_features group>", result.stderr)

    def test_selected_absent_model_fails(self):
        self.selection.write_text("not_in_file\n")
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("selected models absent from HDF5: not_in_file", result.stderr)

    def test_missing_list_fails(self):
        self.selection.unlink()
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("target.txt", result.stderr)

    def test_empty_target_list_is_allowed(self):
        with h5py.File(self.data / "unused.hdf5", "w") as handle:
            handle.create_group("failed_model")
        (self.lists / "unused.txt").write_text("")
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_entire_selection_empty_fails(self):
        self.selection.write_text("")
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("No graph groups were checked", result.stderr)


if __name__ == "__main__":
    unittest.main()
