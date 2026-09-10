from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from audit_voronoi_dataset import AuditRow, audit_target, select_subset, write_subset


def row(model: str) -> AuditRow:
    parts = model.split(".", 1)[1].split("_")
    return AuditRow("1abc", model, True, "uniform" if len(parts) == 3 else "random", "", "ok", 2, "missing_file")


class AuditVoronoiDatasetTests(unittest.TestCase):
    def test_stratified_tenth_preserves_full_bins(self):
        rows = []
        for quality_bin in range(20):
            rows.extend(row(f"complex.0_{quality_bin}_{model}") for model in range(50))
        rows.extend(row(f"complex.{model}_0") for model in range(100))

        selected = select_subset(rows, fraction=0.1, seed=7)
        self.assertEqual(len(selected), 110)
        for quality_bin in range(20):
            prefix = f"complex.0_{quality_bin}_"
            self.assertEqual(sum(name.startswith(prefix) for name in selected), 5)
        self.assertEqual(sum(len(name.split("_") ) == 2 for name in selected), 10)
        self.assertEqual(selected, select_subset(rows, fraction=0.1, seed=7))

    def test_audit_rejects_missing_and_malformed_features_and_copies_selection(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "1abc.hdf5"
            with h5py.File(source, "w") as handle:
                handle.attrs["example"] = "preserved"
                for name, feature in (
                    ("complex.0_0_0", np.array([[1.0], [2.0]], dtype=np.float32)),
                    ("complex.0_1_0", None),
                    ("complex.0_2_0", np.array([1.0, 2.0], dtype=np.float32)),
                ):
                    group = handle.create_group(name)
                    edges = group.create_group("edge_features")
                    edges.create_dataset("contacts", data=np.array([[0, 1], [1, 2]]))
                    if feature is not None:
                        edges.create_dataset("voronoi_contact_area", data=feature)

            rows = audit_target(source, root / "checkpoints", "voronoi_contact_area")
            self.assertEqual([item.usable for item in rows], [True, False, False])
            self.assertEqual(rows[1].reason, "missing voronoi_contact_area")
            self.assertIn("invalid feature shape", rows[2].reason)

            destination = root / "subset" / "1abc.hdf5"
            write_subset(source, destination, ["complex.0_0_0"])
            with h5py.File(destination, "r") as handle:
                self.assertEqual(list(handle.keys()), ["complex.0_0_0"])
                self.assertEqual(handle.attrs["example"], "preserved")


if __name__ == "__main__":
    unittest.main()
