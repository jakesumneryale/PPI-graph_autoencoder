from __future__ import annotations

import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from make_subset_dataset import (
    allocate_stratified_counts,
    classify_model,
    load_allowed_models,
    select_subset,
    write_subset,
)


class ClassifyModelTests(unittest.TestCase):
    def test_uniform_pattern_extracts_bin(self):
        self.assertEqual(classify_model("complex.12_7_3"), "uniform_bin_07")
        self.assertEqual(classify_model("complex.12_19_3_corrected"), "uniform_bin_19")

    def test_random_pattern(self):
        self.assertEqual(classify_model("complex.10042_5"), "random")

    def test_unparseable_name(self):
        self.assertEqual(classify_model("not_a_model_name"), "other")


class AllocateStratifiedCountsTests(unittest.TestCase):
    def test_sums_to_rounded_total(self):
        strata = {"a": list(range(50)), "b": list(range(30)), "c": list(range(20))}
        counts = allocate_stratified_counts(strata, 0.1)
        self.assertEqual(sum(counts.values()), 10)

    def test_never_exceeds_stratum_size(self):
        strata = {"tiny": ["x"], "big": list(range(1000))}
        counts = allocate_stratified_counts(strata, 0.9)
        self.assertLessEqual(counts["tiny"], 1)
        self.assertLessEqual(counts["big"], 1000)

    def test_zero_fraction_gives_at_least_one_when_data_present(self):
        counts = allocate_stratified_counts({"a": list(range(10))}, 0.001)
        self.assertGreaterEqual(sum(counts.values()), 1)


class SelectSubsetTests(unittest.TestCase):
    def test_stratifies_across_all_bins(self):
        names = [f"complex.{r}_{b}_{m}" for b in range(20) for r, m in [(1, 0), (2, 1)]]
        selected = select_subset(names, "targetA", fraction=0.5, seed=1)
        bins_present = {int(re.fullmatch(r"complex\.\d+_(\d+)_\d+", n).group(1)) for n in selected}
        self.assertEqual(bins_present, set(range(20)))

    def test_deterministic_given_seed(self):
        names = [f"complex.{i}_{i % 5}_0" for i in range(100)]
        a = select_subset(names, "t", 0.1, seed=42)
        b = select_subset(names, "t", 0.1, seed=42)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        names = [f"complex.{i}_{i % 5}_0" for i in range(100)]
        a = select_subset(names, "t", 0.1, seed=1)
        b = select_subset(names, "t", 0.1, seed=2)
        self.assertNotEqual(a, b)

    def test_different_targets_differ_under_same_seed(self):
        names = [f"complex.{i}_{i % 5}_0" for i in range(100)]
        a = select_subset(names, "targetA", 0.1, seed=1)
        b = select_subset(names, "targetB", 0.1, seed=1)
        self.assertNotEqual(a, b)


class WriteSubsetTests(unittest.TestCase):
    def test_copies_groups_and_payload_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src.hdf5"
            with h5py.File(source, "w") as h:
                h.attrs["data_path"] = "unit-test"
                for i in range(5):
                    g = h.create_group(f"complex.{i}_0_0")
                    g.create_dataset("node_features/aa_type", data=np.eye(3, dtype=np.float32))

            selected = ["complex.0_0_0", "complex.2_0_0"]
            destination = root / "out" / "src.hdf5"
            write_subset(source, destination, selected, fraction=0.4, seed=7)

            with h5py.File(destination, "r") as h:
                self.assertEqual(sorted(k for k in h if not k.startswith("__")), selected)
                self.assertEqual(h.attrs["data_path"], "unit-test")
                self.assertEqual(int(h.attrs["subset_model_count"]), 2)
                np.testing.assert_array_equal(
                    h["complex.0_0_0"]["node_features"]["aa_type"][()], np.eye(3, dtype=np.float32)
                )

    def test_source_file_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src.hdf5"
            with h5py.File(source, "w") as h:
                for i in range(3):
                    h.create_group(f"complex.{i}_0_0")
            before = source.stat().st_mtime_ns
            write_subset(source, root / "out.hdf5", ["complex.0_0_0"], 0.3, 1)
            self.assertEqual(source.stat().st_mtime_ns, before)
            with h5py.File(source, "r") as h:
                self.assertEqual(len(list(h.keys())), 3)


class LoadAllowedModelsTests(unittest.TestCase):
    def test_reads_one_name_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1acb.txt").write_text("complex.0_0_0\ncomplex.0_0_1\n\n")
            allowed = load_allowed_models(root, "1acb")
            self.assertEqual(allowed, {"complex.0_0_0", "complex.0_0_1"})

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_allowed_models(Path(tmp), "absent_target"))


if __name__ == "__main__":
    unittest.main()
