"""Commit computed Voronoi contact-area features into a target's source graph HDF5.

Reads whichever models have already been computed into the per-target checkpoint
HDF5 (written by generate_voronoi_contact_area_data.py) and writes the aligned
edge feature into the *source* graph HDF5 for that target. To avoid ever leaving
the source file partially written (e.g. if the process is killed mid-write), all
edits are made to a temporary copy in the same directory, which is only swapped
into place via an atomic rename once every write has completed and been flushed.

A target is only considered "done" (marker file written) once every model listed
in its reference CSV has a corresponding group in the checkpoint HDF5. If some
models are still missing (not yet computed, or failed), whatever has been
computed so far is still committed, but the marker is withheld so the caller
knows to retry.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np

from voronoi_edge_features.common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFERENCE_DIR,
    default_graph_data_path,
    reference_csv_path,
    resolve_target_graph_hdf5,
    target_output_hdf5_path,
    checkpoint_model_is_complete,
)
from voronoi_edge_features.model_reference import load_target_model_references


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_name", help="Target name, e.g. 1ay7")
    parser.add_argument("--data", help="Directory containing target graph HDF5 files.")
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--feature-name", default="voronoi_contact_area")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the edge feature dataset even if it is already present in the source file.",
    )
    parser.add_argument(
        "--marker-path",
        help="If given, touched only once every model for this target has been committed.",
    )
    parser.add_argument("--cluster", action="store_true", help="Use cluster default graph-data paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data is None:
        args.data = default_graph_data_path(cluster=args.cluster)

    target_name = args.target_name
    graph_hdf5_path = resolve_target_graph_hdf5(args.data, target_name)

    reference_csv = reference_csv_path(args.reference_dir, target_name)
    if not reference_csv.exists():
        print(f"No reference CSV found at {reference_csv}; run the compute step first.", file=sys.stderr)
        sys.exit(1)
    references = load_target_model_references(reference_csv)
    all_group_names = {reference.graph_group_name for reference in references}

    checkpoint_path = target_output_hdf5_path(args.output_dir, target_name)
    if not checkpoint_path.exists():
        print(f"No checkpoint HDF5 found at {checkpoint_path}; nothing to commit.", file=sys.stderr)
        sys.exit(1)

    with h5py.File(checkpoint_path, "r") as checkpoint_handle:
        computed_group_names = {
            name
            for name in (set(checkpoint_handle.keys()) & all_group_names)
            if checkpoint_model_is_complete(checkpoint_handle[name])
        }
        if not computed_group_names:
            print(f"No computed models found in {checkpoint_path}; nothing to commit.", file=sys.stderr)
            sys.exit(1)

        with h5py.File(graph_hdf5_path, "r") as graph_handle:
            already_present = {
                name
                for name in computed_group_names
                if args.feature_name in graph_handle[name]["edge_features"]
            }
        to_write = computed_group_names if args.overwrite else (computed_group_names - already_present)

        if not to_write:
            print(f"{target_name}: {len(already_present)} models already committed; nothing new to write.")
        else:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target_name}.",
                suffix=".hdf5.tmp",
                dir=str(graph_hdf5_path.parent),
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                shutil.copy2(graph_hdf5_path, tmp_path)
                with h5py.File(tmp_path, "r+") as tmp_handle:
                    for name in sorted(to_write):
                        edge_group = tmp_handle[name]["edge_features"]
                        feature_data = checkpoint_handle[name]["graph_contact_area"][()].astype(np.float32)
                        if args.feature_name in edge_group:
                            del edge_group[args.feature_name]
                        edge_group.create_dataset(args.feature_name, data=feature_data)
                    tmp_handle.flush()
                os.replace(tmp_path, graph_hdf5_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

            print(f"{target_name}: committed {len(to_write)} models into {graph_hdf5_path}")

    missing = all_group_names - computed_group_names
    if missing:
        print(f"{target_name}: {len(missing)}/{len(all_group_names)} models not yet computed; marker withheld.")
        sys.exit(3)

    if args.marker_path:
        Path(args.marker_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.marker_path).touch()
    print(f"{target_name}: all {len(all_group_names)} models committed.")


if __name__ == "__main__":
    main()
