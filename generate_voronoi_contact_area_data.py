"""Standalone wrapper to compute residue-level Voronoi contact-area edge data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import signal
import time

import h5py
import numpy as np

from voronoi_edge_features.common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFERENCE_DIR,
    default_graph_data_path,
    resolve_target_graph_hdf5,
    target_name_from_dir,
    target_output_hdf5_path,
    target_summary_csv_path,
)
from voronoi_edge_features.contact_area import (
    align_contact_areas_to_graph_edges,
    build_node_metadata_table,
    compute_bounded_voronoi_tessellation,
    compute_residue_contact_area_table,
    load_protein_dataframe,
)
from voronoi_edge_features.model_reference import (
    build_target_model_references,
    load_target_model_references,
    write_target_model_references,
)


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")

_stop_requested = False


def _request_stop(signum, frame) -> None:  # noqa: ARG001
    global _stop_requested
    _stop_requested = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Given /full/path/to/<target_name>, compute per-model residue-level "
            "Voronoi contact areas and save them in an HDF5 file aligned to the graph node IDs."
        )
    )
    parser.add_argument("target_dir", help="Full path to the target directory, e.g. /full/path/to/<target_name>")
    parser.add_argument("--data", help="Directory containing target graph HDF5 files.")
    parser.add_argument(
        "--pdb-root",
        help=(
            "Directory that directly contains sampled_<target_name>/ (and, inside it, "
            "random_negatives/). Defaults to target_dir, i.e. sampled_<target_name>/ is "
            "expected to be nested inside target_dir. Pass this when sampled_<target_name>/ "
            "instead sits directly under a shared directory, independent of target_dir."
        ),
    )
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--probe-size", type=float, default=1.4)
    parser.add_argument("--feature-name", default="voronoi_contact_area")
    parser.add_argument(
        "--write-graph-feature",
        action="store_true",
        help="Also write the aligned edge feature directly into the source graph HDF5 file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output groups and existing graph edge-feature datasets.",
    )
    parser.add_argument("--max-models", type=int, help="Optional limit for quick debugging.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--cluster", action="store_true", help="Use cluster default graph-data paths.")
    return parser.parse_args()


def ensure_reference_file(target_name: str, graph_hdf5_path: Path, reference_dir: str | Path) -> Path:
    # Always rebuilt: it's just an HDF5 key listing plus string logic (cheap), and
    # reusing a stale file here would silently keep serving PDB paths computed by
    # whatever path-inference logic was active the first time this target was run.
    references = build_target_model_references(graph_hdf5_path, target_name)
    csv_path, _ = write_target_model_references(references, reference_dir, target_name)
    return csv_path


def _write_string_dataset(group: h5py.Group, name: str, values) -> None:
    values = np.asarray(values, dtype=object)
    group.create_dataset(name, data=values, dtype=STRING_DTYPE)


def _replace_dataset(group: h5py.Group, name: str, data, dtype=None) -> None:
    if name in group:
        del group[name]
    if dtype is None:
        group.create_dataset(name, data=data)
    else:
        group.create_dataset(name, data=data, dtype=dtype)


def write_model_output(
    output_group: h5py.Group,
    target_name: str,
    graph_group_name: str,
    pdb_path: Path,
    node_table,
    contact_area_table,
    graph_contacts,
    aligned_graph_contact_area,
    missing_mask,
) -> None:
    output_group.attrs["target_name"] = target_name
    output_group.attrs["graph_group_name"] = graph_group_name
    output_group.attrs["source_pdb_path"] = str(pdb_path)
    output_group.attrs["num_nodes"] = int(len(node_table))
    output_group.attrs["num_contact_pairs"] = int(len(contact_area_table))
    output_group.attrs["num_graph_edges"] = int(len(graph_contacts))
    output_group.attrs["num_missing_graph_edges"] = int(missing_mask.sum())

    _replace_dataset(output_group, "node_aa_id", node_table["aa_id"].to_numpy(dtype=np.int32))
    _replace_dataset(output_group, "node_aa_ind", node_table["aa_ind"].to_numpy(dtype=np.int32))
    _replace_dataset(output_group, "node_chain_id", node_table["chain_id"].to_numpy(dtype=np.int32))
    _write_string_dataset(output_group, "node_chain_name", node_table["chain_name"].astype(str).to_numpy())
    _write_string_dataset(output_group, "node_aa_name", node_table["aa_name"].astype(str).to_numpy())

    _replace_dataset(
        output_group,
        "contact_pairs",
        contact_area_table[["aa_id1", "aa_id2"]].to_numpy(dtype=np.int32),
    )
    _replace_dataset(
        output_group,
        "contact_area",
        contact_area_table["voronoi_contact_area"].to_numpy(dtype=np.float32).reshape(-1, 1),
    )
    _replace_dataset(
        output_group,
        "contact_atom_face_count",
        contact_area_table["atom_face_count"].to_numpy(dtype=np.int32).reshape(-1, 1),
    )
    _replace_dataset(output_group, "graph_contacts", np.asarray(graph_contacts, dtype=np.int32))
    _replace_dataset(output_group, "graph_contact_area", aligned_graph_contact_area.astype(np.float32))
    _replace_dataset(output_group, "graph_contact_missing_mask", missing_mask.astype(np.uint8))


def maybe_write_graph_feature(
    graph_hdf5_path: Path,
    graph_group_name: str,
    feature_name: str,
    aligned_graph_contact_area: np.ndarray,
    overwrite: bool,
) -> None:
    with h5py.File(graph_hdf5_path, "r+") as graph_handle:
        edge_group = graph_handle[graph_group_name]["edge_features"]
        if feature_name in edge_group and not overwrite:
            return
        _replace_dataset(edge_group, feature_name, aligned_graph_contact_area.astype(np.float32))


def main() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    args = parse_args()
    if args.data is None:
        args.data = default_graph_data_path(cluster=args.cluster)

    target_dir = Path(args.target_dir).resolve()
    pdb_root = Path(args.pdb_root).resolve() if args.pdb_root else target_dir
    target_name = target_name_from_dir(target_dir)
    graph_hdf5_path = resolve_target_graph_hdf5(args.data, target_name)
    reference_csv = ensure_reference_file(target_name, graph_hdf5_path, args.reference_dir)
    references = load_target_model_references(reference_csv)
    if args.max_models is not None:
        references = references[: args.max_models]

    output_hdf5_path = target_output_hdf5_path(args.output_dir, target_name)
    output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path = target_summary_csv_path(args.output_dir, target_name)

    print(f"Target: {target_name}")
    print(f"Graph HDF5: {graph_hdf5_path}")
    print(f"Reference file: {reference_csv}")
    print(f"Output HDF5: {output_hdf5_path}")
    print(f"Models to process: {len(references)}")

    summary_rows: list[dict[str, object]] = []

    with h5py.File(output_hdf5_path, "a") as output_handle:
        output_handle.attrs["target_name"] = target_name
        output_handle.attrs["graph_hdf5_path"] = str(graph_hdf5_path)
        output_handle.attrs["target_dir"] = str(target_dir)
        output_handle.attrs["feature_name"] = args.feature_name
        output_handle.attrs["probe_size"] = float(args.probe_size)

        for index, reference in enumerate(references, start=1):
            if _stop_requested:
                print(
                    f"  stop signal received; halting after {index - 1}/{len(references)} models. "
                    "Rerun the same command to resume from here."
                )
                break

            start_time = time.time()
            pdb_path = pdb_root / reference.relative_pdb_path
            row = {
                "target_name": target_name,
                "graph_group_name": reference.graph_group_name,
                "relative_pdb_path": reference.relative_pdb_path,
                "pdb_path": str(pdb_path),
                "location_type": reference.location_type,
                "status": "success",
                "message": "",
                "elapsed_seconds": 0.0,
                "num_nodes": 0,
                "num_contact_pairs": 0,
                "num_graph_edges": 0,
                "num_missing_graph_edges": 0,
            }

            try:
                if not pdb_path.exists():
                    raise FileNotFoundError(f"Model file not found: {pdb_path}")

                if reference.graph_group_name in output_handle:
                    if args.overwrite:
                        del output_handle[reference.graph_group_name]
                    else:
                        row["status"] = "skipped_exists"
                        row["message"] = "Output group already exists. Use --overwrite to recompute."
                        summary_rows.append(row)
                        continue

                with h5py.File(graph_hdf5_path, "r") as graph_handle:
                    if reference.graph_group_name not in graph_handle:
                        raise KeyError(
                            f"Graph group {reference.graph_group_name!r} was not found in {graph_hdf5_path.name}"
                        )
                    graph_group = graph_handle[reference.graph_group_name]
                    graph_contacts = graph_group["edge_features"]["contacts"][()]

                protein_df = load_protein_dataframe(pdb_path.name, pdb_path.parent)
                bounded_voronoi_tessellation = compute_bounded_voronoi_tessellation(protein_df, args.probe_size)
                node_table = build_node_metadata_table(protein_df)
                contact_area_table = compute_residue_contact_area_table(protein_df, bounded_voronoi_tessellation)
                aligned_graph_contact_area, missing_mask = align_contact_areas_to_graph_edges(
                    contact_area_table,
                    graph_contacts,
                )

                output_group = output_handle.create_group(reference.graph_group_name)
                write_model_output(
                    output_group=output_group,
                    target_name=target_name,
                    graph_group_name=reference.graph_group_name,
                    pdb_path=pdb_path,
                    node_table=node_table,
                    contact_area_table=contact_area_table,
                    graph_contacts=graph_contacts,
                    aligned_graph_contact_area=aligned_graph_contact_area,
                    missing_mask=missing_mask,
                )

                if args.write_graph_feature:
                    maybe_write_graph_feature(
                        graph_hdf5_path,
                        reference.graph_group_name,
                        args.feature_name,
                        aligned_graph_contact_area,
                        overwrite=args.overwrite,
                    )

                row["num_nodes"] = int(len(node_table))
                row["num_contact_pairs"] = int(len(contact_area_table))
                row["num_graph_edges"] = int(len(graph_contacts))
                row["num_missing_graph_edges"] = int(missing_mask.sum())
            except Exception as exc:  # noqa: BLE001
                row["status"] = "error"
                row["message"] = str(exc)
            finally:
                row["elapsed_seconds"] = round(time.time() - start_time, 3)
                summary_rows.append(row)

            if args.log_every and index % args.log_every == 0:
                print(f"  processed {index}/{len(references)} models")

    with summary_csv_path.open("w", newline="", encoding="utf-8") as summary_handle:
        writer = csv.DictWriter(
            summary_handle,
            fieldnames=(
                "target_name",
                "graph_group_name",
                "relative_pdb_path",
                "pdb_path",
                "location_type",
                "status",
                "message",
                "elapsed_seconds",
                "num_nodes",
                "num_contact_pairs",
                "num_graph_edges",
                "num_missing_graph_edges",
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    success_count = sum(1 for row in summary_rows if row["status"] == "success")
    error_count = sum(1 for row in summary_rows if row["status"] == "error")
    skipped_count = sum(1 for row in summary_rows if row["status"] == "skipped_exists")
    print(
        f"Finished {target_name}: {success_count} succeeded, "
        f"{error_count} failed, {skipped_count} skipped. Summary -> {summary_csv_path}"
    )


if __name__ == "__main__":
    main()
