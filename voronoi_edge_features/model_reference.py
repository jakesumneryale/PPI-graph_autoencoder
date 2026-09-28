"""Helpers for mapping graph HDF5 keys to target-model PDB paths."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import h5py

from voronoi_edge_features.common import (
    DEFAULT_NAME_MAP_DIR,
    ModelReference,
    infer_relative_pdb_path,
    reference_csv_path,
    reference_txt_path,
)


def load_name_map(name_map_dir: str | Path, target_name: str) -> dict[str, tuple[str, str]]:
    """Geometry-derived graph group -> (relative PDB path, location type), if one was written."""
    path = Path(name_map_dir) / f"{target_name}.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["graph_group_name"]: (row["relative_pdb_path"], row["location_type"])
                for row in csv.DictReader(handle)}


def build_target_model_references(
    graph_hdf5_path: str | Path,
    target_name: str,
    name_map_dir: str | Path = DEFAULT_NAME_MAP_DIR,
) -> list[ModelReference]:
    graph_hdf5_path = Path(graph_hdf5_path)
    name_map = load_name_map(name_map_dir, target_name)
    references: list[ModelReference] = []
    skipped = 0

    with h5py.File(graph_hdf5_path, "r") as handle:
        for graph_group_name in sorted(handle.keys()):
            if graph_group_name in name_map:
                relative_pdb_path, location_type = name_map[graph_group_name]
            else:
                try:
                    relative_pdb_path, location_type = infer_relative_pdb_path(target_name, graph_group_name)
                except ValueError:
                    if not name_map:
                        raise
                    skipped += 1  # no unique geometric match for this group; left uncomputed
                    continue
            references.append(
                ModelReference(
                    target_name=target_name,
                    graph_group_name=graph_group_name,
                    relative_pdb_path=relative_pdb_path,
                    location_type=location_type,
                )
            )

    if skipped:
        print(f"{target_name}: {skipped} groups have no PDB mapping and are skipped.")
    return references


def write_target_model_references(
    references: Iterable[ModelReference],
    reference_dir: str | Path,
    target_name: str,
) -> tuple[Path, Path]:
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reference_csv_path(reference_dir, target_name)
    txt_path = reference_txt_path(reference_dir, target_name)

    rows = list(references)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(
            csv_handle,
            fieldnames=("target_name", "graph_group_name", "relative_pdb_path", "location_type"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target_name": row.target_name,
                    "graph_group_name": row.graph_group_name,
                    "relative_pdb_path": row.relative_pdb_path,
                    "location_type": row.location_type,
                }
            )

    with txt_path.open("w", encoding="utf-8") as txt_handle:
        for row in rows:
            txt_handle.write(f"{row.graph_group_name}\t{row.relative_pdb_path}\n")

    return csv_path, txt_path


def load_target_model_references(reference_csv: str | Path) -> list[ModelReference]:
    reference_csv = Path(reference_csv)
    references: list[ModelReference] = []
    with reference_csv.open("r", newline="", encoding="utf-8") as csv_handle:
        reader = csv.DictReader(csv_handle)
        for row in reader:
            references.append(
                ModelReference(
                    target_name=row["target_name"],
                    graph_group_name=row["graph_group_name"],
                    relative_pdb_path=row["relative_pdb_path"],
                    location_type=row["location_type"],
                )
            )
    return references
