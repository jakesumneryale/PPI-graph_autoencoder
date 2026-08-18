"""Helpers for mapping graph HDF5 keys to target-model PDB paths."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import h5py

from voronoi_edge_features.common import (
    ModelReference,
    infer_relative_pdb_path,
    reference_csv_path,
    reference_txt_path,
)


def build_target_model_references(graph_hdf5_path: str | Path, target_name: str) -> list[ModelReference]:
    graph_hdf5_path = Path(graph_hdf5_path)
    references: list[ModelReference] = []

    with h5py.File(graph_hdf5_path, "r") as handle:
        for graph_group_name in sorted(handle.keys()):
            relative_pdb_path, location_type = infer_relative_pdb_path(target_name, graph_group_name)
            references.append(
                ModelReference(
                    target_name=target_name,
                    graph_group_name=graph_group_name,
                    relative_pdb_path=relative_pdb_path,
                    location_type=location_type,
                )
            )

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
