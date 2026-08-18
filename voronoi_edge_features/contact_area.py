"""Compute residue-level Voronoi contact areas aligned to graph node IDs."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from voronoi_edge_features.common import REPO_ROOT


@contextmanager
def _patched_argv(argv: list[str]):
    original_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original_argv


def _load_module_from_repo(module_name: str, filename: str):
    module_path = REPO_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with _patched_argv([str(module_path)]):
        spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_voronoi_dependencies():
    create_protein_graph_structure = _load_module_from_repo(
        "voronoi_create_protein_graph_structure",
        "create_protein_graph_structure.py",
    )
    bounded_voronoi_contacts_radical = _load_module_from_repo(
        "voronoi_bounded_voronoi_contacts_radical",
        "bounded_voronoi_contacts_radical.py",
    )
    return create_protein_graph_structure, bounded_voronoi_contacts_radical


def _face_vertices(cell: dict, face: dict) -> np.ndarray:
    vertex_ids = face.get("vertices", [])
    if len(vertex_ids) < 3:
        return np.empty((0, 3), dtype=float)
    return np.asarray([cell["vertices"][vertex_id] for vertex_id in vertex_ids], dtype=float)


def compute_face_area(cell: dict, face: dict) -> float:
    direct_area = face.get("area")
    if direct_area is not None:
        return float(direct_area)

    vertices = _face_vertices(cell, face)
    if len(vertices) < 3:
        return 0.0

    area_vector = np.zeros(3, dtype=float)
    for index in range(len(vertices)):
        area_vector += np.cross(vertices[index], vertices[(index + 1) % len(vertices)])
    return 0.5 * float(np.linalg.norm(area_vector))


def build_node_metadata_table(protein_df: pd.DataFrame) -> pd.DataFrame:
    node_table = (
        protein_df[["aa_id", "aa_ind", "chain_id", "chain_name", "aa_name"]]
        .drop_duplicates(subset=["aa_id"])
        .sort_values("aa_id")
        .reset_index(drop=True)
    )
    node_table["aa_id"] = node_table["aa_id"].astype(int)
    node_table["aa_ind"] = node_table["aa_ind"].astype(int)
    node_table["chain_id"] = node_table["chain_id"].astype(int)
    node_table["chain_name"] = node_table["chain_name"].astype(str)
    node_table["aa_name"] = node_table["aa_name"].astype(str)
    return node_table


def compute_residue_contact_area_table(protein_df: pd.DataFrame, bounded_voronoi_tessellation: Iterable[dict]) -> pd.DataFrame:
    atom_to_aa_id = protein_df["aa_id"].astype(int).to_numpy()
    pair_to_area: dict[tuple[int, int], float] = defaultdict(float)
    pair_to_face_count: dict[tuple[int, int], int] = defaultdict(int)

    for cell in bounded_voronoi_tessellation:
        cell_id = int(cell.get("cell_id", -1))
        if cell_id < 0:
            continue
        aa_id_1 = int(atom_to_aa_id[cell_id])

        for face in cell.get("faces", []):
            adjacent_cell = int(face.get("adjacent_cell", -1))
            if adjacent_cell < 0 or cell_id >= adjacent_cell:
                continue

            aa_id_2 = int(atom_to_aa_id[adjacent_cell])
            if aa_id_1 == aa_id_2:
                continue

            pair = (min(aa_id_1, aa_id_2), max(aa_id_1, aa_id_2))
            pair_to_area[pair] += compute_face_area(cell, face)
            pair_to_face_count[pair] += 1

    rows = [
        {
            "aa_id1": pair[0],
            "aa_id2": pair[1],
            "voronoi_contact_area": area,
            "atom_face_count": pair_to_face_count[pair],
        }
        for pair, area in sorted(pair_to_area.items())
    ]
    return pd.DataFrame(
        rows,
        columns=("aa_id1", "aa_id2", "voronoi_contact_area", "atom_face_count"),
    )


def align_contact_areas_to_graph_edges(contact_area_table: pd.DataFrame, graph_contacts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    graph_contacts = np.asarray(graph_contacts, dtype=int)
    area_map = {
        (int(min(row.aa_id1, row.aa_id2)), int(max(row.aa_id1, row.aa_id2))): float(row.voronoi_contact_area)
        for row in contact_area_table.itertuples(index=False)
    }

    aligned = np.zeros((len(graph_contacts), 1), dtype=np.float32)
    missing_mask = np.zeros((len(graph_contacts), 1), dtype=bool)

    for index, pair in enumerate(graph_contacts):
        key = (int(min(pair[0], pair[1])), int(max(pair[0], pair[1])))
        value = area_map.get(key)
        if value is None:
            missing_mask[index, 0] = True
            continue
        aligned[index, 0] = value

    return aligned, missing_mask


def load_protein_dataframe(pdb_filename: str, pdb_directory: str | Path) -> pd.DataFrame:
    create_protein_graph_structure, _ = load_voronoi_dependencies()
    original_cwd = Path.cwd()
    try:
        return create_protein_graph_structure.get_protein_information(pdb_filename, Path(pdb_directory))
    finally:
        os.chdir(original_cwd)


def compute_bounded_voronoi_tessellation(protein_df: pd.DataFrame, probe_size: float):
    _, bounded_voronoi_contacts_radical = load_voronoi_dependencies()
    return bounded_voronoi_contacts_radical.get_bounded_voro(
        protein_df,
        box_margin=1,
        dispersion=4.5,
        probe_size=probe_size,
    )
