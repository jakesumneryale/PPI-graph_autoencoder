#!/usr/bin/env python3
"""Compute three 1acb Voronoi models and export notebook-friendly geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd

from voronoi_edge_features.contact_area import (
    build_node_metadata_table,
    compute_bounded_voronoi_tessellation,
    compute_face_area,
    compute_residue_contact_area_table,
    load_protein_dataframe,
)
from voronoi_edge_features.model_reference import build_target_model_references


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = Path("/scratch/ppi_autoencoder_code/processed_graph_data/1acb.hdf5")
DEFAULT_PDB_ROOT = Path("/scratch/uniformly_sampled_ppi_data")
DEFAULT_OUTPUT = REPO_ROOT / "voronoi_demo_outputs" / "1acb_three_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--pdb-root", type=Path, default=DEFAULT_PDB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="*", help="Graph group names; defaults to the first three available positives")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--probe-size", type=float, default=1.4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def residue_centres(protein_df: pd.DataFrame, coordinate_shift: np.ndarray) -> pd.DataFrame:
    rows = []
    for aa_id, atoms in protein_df.groupby("aa_id", sort=True):
        ca_atoms = atoms[atoms["atom_name"].astype(str).str.strip() == "CA"] if "atom_name" in atoms else atoms.iloc[0:0]
        representative = ca_atoms.iloc[0] if len(ca_atoms) else atoms.iloc[0]
        coords = representative[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=float) - coordinate_shift
        rows.append(
            {
                "aa_id": int(aa_id),
                "aa_ind": int(representative["aa_ind"]),
                "aa_name": str(representative["aa_name"]),
                "chain_id": int(representative["chain_id"]),
                "chain_name": str(representative["chain_name"]),
                "x": coords[0],
                "y": coords[1],
                "z": coords[2],
            }
        )
    return pd.DataFrame(rows)


def face_geometry(
    protein_df: pd.DataFrame,
    tessellation: list[dict],
    contact_areas: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    atom_to_aa = protein_df["aa_id"].astype(int).to_numpy()
    atom_to_chain = protein_df["chain_name"].astype(str).to_numpy()
    area_lookup = {
        (int(row.aa_id1), int(row.aa_id2)): float(row.voronoi_contact_area)
        for row in contact_areas.itertuples(index=False)
    }
    triangles: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []

    for cell in tessellation:
        cell_id = int(cell.get("cell_id", -1))
        if cell_id < 0:
            continue
        for face in cell.get("faces", []):
            adjacent = int(face.get("adjacent_cell", -1))
            if adjacent < 0 or cell_id >= adjacent:
                continue
            aa1, aa2 = int(atom_to_aa[cell_id]), int(atom_to_aa[adjacent])
            if aa1 == aa2:
                continue
            vertex_ids = face.get("vertices", [])
            if len(vertex_ids) < 3:
                continue
            vertices = np.asarray([cell["vertices"][index] for index in vertex_ids], dtype=np.float32)
            pair = (min(aa1, aa2), max(aa1, aa2))
            chain1, chain2 = str(atom_to_chain[cell_id]), str(atom_to_chain[adjacent])
            face_area = compute_face_area(cell, face)
            for index in range(1, len(vertices) - 1):
                triangles.append(np.asarray([vertices[0], vertices[index], vertices[index + 1]], dtype=np.float32))
                metadata.append(
                    {
                        "aa_id1": pair[0],
                        "aa_id2": pair[1],
                        "chain1": chain1,
                        "chain2": chain2,
                        "is_interchain": chain1 != chain2,
                        "face_area": face_area,
                        "contact_area": area_lookup[pair],
                    }
                )

    triangle_array = np.stack(triangles) if triangles else np.empty((0, 3, 3), dtype=np.float32)
    return triangle_array, pd.DataFrame(metadata)


def write_string_array(group: h5py.Group, name: str, values: pd.Series) -> None:
    group.create_dataset(name, data=values.astype(str).to_numpy(dtype=object), dtype=h5py.string_dtype("utf-8"))


def write_demo_hdf5(
    output_path: Path,
    model_name: str,
    pdb_path: Path,
    probe_size: float,
    elapsed_seconds: float,
    residues: pd.DataFrame,
    contacts: pd.DataFrame,
    triangles: np.ndarray,
    triangle_metadata: pd.DataFrame,
) -> None:
    with h5py.File(output_path, "w") as handle:
        handle.attrs.update(
            model_name=model_name,
            pdb_path=str(pdb_path),
            probe_size=probe_size,
            elapsed_seconds=elapsed_seconds,
            area_units="angstrom^2",
            coordinate_units="angstrom",
        )
        residue_group = handle.create_group("residues")
        for column in ("aa_id", "aa_ind", "chain_id", "x", "y", "z"):
            residue_group.create_dataset(column, data=residues[column].to_numpy())
        for column in ("aa_name", "chain_name"):
            write_string_array(residue_group, column, residues[column])

        contact_group = handle.create_group("contacts")
        for column in ("aa_id1", "aa_id2", "voronoi_contact_area", "atom_face_count"):
            contact_group.create_dataset(column, data=contacts[column].to_numpy())

        face_group = handle.create_group("faces")
        face_group.create_dataset("triangles", data=triangles, compression="gzip", compression_opts=4)
        if len(triangle_metadata):
            for column in ("aa_id1", "aa_id2", "is_interchain", "face_area", "contact_area"):
                face_group.create_dataset(column, data=triangle_metadata[column].to_numpy(), compression="gzip")
            for column in ("chain1", "chain2"):
                write_string_array(face_group, column, triangle_metadata[column])


def main() -> None:
    args = parse_args()
    graph_path = args.graph.resolve()
    pdb_root = args.pdb_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    references = [
        row
        for row in build_target_model_references(graph_path, "1acb")
        if row.location_type == "sampled" and (pdb_root / row.relative_pdb_path).is_file()
    ]
    if args.models:
        wanted = set(args.models)
        references = [row for row in references if row.graph_group_name in wanted]
        missing = wanted - {row.graph_group_name for row in references}
        if missing:
            raise SystemExit(f"Requested model(s) not found with matching PDBs: {sorted(missing)}")
    if not references:
        raise SystemExit("No matching positive 1acb models were found")

    manifest = []
    completed = 0
    attempted = 0
    requested_total = len(references) if args.models else args.count
    for reference in references:
        if not args.models and completed >= args.count:
            break
        attempted += 1
        output_path = output_dir / f"{reference.graph_group_name}.h5"
        if output_path.exists() and not args.overwrite:
            print(f"[{completed + 1}/{requested_total}] {reference.graph_group_name}: already exists; using it")
            manifest.append({"model_name": reference.graph_group_name, "output_hdf5": str(output_path)})
            completed += 1
            continue

        pdb_path = pdb_root / reference.relative_pdb_path
        print(f"[{completed + 1}/{requested_total}] {reference.graph_group_name}: {pdb_path}", flush=True)
        started = time.time()
        try:
            protein_df = load_protein_dataframe(pdb_path.name, pdb_path.parent)
            coordinate_shift = protein_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=float).mean(axis=0)
            tessellation = compute_bounded_voronoi_tessellation(protein_df, args.probe_size)
            contacts = compute_residue_contact_area_table(protein_df, tessellation)
            residues = residue_centres(protein_df, coordinate_shift)
            triangles, triangle_metadata = face_geometry(protein_df, tessellation, contacts)
        except Exception as exc:  # keep searching for three demonstrable models
            elapsed = time.time() - started
            print(f"  FAILED after {elapsed:.1f}s: {exc}", flush=True)
            manifest.append(
                {
                    "model_name": reference.graph_group_name,
                    "pdb_path": str(pdb_path),
                    "status": "error",
                    "message": str(exc),
                    "elapsed_seconds": elapsed,
                }
            )
            continue
        elapsed = time.time() - started
        write_demo_hdf5(
            output_path,
            reference.graph_group_name,
            pdb_path,
            args.probe_size,
            elapsed,
            residues,
            contacts,
            triangles,
            triangle_metadata,
        )
        contacts.to_csv(output_dir / f"{reference.graph_group_name}_contact_areas.csv", index=False)
        print(
            f"  {len(residues)} residues, {len(contacts)} residue contacts, "
            f"{len(triangles)} face triangles in {elapsed:.1f}s",
            flush=True,
        )
        manifest.append(
            {
                "model_name": reference.graph_group_name,
                "status": "success",
                "pdb_path": str(pdb_path),
                "output_hdf5": str(output_path),
                "contact_csv": str(output_dir / f"{reference.graph_group_name}_contact_areas.csv"),
                "elapsed_seconds": elapsed,
                "residue_count": len(residues),
                "contact_count": len(contacts),
                "triangle_count": len(triangles),
            }
        )
        completed += 1

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if completed < requested_total:
        raise SystemExit(f"Only produced {completed}/{requested_total} requested models after {attempted} attempts")
    print(f"Demo data written to {output_dir}")


if __name__ == "__main__":
    main()
