"""HDF5 layout for APBS surface electrostatics.

One file per target, one group per model, mirroring the Voronoi checkpoint
convention. Nothing here ever touches the training graph HDF5 files: results
live in their own store so they can be regenerated independently.
"""

from __future__ import annotations

import h5py
import numpy as np

from apbs_analysis.common import IN_PROGRESS_PREFIX
from apbs_analysis.electrostatics import POTENTIAL_UNITS, SurfaceElectrostatics


COMPRESSION = {"compression": "gzip", "compression_opts": 4}


def _write_strings(group: h5py.Group, name: str, values: np.ndarray) -> None:
    """Store a string column as fixed-width bytes, not variable-length strings.

    Every chain/atom/residue name here is at most a few characters. HDF5's
    variable-length strings put each value's bytes in a global heap that the
    dataset filter never compresses, and the per-value heap IDs alone cost more
    than the strings do -- measured at 7.2x the total file size across a
    representative target, i.e. ~130 GiB rather than ~18 GiB over the full
    cluster run. Fixed-width `S<n>` columns compress normally and still read
    back through `.asstr()` exactly as before.
    """
    values = np.asarray(values).astype(str)
    width = max(1, int(max((len(value) for value in values.ravel()), default=1)))
    encoded = np.char.encode(values, "utf-8").astype(f"S{width}")
    group.create_dataset(name, data=encoded, **COMPRESSION)


def _write_array(group: h5py.Group, name: str, values: np.ndarray, dtype) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=dtype), **COMPRESSION)


def write_model_group(group: h5py.Group, result: SurfaceElectrostatics) -> None:
    """Write one model's atom-, residue-, and (optional) point-level results."""
    structure = result.structure
    prepared = result.prepared
    group.attrs["model_id"] = result.model_id
    group.attrs["source_pdb_path"] = str(result.pdb_path)
    group.attrs["num_atoms"] = int(len(structure))
    group.attrs["num_residues"] = int(len(result.residue_number))
    group.attrs["potential_units"] = POTENTIAL_UNITS
    group.attrs["grid_origin"] = result.grid_origin.astype(np.float64)
    group.attrs["grid_spacing"] = result.grid_spacing.astype(np.float64)
    group.attrs["grid_shape"] = np.asarray(result.grid_shape, dtype=np.int32)
    group.attrs["apbs_dime"] = np.asarray(result.grid_parameters.dime, dtype=np.int32)
    group.attrs["apbs_cglen"] = np.asarray(result.grid_parameters.cglen, dtype=np.float64)
    group.attrs["apbs_fglen"] = np.asarray(result.grid_parameters.fglen, dtype=np.float64)
    group.attrs["total_charge"] = float(structure.charge.sum())
    group.attrs["total_sasa"] = float(np.nansum(result.residue_sasa))
    group.attrs["num_incomplete_residues"] = int(prepared.incomplete.sum())
    group.attrs["num_truncated_residues"] = int(prepared.truncated.sum())
    group.attrs["num_residues_absent_from_pqr"] = int((~result.residue_in_pqr).sum())
    if result.warnings:
        group.attrs["warnings"] = "; ".join(result.warnings)

    # Atom chain/residue identity is taken from the *original* PDB via aa_id,
    # not from the PQR, whose numbering is the prepared sequential one.
    _write_array(group, "atom_aa_id", result.atom_aa_id, np.int32)
    _write_strings(group, "atom_chain", prepared.chain[result.atom_aa_id])
    _write_array(group, "atom_resnum", prepared.number[result.atom_aa_id], np.int32)
    _write_strings(group, "atom_resname", prepared.name[result.atom_aa_id])
    _write_strings(group, "atom_name", structure.atom_name)
    _write_strings(group, "atom_pqr_resname", structure.resname)
    _write_array(group, "atom_xyz", structure.xyz, np.float32)
    _write_array(group, "atom_charge", structure.charge, np.float32)
    _write_array(group, "atom_radius", structure.radius, np.float32)
    _write_array(group, "atom_potential", result.atom_potential, np.float32)

    # aa_id is the join key back to graph nodes: entry i is the (i+1)-th
    # residue in the source PDB's file order, matching the sequential counter
    # create_protein_graph_structure.py assigns.
    _write_array(group, "residue_aa_id", np.arange(len(result.residue_number)), np.int32)
    _write_strings(group, "residue_chain", result.residue_chain)
    _write_array(group, "residue_number", result.residue_number, np.int32)
    _write_strings(group, "residue_insertion_code", prepared.insertion_code)
    _write_strings(group, "residue_name", result.residue_name)
    _write_strings(group, "residue_modeled_name", prepared.modeled_name)
    _write_array(group, "residue_incomplete", prepared.incomplete, np.uint8)
    _write_array(group, "residue_truncated", prepared.truncated, np.uint8)
    _write_array(group, "residue_in_pqr", result.residue_in_pqr, np.uint8)
    _write_array(group, "residue_charge", result.residue_charge, np.float32)
    _write_array(group, "residue_sasa", result.residue_sasa, np.float32)
    _write_array(group, "residue_surface_point_count", result.residue_surface_point_count, np.int32)
    _write_array(group, "residue_potential_mean", result.residue_potential_mean, np.float32)
    _write_array(group, "residue_potential_min", result.residue_potential_min, np.float32)
    _write_array(group, "residue_potential_max", result.residue_potential_max, np.float32)
    _write_array(group, "residue_potential_std", result.residue_potential_std, np.float32)

    if result.surface_xyz is not None:
        _write_array(group, "surface_xyz", result.surface_xyz, np.float32)
        _write_array(group, "surface_potential", result.surface_potential, np.float32)
        _write_array(group, "surface_atom_index", result.surface_atom_index, np.int32)
        _write_array(group, "surface_residue_index", result.surface_residue_index, np.int32)
        group.attrs["num_surface_points"] = int(len(result.surface_xyz))

    if result.potential_grid is not None:
        group.create_dataset(
            "potential_grid",
            data=result.potential_grid.astype(np.float32),
            chunks=True,
            **COMPRESSION,
        )


def commit_model_group(
    handle: h5py.File, model_id: str, result: SurfaceElectrostatics
) -> None:
    """Write to a staging group and rename only once every dataset landed.

    A killed or requeued job can therefore never leave a half-written group
    that a later resume would mistake for finished work.
    """
    staging_name = f"{IN_PROGRESS_PREFIX}{model_id}"
    if staging_name in handle:
        del handle[staging_name]
    staging_group = handle.create_group(staging_name)
    try:
        write_model_group(staging_group, result)
        handle.flush()
        if model_id in handle:
            del handle[model_id]
        handle.move(staging_name, model_id)
        handle.flush()
    except BaseException:
        if staging_name in handle:
            del handle[staging_name]
            handle.flush()
        raise


def discard_staging_groups(handle: h5py.File) -> int:
    """Drop leftover staging groups from a previously interrupted run."""
    stale = [name for name in handle.keys() if name.startswith(IN_PROGRESS_PREFIX)]
    for name in stale:
        del handle[name]
    if stale:
        handle.flush()
    return len(stale)
