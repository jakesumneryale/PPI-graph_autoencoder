"""Monomer electrostatics derived from an already-solved complex.

Rather than re-running pdb2pqr on an isolated chain, the monomer's atoms,
charges and radii are taken verbatim from the complex's own solved structure.
That is a deliberate choice, not a shortcut:

  * pdb2pqr run again on a lone chain sees a different environment, and PROPKA
    can return different titration states -- a buried salt bridge that fixed a
    HIS in the complex is solvent-exposed in the monomer. The monomer would
    then carry different charges from the same residue in the complex.
  * Any binding-electrostatics quantity is a difference over a thermodynamic
    cycle, complex minus the two monomers. That difference is only meaningful
    if the charge set is identical on both sides; otherwise part of the answer
    is just the protonation changing underneath it.

So this computes the rigid-body, fixed-charge decomposition: same coordinates,
same charges, same radii, same grid -- only the surrounding chain removed. If
the *relaxed* monomer is wanted instead (its own protonation, its own box),
that is a different quantity: split the chains to PDB with
structure_prep.split_chains and run them through the normal PDB entry point.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import h5py
import numpy as np

from apbs_analysis.dx_grid import read_dx
from apbs_analysis.electrostatics import (
    ApbsSettings,
    PqrStructure,
    SurfaceElectrostatics,
    aggregate_by_residue,
    run_apbs,
    solvent_accessible_points,
    write_apbs_input,
    write_pqr,
)
from apbs_analysis.grid_sizing import GridParameters, compute_grid_parameters
from apbs_analysis.structure_prep import PreparedStructure


def complex_grid(group: h5py.Group) -> GridParameters:
    """Recover the exact APBS box a stored model was solved on.

    The centre comes from the written map (origin plus half the fine extent),
    not from recomputing a molecular centre: APBS's `mol 1` centring differs
    from a recomputed centre by up to ~0.1 A, and it is APBS's choice the
    complex map actually sits on.
    """
    origin = np.asarray(group.attrs["grid_origin"], dtype=float)
    spacing = np.asarray(group.attrs["grid_spacing"], dtype=float)
    dime = np.asarray(group.attrs["grid_shape"], dtype=int)
    return GridParameters(
        dime=tuple(int(value) for value in dime),
        cglen=tuple(float(value) for value in np.asarray(group.attrs["apbs_cglen"], dtype=float)),
        fglen=tuple(float(value) for value in np.asarray(group.attrs["apbs_fglen"], dtype=float)),
        center=tuple(float(value) for value in (origin + (dime - 1) * spacing / 2.0)),
    )


def list_chains(store_path: str | Path, complex_id: str) -> list[str]:
    with h5py.File(Path(store_path).expanduser(), "r") as handle:
        chains = handle[complex_id]["atom_chain"].asstr()[:]
    return sorted(set(chains.tolist()))


def _subset(store_path: Path, complex_id: str, chain: str):
    """Pull one chain's atoms and residues out of a solved complex."""
    with h5py.File(store_path, "r") as handle:
        group = handle[complex_id]
        atom_chain = group["atom_chain"].asstr()[:]
        atom_mask = atom_chain == chain
        if not atom_mask.any():
            raise ValueError(f"{complex_id} has no chain {chain!r}")

        atom_aa_id_full = group["atom_aa_id"][:]
        residue_chain = group["residue_chain"].asstr()[:]
        residue_mask = residue_chain == chain
        parent_aa_id = np.flatnonzero(residue_mask).astype(np.int32)
        # Map the complex's aa_ids onto 0..R_monomer-1 for this chain.
        remap = np.full(len(residue_chain), -1, dtype=np.int32)
        remap[parent_aa_id] = np.arange(len(parent_aa_id), dtype=np.int32)
        atom_aa_id = remap[atom_aa_id_full[atom_mask]]
        if (atom_aa_id < 0).any():
            raise ValueError(f"{complex_id} chain {chain}: atom outside the chain's residues")

        structure = PqrStructure(
            chain=atom_chain[atom_mask].astype("<U4"),
            resnum=group["atom_resnum"][:][atom_mask].astype(np.int32),
            resname=group["atom_pqr_resname"].asstr()[:][atom_mask].astype("<U8"),
            atom_name=group["atom_name"].asstr()[:][atom_mask].astype("<U8"),
            xyz=group["atom_xyz"][:][atom_mask].astype(np.float64),
            charge=group["atom_charge"][:][atom_mask].astype(np.float64),
            radius=group["atom_radius"][:][atom_mask].astype(np.float64),
        )
        prepared = PreparedStructure(
            pdb_path=Path(str(group.attrs.get("source_pdb_path", ""))),
            chain=residue_chain[residue_mask].astype("<U4"),
            number=group["residue_number"][:][residue_mask].astype(np.int32),
            insertion_code=group["residue_insertion_code"].asstr()[:][residue_mask].astype("<U2"),
            name=group["residue_name"].asstr()[:][residue_mask].astype("<U8"),
            modeled_name=group["residue_modeled_name"].asstr()[:][residue_mask].astype("<U8"),
            incomplete=group["residue_incomplete"][:][residue_mask].astype(bool),
            truncated=group["residue_truncated"][:][residue_mask].astype(bool),
            atom_count=np.bincount(atom_aa_id, minlength=len(parent_aa_id)).astype(np.int32),
            warnings=[],
        )
        grid = complex_grid(group)
    return structure, prepared, atom_aa_id, parent_aa_id, grid


def compute_monomer_electrostatics(
    store_path: str | Path,
    complex_id: str,
    chain: str,
    settings: ApbsSettings | None = None,
    match_complex_grid: bool = True,
    scratch_dir: str | Path | None = None,
    keep_grid: bool = False,
    keep_surface_points: bool = False,
    timeout: float | None = None,
) -> SurfaceElectrostatics:
    """Solve one chain of a stored complex, on the complex's own grid."""
    settings = settings or ApbsSettings()
    store_path = Path(store_path).expanduser()
    structure, prepared, atom_aa_id, parent_aa_id, parent_grid = _subset(
        store_path, complex_id, chain
    )

    grid = parent_grid if match_complex_grid else compute_grid_parameters(
        structure.xyz,
        structure.radius,
        coarse_factor=settings.coarse_factor,
        fine_padding=settings.fine_padding,
        target_spacing=settings.target_spacing,
        memory_ceiling_mb=settings.memory_ceiling_mb,
    )

    scratch_parent = Path(scratch_dir) if scratch_dir else None
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)

    model_id = f"{complex_id}_{chain}"
    with tempfile.TemporaryDirectory(prefix=f"apbs_{model_id}_", dir=scratch_parent) as temporary:
        work_dir = Path(temporary)
        pqr_path = write_pqr(structure, work_dir / "monomer.pqr")
        input_path = work_dir / "apbs.in"
        write_apbs_input(
            input_path,
            pqr_path.name,
            grid,
            settings,
            use_explicit_center=match_complex_grid,
        )
        potential = read_dx(run_apbs(input_path, work_dir, settings, timeout=timeout))

    surface_xyz, surface_atom_index, surface_point_area, atom_sasa = solvent_accessible_points(
        structure.xyz,
        structure.radius,
        probe_radius=settings.probe_radius,
        sphere_points=settings.sphere_points,
    )
    surface_potential = (
        potential.sample(surface_xyz) if surface_xyz.size else np.empty(0, dtype=np.float32)
    )
    atom_potential = potential.sample(structure.xyz)

    residue_count = len(prepared)
    aggregates = aggregate_by_residue(
        structure,
        atom_aa_id,
        residue_count,
        surface_atom_index,
        surface_potential,
        surface_point_area,
        atom_sasa,
    )

    warnings: list[str] = []
    buried = int(np.sum(aggregates["residue_surface_point_count"] == 0))
    if buried:
        warnings.append(f"{buried}/{residue_count} residues are fully buried (no surface points)")

    return SurfaceElectrostatics(
        model_id=model_id,
        pdb_path=prepared.pdb_path,
        structure=structure,
        prepared=prepared,
        atom_aa_id=atom_aa_id,
        atom_potential=atom_potential,
        residue_chain=prepared.chain,
        residue_number=prepared.number,
        residue_name=prepared.name,
        residue_in_pqr=aggregates["residue_in_pqr"],
        residue_charge=aggregates["residue_charge"],
        residue_sasa=aggregates["residue_sasa"],
        residue_surface_point_count=aggregates["residue_surface_point_count"],
        residue_potential_mean=aggregates["residue_potential_mean"],
        residue_potential_min=aggregates["residue_potential_min"],
        residue_potential_max=aggregates["residue_potential_max"],
        residue_potential_std=aggregates["residue_potential_std"],
        grid_origin=potential.origin,
        grid_spacing=potential.spacing,
        grid_shape=potential.shape,
        grid_parameters=grid,
        surface_xyz=surface_xyz.astype(np.float32) if keep_surface_points else None,
        surface_potential=surface_potential if keep_surface_points else None,
        surface_point_area=surface_point_area.astype(np.float32) if keep_surface_points else None,
        surface_residue_index=aggregates["surface_residue_index"] if keep_surface_points else None,
        surface_atom_index=surface_atom_index if keep_surface_points else None,
        potential_grid=potential.values if keep_grid else None,
        warnings=warnings,
        extra_attributes={
            "parent_complex_id": complex_id,
            "chain": chain,
            "residue_parent_aa_id": parent_aa_id,
            "grid_matches_complex": bool(match_complex_grid),
            "charges_from_complex": True,
        },
    )
