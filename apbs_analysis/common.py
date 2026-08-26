"""Shared paths, naming, and the HDF5 output contract for APBS surface data."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent

# Cluster store the array jobs write into (one HDF5 + one summary CSV per target).
CLUSTER_OUTPUT_DIR = Path("/nfs/roberts/pi/pi_co54/jas485/ppi_gnn_data_store/apbs_model_data")
CLUSTER_PDB_BASE_DIR = Path("/nfs/roberts/pi/pi_co54/jas485/uniformly_sampled_target_data")
# Local proof-of-concept store for the 84 bound complexes.
LOCAL_OUTPUT_DIR = Path.home() / "Documents" / "apbs_electrostatics_84_targets"

OUTPUT_DIR_ENV_VAR = "PPI_APBS_OUTPUT_DIR"

# Sampled model PDBs are named <model_id>_corrected_H_0001.pdb; the graph HDF5
# group for the same model is <model_id> (optionally with a _corrected suffix).
SAMPLED_PDB_SUFFIX = "_corrected_H_0001.pdb"

# Every dataset a finished per-model group must contain. Point-cloud and grid
# datasets are optional (opt-in flags), so they are checked separately.
REQUIRED_MODEL_DATASETS = (
    "atom_aa_id",
    "atom_chain",
    "atom_resnum",
    "atom_resname",
    "atom_name",
    "atom_xyz",
    "atom_charge",
    "atom_radius",
    "atom_potential",
    "residue_aa_id",
    "residue_chain",
    "residue_number",
    "residue_name",
    "residue_in_pqr",
    "residue_incomplete",
    "residue_truncated",
    "residue_charge",
    "residue_sasa",
    "residue_surface_point_count",
    "residue_potential_mean",
    "residue_potential_min",
    "residue_potential_max",
    "residue_potential_std",
)
SURFACE_POINT_DATASETS = (
    "surface_xyz",
    "surface_potential",
    "surface_atom_index",
    "surface_residue_index",
)

IN_PROGRESS_PREFIX = "__in_progress__"


@dataclass(frozen=True)
class ModelInput:
    """One PDB structure to run pdb2pqr/APBS on."""

    target_name: str
    model_id: str          # HDF5 group name for this model's results
    pdb_path: Path
    location_type: str     # "sampled" | "random_negative" | "bound_complex"


def target_name_from_dir(target_dir: str | Path) -> str:
    return Path(target_dir).resolve().name


def model_id_from_pdb_name(pdb_name: str) -> str:
    """Map a model PDB filename back to its graph-group-style model id.

    complex.0_0_11_corrected_H_0001.pdb -> complex.0_0_11
    1acb_complex_H.pdb                  -> 1acb
    Any other name falls back to the bare stem.
    """
    name = Path(pdb_name).name
    if name.endswith(SAMPLED_PDB_SUFFIX):
        return name[: -len(SAMPLED_PDB_SUFFIX)]
    stem = Path(name).stem
    return re.sub(r"_complex(_H)?$", "", stem)


def graph_group_candidates(model_id: str) -> tuple[str, str]:
    """Graph HDF5 group names this model id could correspond to.

    The PDB filename is identical for `complex.X_Y_Z` and its `_corrected`
    variant, so the inverse mapping is genuinely ambiguous. Both names are
    returned rather than guessing; consumers joining APBS output back onto a
    graph should take whichever key exists in that target's graph HDF5.
    """
    base = model_id.removesuffix("_corrected")
    return (base, f"{base}_corrected")


def sampled_dir_for_target(pdb_root: str | Path, target_name: str) -> Path:
    return Path(pdb_root) / f"sampled_{target_name}"


def discover_target_models(pdb_root: str | Path, target_name: str) -> list[ModelInput]:
    """Enumerate every model PDB for a target, positives then random negatives.

    Enumerating the filesystem (rather than the graph HDF5 keys) keeps this
    pipeline independent of the training graphs, which is the whole point of
    storing electrostatics separately.
    """
    sampled_dir = sampled_dir_for_target(pdb_root, target_name)
    models: list[ModelInput] = []
    for directory, location_type in (
        (sampled_dir, "sampled"),
        (sampled_dir / "random_negatives", "random_negative"),
    ):
        if not directory.is_dir():
            continue
        for pdb_path in sorted(directory.glob("*.pdb")):
            models.append(
                ModelInput(
                    target_name=target_name,
                    model_id=model_id_from_pdb_name(pdb_path.name),
                    pdb_path=pdb_path,
                    location_type=location_type,
                )
            )
    return models


def target_output_hdf5_path(output_dir: str | Path, target_name: str) -> Path:
    return Path(output_dir) / f"{target_name}_apbs_surface.hdf5"


def target_summary_csv_path(output_dir: str | Path, target_name: str) -> Path:
    return Path(output_dir) / f"{target_name}_apbs_summary.csv"


def target_dx_dir(output_dir: str | Path, target_name: str) -> Path:
    return Path(output_dir) / "dx" / target_name


def model_group_is_complete(group, want_surface_points: bool = False, want_grid: bool = False) -> bool:
    """Whether a per-model group holds a full, shape-consistent result."""
    if any(name not in group for name in REQUIRED_MODEL_DATASETS):
        return False
    if want_surface_points and any(name not in group for name in SURFACE_POINT_DATASETS):
        return False
    if want_grid and "potential_grid" not in group:
        return False

    atom_count = len(group["atom_xyz"])
    residue_count = len(group["residue_number"])
    if atom_count == 0 or residue_count == 0:
        return False
    if group["atom_xyz"].ndim != 2 or group["atom_xyz"].shape[1] != 3:
        return False
    if any(len(group[name]) != atom_count for name in (
        "atom_aa_id", "atom_chain", "atom_resnum", "atom_resname", "atom_name",
        "atom_charge", "atom_radius", "atom_potential",
    )):
        return False
    if any(len(group[name]) != residue_count for name in (
        "residue_aa_id", "residue_chain", "residue_name", "residue_in_pqr",
        "residue_incomplete", "residue_truncated", "residue_charge", "residue_sasa",
        "residue_surface_point_count", "residue_potential_mean",
        "residue_potential_min", "residue_potential_max", "residue_potential_std",
    )):
        return False

    if want_surface_points:
        point_count = len(group["surface_xyz"])
        if group["surface_xyz"].ndim != 2 or group["surface_xyz"].shape[1] != 3:
            return False
        if any(len(group[name]) != point_count for name in SURFACE_POINT_DATASETS[1:]):
            return False

    return True


def default_output_dir(cluster: bool = False) -> str:
    if cluster:
        return str(CLUSTER_OUTPUT_DIR)
    env_path = os.environ.get(OUTPUT_DIR_ENV_VAR)
    return env_path or str(LOCAL_OUTPUT_DIR)
