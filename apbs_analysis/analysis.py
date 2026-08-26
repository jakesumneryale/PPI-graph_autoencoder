"""Loaders for turning the APBS HDF5 store into pandas tables.

The store is deliberately not merged into the training graph HDF5 files, so
this is the join layer: residue rows carry chain/number/name, which is enough
to line them up with graph nodes (or with the Voronoi contact tables) when
that is wanted.

    from apbs_analysis.analysis import load_residue_table
    residues = load_residue_table("~/Documents/apbs_electrostatics_84_targets/targets_84_apbs_surface.hdf5")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd

from apbs_analysis.common import IN_PROGRESS_PREFIX
from apbs_analysis.dx_grid import DxGrid


RESIDUE_COLUMNS = (
    "residue_aa_id",
    "residue_chain",
    "residue_number",
    "residue_name",
    "residue_incomplete",
    "residue_truncated",
    "residue_in_pqr",
    "residue_charge",
    "residue_sasa",
    "residue_surface_point_count",
    "residue_potential_mean",
    "residue_potential_min",
    "residue_potential_max",
    "residue_potential_std",
)

ATOM_COLUMNS = (
    "atom_aa_id",
    "atom_chain",
    "atom_resnum",
    "atom_resname",
    "atom_name",
    "atom_charge",
    "atom_radius",
    "atom_potential",
)


def list_models(hdf5_path: str | Path) -> list[str]:
    with h5py.File(Path(hdf5_path).expanduser(), "r") as handle:
        return sorted(name for name in handle.keys() if not name.startswith(IN_PROGRESS_PREFIX))


def _read_column(group: h5py.Group, name: str) -> np.ndarray:
    dataset = group[name]
    return dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) else dataset[:]


def _stack(hdf5_path: Path, columns: Sequence[str], models: Iterable[str] | None, extra) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with h5py.File(hdf5_path, "r") as handle:
        names = (
            list(models)
            if models is not None
            else sorted(n for n in handle.keys() if not n.startswith(IN_PROGRESS_PREFIX))
        )
        for name in names:
            group = handle[name]
            if any(column not in group for column in columns):
                continue  # incomplete group from an interrupted run
            frame = pd.DataFrame({column: _read_column(group, column) for column in columns})
            # Stored as uint8 for HDF5 compactness; read back as real booleans.
            for flag in ("residue_truncated", "residue_incomplete", "residue_in_pqr"):
                if flag in frame:
                    frame[flag] = frame[flag].astype(bool)
            frame.insert(0, "model_id", name)
            frame.insert(1, "target_name", handle.attrs.get("target_name", ""))
            for column, value in extra(group).items():
                frame[column] = value
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["model_id", "target_name", *columns])
    return pd.concat(frames, ignore_index=True)


def load_residue_table(
    hdf5_path: str | Path, models: Iterable[str] | None = None, exposed_only: bool = False
) -> pd.DataFrame:
    """One row per residue per model.

    Buried residues have no surface points, so their potential statistics are
    NaN; exposed_only drops them rather than letting NaNs propagate silently.
    """
    frame = _stack(
        Path(hdf5_path).expanduser(),
        RESIDUE_COLUMNS,
        models,
        lambda group: {"total_charge": float(group.attrs.get("total_charge", np.nan))},
    )
    if exposed_only and not frame.empty:
        frame = frame[frame["residue_surface_point_count"] > 0].reset_index(drop=True)
    return frame


def load_atom_table(hdf5_path: str | Path, models: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per atom per model, with the potential sampled at each centre."""
    return _stack(Path(hdf5_path).expanduser(), ATOM_COLUMNS, models, lambda group: {})


def load_surface_points(hdf5_path: str | Path, model_id: str) -> pd.DataFrame:
    """The per-point surface cloud for one model (requires --store-surface-points)."""
    with h5py.File(Path(hdf5_path).expanduser(), "r") as handle:
        group = handle[model_id]
        if "surface_xyz" not in group:
            raise KeyError(f"{model_id} was written without a surface point cloud")
        xyz = group["surface_xyz"][:]
        residue_index = group["surface_residue_index"][:]
        return pd.DataFrame(
            {
                "x": xyz[:, 0],
                "y": xyz[:, 1],
                "z": xyz[:, 2],
                "potential": group["surface_potential"][:],
                "atom_index": group["surface_atom_index"][:],
                "residue_index": residue_index,
                "residue_chain": group["residue_chain"].asstr()[:][residue_index],
                "residue_number": group["residue_number"][:][residue_index],
                "residue_name": group["residue_name"].asstr()[:][residue_index],
            }
        )


def load_potential_grid(hdf5_path: str | Path, model_id: str) -> DxGrid:
    """The stored volumetric potential for one model (requires --store-grid)."""
    with h5py.File(Path(hdf5_path).expanduser(), "r") as handle:
        group = handle[model_id]
        if "potential_grid" not in group:
            raise KeyError(f"{model_id} was written without a potential grid")
        return DxGrid(
            origin=np.asarray(group.attrs["grid_origin"], dtype=np.float64),
            spacing=np.asarray(group.attrs["grid_spacing"], dtype=np.float64),
            values=group["potential_grid"][:],
        )


def interface_residues(
    residues: pd.DataFrame, potential_column: str = "residue_potential_mean"
) -> pd.DataFrame:
    """Per-model, per-chain summary: charge, area, and area-weighted potential.

    Surface points are near-equal-area samples, so weighting each residue's
    mean by its SASA recovers the whole-chain mean surface potential.
    """
    exposed = residues[residues["residue_surface_point_count"] > 0].copy()
    exposed["_weighted"] = exposed[potential_column] * exposed["residue_sasa"]
    grouped = exposed.groupby(["model_id", "residue_chain"], as_index=False).agg(
        residue_count=("residue_number", "size"),
        total_charge=("residue_charge", "sum"),
        total_sasa=("residue_sasa", "sum"),
        weighted_potential=("_weighted", "sum"),
        min_potential=(potential_column, "min"),
        max_potential=(potential_column, "max"),
    )
    grouped["mean_surface_potential"] = grouped["weighted_potential"] / grouped["total_sasa"]
    return grouped.drop(columns="weighted_potential")
