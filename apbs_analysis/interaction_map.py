"""Interaction potential map: complex minus its isolated chains.

    dphi(r) = phi_complex(r) - sum_chains phi_chain(r)

This is only meaningful because the monomer runs share the complex's exact
lattice *and* its charge set (see METHODS.md section 7). It isolates what the
partner chain does to the field: where dphi is large, the presence of the other
chain changes the local potential, which is where the electrostatic component
of binding lives.

Reads the complex store and the monomer store, writes an OpenDX map PyMOL can
ramp-colour, and reports per-residue interaction potentials.

    python -m apbs_analysis.interaction_map <complex.hdf5> <monomers.hdf5> 1acb -o out/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

from apbs_analysis.common import IN_PROGRESS_PREFIX
from apbs_analysis.dx_grid import DxGrid
from apbs_analysis.export_dx import write_dx


def _require_aligned(complex_group: h5py.Group, monomer_groups: list[h5py.Group]) -> None:
    """Refuse to subtract maps that are not on the same lattice."""
    for group in monomer_groups:
        for key in ("grid_origin", "grid_spacing", "grid_shape"):
            if not np.allclose(np.asarray(group.attrs[key], dtype=float),
                               np.asarray(complex_group.attrs[key], dtype=float)):
                raise ValueError(
                    f"{group.attrs['model_id']} is not on the complex's lattice ({key} differs). "
                    "Regenerate the monomers without --independent-grids."
                )


def interaction_grid(complex_store: Path, monomer_store: Path, complex_id: str) -> tuple[DxGrid, list[str]]:
    with h5py.File(complex_store, "r") as complexes, h5py.File(monomer_store, "r") as monomers:
        if complex_id not in complexes:
            raise KeyError(f"{complex_id} not in {complex_store}")
        complex_group = complexes[complex_id]
        chain_names = sorted(
            name for name in monomers
            if not name.startswith(IN_PROGRESS_PREFIX)
            and monomers[name].attrs.get("parent_complex_id") == complex_id
        )
        if not chain_names:
            raise KeyError(f"No monomers for {complex_id} in {monomer_store}")
        chain_groups = [monomers[name] for name in chain_names]
        _require_aligned(complex_group, chain_groups)

        if "potential_grid" not in complex_group:
            raise KeyError(f"{complex_id} was stored without a potential grid")
        values = complex_group["potential_grid"][:].astype(np.float64)
        for group in chain_groups:
            if "potential_grid" not in group:
                raise KeyError(f"{group.attrs['model_id']} was stored without a potential grid")
            values -= group["potential_grid"][:].astype(np.float64)

        grid = DxGrid(
            origin=np.asarray(complex_group.attrs["grid_origin"], dtype=float),
            spacing=np.asarray(complex_group.attrs["grid_spacing"], dtype=float),
            values=values.astype(np.float32),
        )
    return grid, chain_names


def _fixed_surface_delta(
    complex_group: h5py.Group, monomer_group: h5py.Group, residue_count: int
) -> np.ndarray | None:
    """Field change at fixed geometry: both maps sampled at the *same* points.

    Needed because a residue's solvent-accessible patch shrinks when the
    partner binds. Averaging the complex map over the complex's surface and
    the monomer map over the monomer's surface therefore compares two
    different areas, mixing the geometry change into what should be a field
    change. For interface residues those two definitions differ by several
    kT/e. Sampling both maps on the monomer's point set removes that: what is
    left is purely how the partner chain altered the potential.

    Returns None when either store was written without a volumetric map.
    """
    if "potential_grid" not in complex_group or "potential_grid" not in monomer_group:
        return None
    if "surface_xyz" not in monomer_group:
        return None

    def as_grid(group: h5py.Group) -> DxGrid:
        return DxGrid(
            origin=np.asarray(group.attrs["grid_origin"], dtype=float),
            spacing=np.asarray(group.attrs["grid_spacing"], dtype=float),
            values=group["potential_grid"][:],
        )

    points = monomer_group["surface_xyz"][:]
    residue_index = monomer_group["surface_residue_index"][:]
    area = monomer_group["surface_point_area"][:].astype(np.float64)
    delta = (
        as_grid(complex_group).sample(points).astype(np.float64)
        - as_grid(monomer_group).sample(points).astype(np.float64)
    )
    weighted = np.bincount(residue_index, weights=delta * area, minlength=residue_count)
    total = np.bincount(residue_index, weights=area, minlength=residue_count)
    return np.where(total > 0, weighted / np.maximum(total, 1e-12), np.nan)


def interaction_residues(complex_store: Path, monomer_store: Path, complex_id: str) -> pd.DataFrame:
    """Per-residue interaction potential.

    Two columns, because there are two defensible definitions and they differ
    substantially at the interface:

    * `interaction_potential` -- both maps sampled at the monomer's surface
      points, so it is purely the field change. Use this one.
    * `interaction_potential_own_surface` -- each state averaged over its own
      surface, which also folds in the surface a residue loses on binding.

    Away from the interface the two agree to ~1e-3 kT/e; on interface residues
    they can differ by tens of kT/e. Residues buried in either state are NaN.
    """
    rows = []
    with h5py.File(complex_store, "r") as complexes, h5py.File(monomer_store, "r") as monomers:
        complex_group = complexes[complex_id]
        complex_mean = complex_group["residue_potential_mean"][:].astype(np.float64)
        complex_sasa = complex_group["residue_sasa"][:].astype(np.float64)
        for name in sorted(monomers):
            group = monomers[name]
            if group.attrs.get("parent_complex_id") != complex_id:
                continue
            parent = group.attrs["residue_parent_aa_id"]
            monomer_mean = group["residue_potential_mean"][:].astype(np.float64)
            monomer_sasa = group["residue_sasa"][:].astype(np.float64)
            fixed = _fixed_surface_delta(complex_group, group, len(parent))
            rows.append(
                pd.DataFrame({
                    "complex_id": complex_id,
                    "chain": str(group.attrs["chain"]),
                    "residue_aa_id": parent,
                    "residue_number": group["residue_number"][:],
                    "residue_name": group["residue_name"].asstr()[:],
                    "residue_charge": group["residue_charge"][:],
                    "potential_complex": complex_mean[parent],
                    "potential_monomer": monomer_mean,
                    "sasa_complex": complex_sasa[parent],
                    "sasa_monomer": monomer_sasa,
                    "interaction_potential": (
                        fixed if fixed is not None else np.full(len(parent), np.nan)
                    ),
                })
            )
    frame = pd.concat(rows, ignore_index=True)
    frame["interaction_potential_own_surface"] = (
        frame["potential_complex"] - frame["potential_monomer"]
    )
    if frame["interaction_potential"].isna().all():
        # No volumetric maps stored, so the fixed-geometry form is unavailable.
        frame["interaction_potential"] = frame["interaction_potential_own_surface"]
    # SASA lost on binding: the standard, purely geometric interface definition.
    frame["buried_sasa"] = frame["sasa_monomer"] - frame["sasa_complex"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("complex_store")
    parser.add_argument("monomer_store")
    parser.add_argument("complex_id")
    parser.add_argument("-o", "--output-dir", default="apbs_interaction")
    parser.add_argument("--no-gzip", action="store_true")
    parser.add_argument("--interface-cutoff", type=float, default=1.0,
                        help="Buried SASA (A^2) above which a residue is called interface")
    args = parser.parse_args()

    complex_store = Path(args.complex_store).expanduser()
    monomer_store = Path(args.monomer_store).expanduser()
    for path in (complex_store, monomer_store):
        if not path.is_file():
            sys.exit(f"No store at {path}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        grid, chains = interaction_grid(complex_store, monomer_store, args.complex_id)
    except (KeyError, ValueError) as exc:
        sys.exit(str(exc))

    suffix = ".dx" if args.no_gzip else ".dx.gz"
    dx_path = write_dx(grid, output_dir / f"{args.complex_id}_interaction{suffix}")

    residues = interaction_residues(complex_store, monomer_store, args.complex_id)
    csv_path = output_dir / f"{args.complex_id}_interaction_residues.csv"
    residues.to_csv(csv_path, index=False)

    interface = residues[residues["buried_sasa"] > args.interface_cutoff]
    print(f"{args.complex_id}: complex - {' - '.join(chains)}")
    print(f"  map      {dx_path}  ({dx_path.stat().st_size / 1024**2:.1f} MiB)")
    print(f"           range {grid.values.min():+.2f} .. {grid.values.max():+.2f} kT/e; "
          f"|dphi| > 1 in {100 * np.mean(np.abs(grid.values) > 1):.2f}% of voxels")
    print(f"  residues {csv_path}  ({len(residues)} rows)")
    print(f"  interface (buried SASA > {args.interface_cutoff} A^2): {len(interface)} residues")
    if len(interface):
        print(f"           mean interaction potential {interface['interaction_potential'].mean():+.3f} kT/e")
        print(f"           total buried SASA {interface['buried_sasa'].sum():.0f} A^2")
        strongest = interface.reindex(interface["interaction_potential"].abs().sort_values(ascending=False).index)
        print("  strongest shifts:")
        for _, row in strongest.head(5).iterrows():
            print(f"    {row['chain']}/{row['residue_name']}{int(row['residue_number']):<5d} "
                  f"dphi {row['interaction_potential']:+7.3f} kT/e   q {row['residue_charge']:+.1f}   "
                  f"buried {row['buried_sasa']:6.1f} A^2")


if __name__ == "__main__":
    main()
