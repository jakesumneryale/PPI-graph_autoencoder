"""Read-only QC summary of an APBS surface-electrostatics HDF5.

Reports coverage against the source PDB directory, then spot-checks one model:
value ranges, buried-residue counts, and the charge/potential agreement that
should hold for any correctly solved map (positive residues sit in positive
potential). Analogous to cluster/check_voronoi_output.sh for the Voronoi run.

    python -m apbs_analysis.inspect_apbs_output ~/Documents/apbs_electrostatics_84_targets/targets_84_apbs_surface.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

from apbs_analysis.common import IN_PROGRESS_PREFIX, model_group_is_complete


POSITIVE_RESIDUES = ("ARG", "LYS")
NEGATIVE_RESIDUES = ("ASP", "GLU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hdf5_path")
    parser.add_argument("model", nargs="?", help="Model/group to spot-check (default: the first one)")
    parser.add_argument("--pdb-dir", help="Compare the stored groups against the .pdb files in this directory")
    parser.add_argument("--list", action="store_true", help="List every group with its residue count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.hdf5_path).expanduser()
    if not path.is_file():
        sys.exit(f"No HDF5 at {path}")

    with h5py.File(path, "r") as handle:
        groups = sorted(name for name in handle.keys() if not name.startswith(IN_PROGRESS_PREFIX))
        staging = [name for name in handle.keys() if name.startswith(IN_PROGRESS_PREFIX)]
        stores_points = bool(handle.attrs.get("stores_surface_points", False))
        stores_grid = bool(handle.attrs.get("stores_potential_grid", False))

        print(f"FILE  {path}  ({path.stat().st_size / 1024**2:.1f} MiB)")
        print("RUN SETTINGS")
        for key in sorted(handle.attrs.keys()):
            print(f"  {key:26s} {handle.attrs[key]}")

        incomplete = [
            name for name in groups
            if not model_group_is_complete(handle[name], want_surface_points=stores_points, want_grid=stores_grid)
        ]
        print("\nCOVERAGE")
        print(f"  complete model groups: {len(groups) - len(incomplete)}/{len(groups)}")
        if incomplete:
            print(f"  incomplete:            {', '.join(incomplete[:8])}")
        if staging:
            print(f"  staging leftovers:     {len(staging)} (cleared on the next run)")
        if args.pdb_dir:
            from apbs_analysis.common import model_id_from_pdb_name

            # rglob, not glob: a target's models live in sampled_<target>/ *and*
            # its random_negatives/ subdirectory.
            expected = {model_id_from_pdb_name(p.name) for p in Path(args.pdb_dir).rglob("*.pdb")}
            missing = sorted(expected - set(groups))
            print(f"  PDBs in {args.pdb_dir}: {len(expected)}; missing from HDF5: {len(missing)}")
            if missing:
                print(f"  missing examples:      {', '.join(missing[:8])}")

        if args.list:
            print("\nGROUPS")
            for name in groups:
                group = handle[name]
                print(f"  {name:28s} {group.attrs.get('num_residues', '?'):>5} residues"
                      f"  {group.attrs.get('num_atoms', '?'):>6} atoms")

        if not groups:
            return
        model = args.model or groups[0]
        if model not in handle:
            sys.exit(f"Model {model!r} not in {path}")
        group = handle[model]

        print(f"\nMODEL SPOT CHECK: {model}")
        for key in ("source_pdb_path", "num_atoms", "num_residues", "num_surface_points",
                    "grid_shape", "grid_spacing", "total_charge", "total_sasa",
                    "num_incomplete_residues", "num_truncated_residues",
                    "num_residues_absent_from_pqr", "warnings"):
            if key in group.attrs:
                print(f"  {key:22s} {group.attrs[key]}")

        chains = group["residue_chain"].asstr()[:]
        names = group["residue_name"].asstr()[:]
        potential = group["residue_potential_mean"][:]
        charge = group["residue_charge"][:]
        exposed = group["residue_surface_point_count"][:] > 0

        aa_id = group["residue_aa_id"][:]
        contiguous = np.array_equal(aa_id, np.arange(len(aa_id)))
        print(f"  aa_id join key         0..{len(aa_id) - 1} "
              f"[{'contiguous, aligns with graph nodes' if contiguous else 'NOT CONTIGUOUS -- join is unsafe'}]")

        unique_chains, chain_counts = np.unique(chains, return_counts=True)
        print(f"  chains                 {dict(zip(unique_chains.tolist(), chain_counts.tolist()))}")
        print(f"  surface-exposed        {int(exposed.sum())}/{len(exposed)} residues")
        if exposed.any():
            values = potential[exposed]
            print(f"  residue potential      min {values.min():+.3f} / mean {values.mean():+.3f} "
                  f"/ max {values.max():+.3f} kT/e")

        print("  sign check (mean surface potential by residue type):")
        for label, selection in (
            ("ARG/LYS", np.isin(names, POSITIVE_RESIDUES)),
            ("ASP/GLU", np.isin(names, NEGATIVE_RESIDUES)),
            ("all other", ~np.isin(names, POSITIVE_RESIDUES + NEGATIVE_RESIDUES)),
        ):
            mask = selection & exposed
            value = potential[mask].mean() if mask.any() else float("nan")
            print(f"    {label:10s} n={int(mask.sum()):5d}  {value:+.3f} kT/e")
        if exposed.sum() > 2:
            # Across the 84 bound complexes this ran from +0.11 to +0.53, with
            # the low end being net-charged complexes whose whole surface is
            # offset one way. The threshold sits below that range: it is there
            # to catch a broken pipeline (a transposed grid axis or a bad
            # residue mapping would drive this to ~0), not to grade a model.
            correlation = float(np.corrcoef(charge[exposed], potential[exposed])[0, 1])
            verdict = (
                "OK" if correlation > 0.05
                else "SUSPECT -- charge and potential should correlate positively"
            )
            print(f"    corr(formal charge, potential) = {correlation:+.3f}  [{verdict}]")

        if stores_points and "surface_potential" in group:
            points = group["surface_potential"][:]
            print(f"  surface points         {points.size}; potential "
                  f"{points.min():+.3f} / {points.mean():+.3f} / {points.max():+.3f} kT/e")
            finite = np.isfinite(points)
            if not finite.all():
                print(f"    WARNING: {int((~finite).sum())} non-finite point potentials")
        if stores_grid and "potential_grid" in group:
            dataset = group["potential_grid"]
            print(f"  potential grid         {dataset.shape}; "
                  f"{dataset.id.get_storage_size() / 1024**2:.1f} MiB stored")


if __name__ == "__main__":
    main()
