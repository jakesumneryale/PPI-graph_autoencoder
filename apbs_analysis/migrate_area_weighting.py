"""Recompute residue potential statistics as area-weighted, in place.

Stores written before the area-weighting fix hold `residue_potential_mean` as
an unweighted mean over surface points. Points represent different areas
(PARSE radii run 0.0-2.0 A, so nearly 6x), which makes that an approximation
to the true surface average rather than the average itself.

Everything needed to correct it is already in the file -- `surface_atom_index`
and `atom_radius` give each point's area -- so this is a recomputation from
stored values, not a new APBS run. It requires a store written with
--store-surface-points; cluster stores without points must be regenerated.

    python -m apbs_analysis.migrate_area_weighting <store.hdf5>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

from apbs_analysis.common import IN_PROGRESS_PREFIX
from apbs_analysis.electrostatics import _grouped_statistics


RECOMPUTED = ("residue_potential_mean", "residue_potential_std")


def migrate_group(group: h5py.Group, probe_radius: float, sphere_points: int) -> float:
    """Rewrite one model's weighted statistics; return the largest change."""
    atom_radius = group["atom_radius"][:].astype(np.float64)
    owner = group["surface_atom_index"][:]
    potential = group["surface_potential"][:].astype(np.float64)
    residue_index = group["surface_residue_index"][:]
    residue_count = len(group["residue_number"])

    point_area = 4.0 * np.pi * (atom_radius[owner] + probe_radius) ** 2 / sphere_points
    means, _minima, _maxima, stds, _counts = _grouped_statistics(
        potential, residue_index, residue_count, point_area
    )

    previous = group["residue_potential_mean"][:].astype(np.float64)
    finite = np.isfinite(previous) & np.isfinite(means)
    largest_change = float(np.abs(previous[finite] - means[finite]).max()) if finite.any() else 0.0

    for name, values in (("residue_potential_mean", means), ("residue_potential_std", stds)):
        del group[name]
        group.create_dataset(
            name, data=values.astype(np.float32), compression="gzip", compression_opts=4
        )
    if "surface_point_area" not in group:
        group.create_dataset(
            "surface_point_area",
            data=point_area.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
    return largest_change


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store_path")
    parser.add_argument("--dry-run", action="store_true", help="Report the changes without writing")
    args = parser.parse_args()

    path = Path(args.store_path).expanduser()
    if not path.is_file():
        sys.exit(f"No store at {path}")

    mode = "r" if args.dry_run else "r+"
    with h5py.File(path, mode) as handle:
        if not handle.attrs.get("stores_surface_points", False):
            sys.exit(
                "This store has no surface point cloud, so the point areas cannot be "
                "recovered. Regenerate it with the current code instead."
            )
        probe_radius = float(handle.attrs["probe_radius"])
        sphere_points = int(handle.attrs["sphere_points"])
        names = sorted(name for name in handle if not name.startswith(IN_PROGRESS_PREFIX))

        changes = []
        for name in names:
            group = handle[name]
            if args.dry_run:
                atom_radius = group["atom_radius"][:].astype(np.float64)
                owner = group["surface_atom_index"][:]
                area = 4.0 * np.pi * (atom_radius[owner] + probe_radius) ** 2 / sphere_points
                means, _, _, _, _ = _grouped_statistics(
                    group["surface_potential"][:].astype(np.float64),
                    group["surface_residue_index"][:],
                    len(group["residue_number"]),
                    area,
                )
                previous = group["residue_potential_mean"][:].astype(np.float64)
                finite = np.isfinite(previous) & np.isfinite(means)
                changes.append(float(np.abs(previous[finite] - means[finite]).max()) if finite.any() else 0.0)
            else:
                changes.append(migrate_group(group, probe_radius, sphere_points))

        if not args.dry_run:
            handle.attrs["residue_statistics_area_weighted"] = True

    changes = np.asarray(changes)
    verb = "would change" if args.dry_run else "changed"
    print(f"{len(names)} models {verb}; largest per-residue shift in kT/e:")
    print(f"  median {np.median(changes):.4f}   p90 {np.percentile(changes, 90):.4f}   max {changes.max():.4f}")
    if not args.dry_run:
        print("Residue means and stds are now area-weighted; surface_point_area added.")


if __name__ == "__main__":
    main()
