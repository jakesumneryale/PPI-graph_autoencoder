"""Screened-electrostatics approach curves for all 84 dimers.

For each complex the smaller chain is walked in from a long-range separation to
its crystallographic bound pose, and the Debye-screened interaction with the
stationary chain is evaluated at every step.

Writes per dimer:
  <id>_approach.csv           one row per step: energies, geometry, path diagnostics
  <id>_approach_residues.csv  one row per (step, residue): that residue's share
  approach_summary.csv        one row per dimer: the headline numbers

    python -m apbs_analysis.run_local_84_approach --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures_module
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd

from apbs_analysis.approach import (
    COULOMB_CONSTANT, ScreeningModel, analyse_pair, load_chain_pair,
)
from apbs_analysis.common import IN_PROGRESS_PREFIX, LOCAL_OUTPUT_DIR


COMPLEX_RUN_NAME = "targets_84"
OUTPUT_SUBDIR = "approach"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(LOCAL_OUTPUT_DIR))
    parser.add_argument("--complex-store")
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--max-displacement", type=float, default=50.0,
                        help="Starting separation along the pull-out axis (A)")
    parser.add_argument("--step", type=float, default=0.5, help="Increment along the path (A)")
    parser.add_argument("--ionic-strength", type=float, default=0.150,
                        help="mol/L; 0.150 is cytosolic and matches the APBS runs")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--dielectric", type=float, default=78.54)
    parser.add_argument("--residue-stride", type=int, default=1,
                        help="Write per-residue rows every Nth step (0 disables)")
    parser.add_argument("--move-larger-chain", action="store_true",
                        help="Move the larger chain instead of the smaller one")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_one(complex_id: str, store_path: str, output_dir: str, options: dict) -> dict:
    model = ScreeningModel(
        ionic_strength=options["ionic_strength"],
        temperature=options["temperature"],
        dielectric=options["dielectric"],
    )
    output_dir = Path(output_dir)
    trajectory_path = output_dir / f"{complex_id}_approach.csv"
    residue_path = output_dir / f"{complex_id}_approach_residues.csv"

    if trajectory_path.is_file() and not options["overwrite"]:
        previous = pd.read_csv(trajectory_path)
        return {"complex_id": complex_id, "status": "skipped_exists", **_headline(previous, model)}

    start_time = time.time()
    first, second = load_chain_pair(store_path, complex_id)
    if options["move_larger_chain"]:
        moving, fixed = (first, second) if len(first.xyz) >= len(second.xyz) else (second, first)
    else:
        moving, fixed = (first, second) if len(first.xyz) < len(second.xyz) else (second, first)

    path, trajectory, residues = analyse_pair(
        moving, fixed, model,
        max_displacement=options["max_displacement"],
        step=options["step"],
        residue_stride=options["residue_stride"],
    )
    trajectory.insert(0, "complex_id", complex_id)
    trajectory.to_csv(trajectory_path, index=False)
    if residues is not None:
        residues.insert(0, "complex_id", complex_id)
        residues.to_csv(residue_path, index=False)

    return {
        "complex_id": complex_id,
        "status": "success",
        "moving_chain": moving.label,
        "fixed_chain": fixed.label,
        "moving_atoms": len(moving.xyz),
        "fixed_atoms": len(fixed.xyz),
        "moving_net_charge": round(moving.net_charge, 3),
        "fixed_net_charge": round(fixed.net_charge, 3),
        "path_mode": path.mode,
        "axis_angle_from_com_deg": round(path.axis_angle_from_com, 2),
        "path_clash_free": path.clash_free,
        "native_gap_angstrom": round(path.native_gap, 3),
        "worst_gap_angstrom": round(path.worst_gap, 3),
        "max_rotation_degrees": round(path.max_rotation_degrees, 2),
        "path_notes": " | ".join(path.notes),
        "elapsed_seconds": round(time.time() - start_time, 2),
        **_headline(trajectory, model),
    }


def _headline(trajectory: pd.DataFrame, model: ScreeningModel) -> dict:
    """The numbers worth having per dimer without opening the full curve."""
    energy = trajectory["interaction_energy_kcal_per_mol"].to_numpy()
    displacement = trajectory["displacement_angstrom"].to_numpy()
    bound = float(energy[-1])
    longest = float(energy[0])

    # Where the interaction first becomes appreciable on the way in.
    threshold = 0.5 * model.kt_kcal  # half a kT
    significant = np.flatnonzero(np.abs(energy) > threshold)
    onset = float(displacement[significant[0]]) if significant.size else float("nan")

    # A sign change between long range and contact means the monopoles and the
    # interface patches disagree -- steering that reverses on close approach.
    long_range = energy[displacement > 25.0]
    long_range_sign = float(np.sign(np.mean(long_range))) if long_range.size else 0.0
    overlap = trajectory["steric_overlap"].to_numpy() if "steric_overlap" in trajectory else np.zeros(len(trajectory), bool)
    clean = ~overlap
    return {
        "frames": len(trajectory),
        "frames_with_steric_overlap": int(overlap.sum()),
        "energy_bound_kcal_per_mol": round(bound, 4),
        "energy_bound_kT": round(bound / model.kt_kcal, 3),
        "energy_at_max_separation_kcal_per_mol": round(longest, 6),
        "energy_unscreened_bound_kcal_per_mol": round(
            float(trajectory["interaction_energy_unscreened_kcal_per_mol"].iloc[-1]), 4),
        "screening_factor_at_contact": round(
            float(energy[-1] / trajectory["interaction_energy_unscreened_kcal_per_mol"].iloc[-1])
            if trajectory["interaction_energy_unscreened_kcal_per_mol"].iloc[-1] != 0 else np.nan, 4),
        "onset_displacement_angstrom": round(onset, 2) if np.isfinite(onset) else "",
        "monotonic_attraction": bool(np.all(np.diff(energy) <= 1e-9)) if bound < 0 else False,
        "sign_flips_on_approach": bool(bound * long_range_sign < 0),
        "min_energy_kcal_per_mol": round(float(energy.min()), 4),
        "min_energy_displacement_angstrom": round(float(displacement[int(energy.argmin())]), 2),
        # Same, restricted to frames without steric overlap, where the model holds.
        "min_energy_clean_kcal_per_mol": round(float(energy[clean].min()), 4) if clean.any() else "",
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser()
    store_path = Path(args.complex_store or output_root / f"{COMPLEX_RUN_NAME}_apbs_surface.hdf5").expanduser()
    if not store_path.is_file():
        sys.exit(f"No complex store at {store_path}")

    output_dir = output_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(store_path, "r") as handle:
        available = sorted(n for n in handle if not n.startswith(IN_PROGRESS_PREFIX))
    targets = args.targets or available
    missing = [t for t in targets if t not in available]
    if missing:
        sys.exit(f"Not in the store: {', '.join(missing)}")

    model = ScreeningModel(args.ionic_strength, args.temperature, args.dielectric)
    print(f"Complex store:  {store_path}")
    print(f"Output:         {output_dir}")
    print(f"Dimers:         {len(targets)}")
    print(f"Screening:      I = {model.ionic_strength} M, T = {model.temperature} K, "
          f"eps = {model.dielectric}  ->  Debye length {model.debye_length:.2f} A")
    print(f"Path:           {args.max_displacement} A in to the bound pose in {args.step} A steps "
          f"({int(args.max_displacement / args.step) + 1} frames)", flush=True)

    options = {
        "max_displacement": args.max_displacement,
        "step": args.step,
        "ionic_strength": args.ionic_strength,
        "temperature": args.temperature,
        "dielectric": args.dielectric,
        "residue_stride": args.residue_stride,
        "move_larger_chain": args.move_larger_chain,
        "overwrite": args.overwrite,
    }

    rows: list[dict] = []
    if args.workers > 1:
        with futures_module.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one, t, str(store_path), str(output_dir), options): t
                for t in targets
            }
            for done, future in enumerate(futures_module.as_completed(futures), start=1):
                target = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    rows.append({"complex_id": target, "status": "error", "message": str(exc)[:400]})
                if done % 10 == 0:
                    print(f"  {done}/{len(targets)} dimers", flush=True)
    else:
        for index, target in enumerate(targets, start=1):
            try:
                rows.append(run_one(target, str(store_path), str(output_dir), options))
            except Exception as exc:  # noqa: BLE001
                rows.append({"complex_id": target, "status": "error", "message": str(exc)[:400]})
            if index % 10 == 0:
                print(f"  {index}/{len(targets)} dimers", flush=True)

    summary = pd.DataFrame(rows).sort_values("complex_id")
    for key, value in model.as_attributes().items():
        summary[key] = value
    summary_path = output_dir / "approach_summary.csv"
    summary.to_csv(summary_path, index=False)

    errors = int((summary["status"] == "error").sum())
    print(f"\nFinished: {len(summary) - errors} succeeded, {errors} failed -> {summary_path}")
    if errors:
        for _, row in summary[summary["status"] == "error"].iterrows():
            print(f"  {row['complex_id']}: {row.get('message', '')}")


if __name__ == "__main__":
    main()
