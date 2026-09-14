"""APBS surface electrostatics for each *monomer* of the 84 bound complexes.

Each chain of each solved complex is re-solved on its own, with two properties
that make the results directly comparable to the complex:

  * **Same grid.** The monomer reuses the complex's exact APBS box (explicit
    centre, same dime/cglen/fglen), so the maps share a lattice and can be
    subtracted voxel by voxel. Left to itself APBS centres each box on its own
    molecule, offsetting the monomer lattice by up to ~0.1 A.
  * **Same charges.** Atoms, charges and radii come from the complex's solved
    structure rather than a fresh pdb2pqr run, so a residue carries identical
    charge in both. See apbs_analysis/monomer.py for why that matters.

Together these give the rigid-body, fixed-charge decomposition that a binding
electrostatics term needs. Pass --independent-grids for the other quantity:
each monomer on its own box (still complex-derived charges).

    python -m apbs_analysis.run_local_84_monomers --workers 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py

from apbs_analysis.cli_options import add_apbs_arguments, settings_from_args
from apbs_analysis.common import IN_PROGRESS_PREFIX, LOCAL_OUTPUT_DIR, ModelInput
from apbs_analysis.monomer import list_chains
from apbs_analysis.pipeline import RunOptions, install_signal_handlers, report, run_models


COMPLEX_RUN_NAME = "targets_84"
RUN_NAME = "targets_84_monomers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(LOCAL_OUTPUT_DIR))
    parser.add_argument(
        "--complex-store",
        help="Solved complex store to derive monomers from "
             "(default: <output-dir>/targets_84_apbs_surface.hdf5)",
    )
    parser.add_argument(
        "--independent-grids",
        action="store_true",
        help="Give each monomer its own box instead of the complex's. Better resolved "
             "per monomer, but the maps can no longer be subtracted from the complex.",
    )
    parser.add_argument("--targets", nargs="*", help="Subset of complex ids, e.g. 1acb 2grn")
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--no-store-points", action="store_true")
    parser.add_argument("--no-store-grid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--per-model-timeout", type=float, default=3600.0)
    add_apbs_arguments(parser)
    return parser.parse_args()


def build_models(store_path: Path, wanted: set[str] | None, match_grid: bool) -> list[ModelInput]:
    with h5py.File(store_path, "r") as handle:
        complex_ids = sorted(name for name in handle if not name.startswith(IN_PROGRESS_PREFIX))
    if wanted is not None:
        missing = wanted - set(complex_ids)
        if missing:
            sys.exit(f"Not in the complex store: {', '.join(sorted(missing))}")
        complex_ids = [name for name in complex_ids if name in wanted]

    models: list[ModelInput] = []
    for complex_id in complex_ids:
        for chain in list_chains(store_path, complex_id):
            models.append(
                ModelInput(
                    target_name=RUN_NAME,
                    model_id=f"{complex_id}_{chain}",
                    pdb_path=store_path,
                    location_type="monomer",
                    # False is the sentinel for "do not match the parent grid";
                    # None would read as "no override supplied".
                    grid_override=None if match_grid else False,
                    monomer_source=(str(store_path), complex_id, chain),
                )
            )
    return models


def main() -> None:
    install_signal_handlers()
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser()
    store_path = Path(
        args.complex_store or output_dir / f"{COMPLEX_RUN_NAME}_apbs_surface.hdf5"
    ).expanduser()
    if not store_path.is_file():
        sys.exit(f"No complex store at {store_path}. Run run_local_84_targets first.")

    wanted = set(args.targets) if args.targets else None
    models = build_models(store_path, wanted, match_grid=not args.independent_grids)
    if args.max_models is not None:
        models = models[: args.max_models]
    if not models:
        sys.exit("No monomers to process")

    output_hdf5_path = output_dir / f"{RUN_NAME}_apbs_surface.hdf5"
    summary_csv_path = output_dir / f"{RUN_NAME}_apbs_summary.csv"

    print(f"Complex store: {store_path}")
    print(f"Output HDF5:   {output_hdf5_path}")
    print(f"Monomers:      {len(models)}")
    print(f"Grids:         {'matched to parent complex' if not args.independent_grids else 'independent'}")
    print(f"Charges:       taken from the solved complex", flush=True)

    options = RunOptions(
        output_hdf5_path=output_hdf5_path,
        summary_csv_path=summary_csv_path,
        settings=settings_from_args(args),
        store_surface_points=not args.no_store_points,
        store_grid=not args.no_store_grid,
        overwrite=args.overwrite,
        log_every=args.log_every,
        timeout=args.per_model_timeout or None,
        workers=max(1, args.workers),
    )
    rows = run_models(
        models,
        options,
        run_attributes={
            "target_name": RUN_NAME,
            "complex_store": str(store_path),
            "grids_match_complex": not args.independent_grids,
        },
    )
    errors = report(RUN_NAME, rows, summary_csv_path)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
