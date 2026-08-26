"""Local proof-of-concept run: APBS surface electrostatics for the 84 bound complexes.

Each PDB in targets_84_complex_only/ is one bound heterodimer, so unlike the
cluster run (thousands of sampled poses per target) this stores everything --
per-point surface cloud and the raw potential volume -- which is what makes the
output directly explorable before committing to the large run.

    python -m apbs_analysis.run_local_84_targets
    python -m apbs_analysis.run_local_84_targets --max-models 2 --no-store-grid
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from apbs_analysis.cli_options import add_apbs_arguments, settings_from_args
from apbs_analysis.common import (
    LOCAL_OUTPUT_DIR,
    REPO_ROOT,
    ModelInput,
    model_id_from_pdb_name,
)
from apbs_analysis.pipeline import RunOptions, install_signal_handlers, report, run_models


DEFAULT_PDB_DIR = REPO_ROOT / "targets_84_complex_only"
RUN_NAME = "targets_84"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb-dir", default=str(DEFAULT_PDB_DIR), help="Directory of *_complex_H.pdb files")
    parser.add_argument("--output-dir", default=str(LOCAL_OUTPUT_DIR))
    parser.add_argument("--targets", nargs="*", help="Optional subset of PDB ids, e.g. 1acb 2grn")
    parser.add_argument("--max-models", type=int, help="Process only the first N structures")
    parser.add_argument("--no-store-points", action="store_true", help="Skip the per-point surface cloud")
    parser.add_argument("--no-store-grid", action="store_true", help="Skip the raw volumetric potential")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Also copy each PQR, APBS input, log, and gzipped DX map into <output-dir>/dx/targets_84/")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Solve N models in parallel processes; each APBS solve needs a few GiB")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--per-model-timeout", type=float, default=3600.0)
    add_apbs_arguments(parser)
    return parser.parse_args()


def main() -> None:
    install_signal_handlers()
    args = parse_args()

    pdb_dir = Path(args.pdb_dir).resolve()
    if not pdb_dir.is_dir():
        sys.exit(f"No PDB directory at {pdb_dir}")

    wanted = set(args.targets) if args.targets else None
    models = [
        ModelInput(
            target_name=RUN_NAME,
            model_id=model_id_from_pdb_name(pdb_path.name),
            pdb_path=pdb_path,
            location_type="bound_complex",
        )
        for pdb_path in sorted(pdb_dir.glob("*.pdb"))
    ]
    if wanted is not None:
        models = [model for model in models if model.model_id in wanted]
        missing = wanted - {model.model_id for model in models}
        if missing:
            sys.exit(f"Requested targets not found in {pdb_dir}: {', '.join(sorted(missing))}")
    if args.max_models is not None:
        models = models[: args.max_models]
    if not models:
        sys.exit(f"No .pdb files matched in {pdb_dir}")

    output_dir = Path(args.output_dir).expanduser()
    output_hdf5_path = output_dir / f"{RUN_NAME}_apbs_surface.hdf5"
    summary_csv_path = output_dir / f"{RUN_NAME}_apbs_summary.csv"

    print(f"PDB source:  {pdb_dir}")
    print(f"Output HDF5: {output_hdf5_path}")
    print(f"Structures:  {len(models)}")
    print(f"Settings:    {args.forcefield} ff, pH {args.ph}, {args.pbe_solver}, "
          f"{args.ionic_strength} M salt, ~{args.target_spacing} A grid", flush=True)

    options = RunOptions(
        output_hdf5_path=output_hdf5_path,
        summary_csv_path=summary_csv_path,
        settings=settings_from_args(args),
        store_surface_points=not args.no_store_points,
        store_grid=not args.no_store_grid,
        intermediates_dir=(output_dir / "dx" / RUN_NAME) if args.keep_intermediates else None,
        overwrite=args.overwrite,
        log_every=args.log_every,
        timeout=args.per_model_timeout or None,
        workers=max(1, args.workers),
    )
    rows = run_models(models, options, run_attributes={"target_name": RUN_NAME, "pdb_root": str(pdb_dir)})
    errors = report(RUN_NAME, rows, summary_csv_path)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
