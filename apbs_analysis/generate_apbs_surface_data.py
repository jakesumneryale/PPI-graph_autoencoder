"""Compute APBS surface electrostatics for every model PDB of one target.

Cluster entry point: one target per SLURM array task, mirroring the Voronoi
workers. Output goes to <output-dir>/<target>_apbs_surface.hdf5 plus a summary
CSV; the training graph HDF5 files are never opened or modified.

    python -m apbs_analysis.generate_apbs_surface_data 1acb \
        --pdb-root /nfs/roberts/pi/pi_co54/jas485/uniformly_sampled_target_data \
        --output-dir /nfs/roberts/pi/pi_co54/jas485/ppi_gnn_data_store/apbs_model_data
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from apbs_analysis.cli_options import add_apbs_arguments, settings_from_args
from apbs_analysis.common import (
    CLUSTER_PDB_BASE_DIR,
    default_output_dir,
    discover_target_models,
    sampled_dir_for_target,
    target_dx_dir,
    target_name_from_dir,
    target_output_hdf5_path,
    target_summary_csv_path,
)
from apbs_analysis.pipeline import RunOptions, install_signal_handlers, report, run_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="Target name (e.g. 1acb) or a full path ending in the target name")
    parser.add_argument(
        "--pdb-root",
        default=str(CLUSTER_PDB_BASE_DIR),
        help="Directory that directly contains sampled_<target>/ (and its random_negatives/)",
    )
    parser.add_argument("--output-dir", help="Where the per-target HDF5 and summary CSV are written")
    parser.add_argument("--cluster", action="store_true", help="Use the cluster default output directory")
    parser.add_argument("--store-surface-points", action="store_true",
                        help="Also store the full per-point surface cloud (large; off by default at scale)")
    parser.add_argument("--store-grid", action="store_true",
                        help="Also store the raw volumetric potential (very large; off by default at scale)")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Copy each model's PQR, APBS input, logs, and DX map into <output-dir>/dx/<target>/")
    parser.add_argument("--scratch-dir", help="Node-local scratch for intermediates (defaults to the system temp dir)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute models that already have a group")
    parser.add_argument("--max-models", type=int, help="Optional limit for quick debugging")
    parser.add_argument("--workers", type=int, default=1,
                        help="Solve N models in parallel processes; keep <= --cpus-per-task")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--per-model-timeout", type=float, default=3600.0,
                        help="Seconds before a single pdb2pqr/APBS call is abandoned (0 disables)")
    add_apbs_arguments(parser)
    return parser.parse_args()


def main() -> None:
    install_signal_handlers()
    args = parse_args()

    target_name = target_name_from_dir(args.target) if "/" in args.target else args.target
    output_dir = Path(args.output_dir or default_output_dir(cluster=args.cluster))
    pdb_root = Path(args.pdb_root).resolve()

    sampled_dir = sampled_dir_for_target(pdb_root, target_name)
    if not sampled_dir.is_dir():
        print(f"SKIP: no sampled PDB directory at {sampled_dir}", file=sys.stderr)
        sys.exit(0)

    models = discover_target_models(pdb_root, target_name)
    if not models:
        print(f"SKIP: {sampled_dir} contains no .pdb files", file=sys.stderr)
        sys.exit(0)
    if args.max_models is not None:
        models = models[: args.max_models]

    output_hdf5_path = target_output_hdf5_path(output_dir, target_name)
    summary_csv_path = target_summary_csv_path(output_dir, target_name)

    print(f"Target:       {target_name}")
    print(f"PDB source:   {sampled_dir}")
    print(f"Output HDF5:  {output_hdf5_path}")
    print(f"Models:       {len(models)}")
    print(f"Settings:     {args.forcefield} ff, pH {args.ph}, {args.pbe_solver}, "
          f"{args.ionic_strength} M salt, ~{args.target_spacing} A grid", flush=True)

    options = RunOptions(
        output_hdf5_path=output_hdf5_path,
        summary_csv_path=summary_csv_path,
        settings=settings_from_args(args),
        store_surface_points=args.store_surface_points,
        store_grid=args.store_grid,
        intermediates_dir=target_dx_dir(output_dir, target_name) if args.keep_intermediates else None,
        scratch_dir=Path(args.scratch_dir) if args.scratch_dir else None,
        overwrite=args.overwrite,
        log_every=args.log_every,
        timeout=args.per_model_timeout or None,
        workers=max(1, args.workers),
    )
    rows = run_models(
        models,
        options,
        run_attributes={"target_name": target_name, "pdb_root": str(pdb_root)},
    )
    errors = report(target_name, rows, summary_csv_path)
    # Non-zero exit tells the wrapper/resubmit script this target is incomplete.
    sys.exit(1 if errors or len(rows) < len(models) else 0)


if __name__ == "__main__":
    main()
