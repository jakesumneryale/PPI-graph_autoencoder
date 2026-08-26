#!/usr/bin/env python3
"""Regenerate apbs_analysis/cluster/targets.txt from the sampled PDB directory.

Line N maps to SLURM array task N, so if the count changes the --array range in
dispatch_apbs_jobs.slurm must be updated to match. Targets are taken from the
sampled_<target>/ directories rather than from the graph HDF5 files, keeping
this pipeline independent of the training graphs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_root", help="Directory containing sampled_<target>/ directories")
    parser.add_argument("output_path", help="Where to write the newline-separated target list")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdb_root = Path(args.pdb_root)
    names = sorted(
        path.name.removeprefix("sampled_")
        for path in pdb_root.iterdir()
        if path.is_dir() and path.name.startswith("sampled_")
    )
    if not names:
        raise SystemExit(f"No sampled_* directories found in {pdb_root}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(names) + "\n")
    print(f"Wrote {len(names)} target names to {output_path}")
    print(f"Update --array=1-{len(names)} in dispatch_apbs_jobs.slurm to match.")


if __name__ == "__main__":
    main()
