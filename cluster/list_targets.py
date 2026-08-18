"""Regenerate cluster/targets.txt from the graph HDF5 directory.

cluster/targets.txt is checked into the repo with the current 146 targets.
Only rerun this if the set of targets on the cluster changes -- each line
number corresponds 1:1 to a SLURM array task ID (line 1 -> task ID 1), so
the --array range in dispatch_voronoi_jobs.slurm must be updated to match
if the count changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_data_dir", help="Directory containing <target>.hdf5 files")
    parser.add_argument("output_path", help="Where to write the newline-separated target list")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_data_dir = Path(args.graph_data_dir)
    names = sorted(
        p.stem for p in graph_data_dir.iterdir() if p.suffix in (".hdf5", ".h5") and p.is_file()
    )
    if not names:
        raise SystemExit(f"No .hdf5/.h5 files found in {graph_data_dir}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(names) + "\n")
    print(f"Wrote {len(names)} target names to {output_path}")


if __name__ == "__main__":
    main()
