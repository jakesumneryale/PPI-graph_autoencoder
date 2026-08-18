"""Precompute per-target text/CSV reference lists for Voronoi edge-feature runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from voronoi_edge_features.common import DEFAULT_REFERENCE_DIR, default_graph_data_path
from voronoi_edge_features.model_reference import (
    build_target_model_references,
    write_target_model_references,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan target graph HDF5 files and save per-target model-reference lists "
            "that map each graph key to its expected PDB path inside a target directory."
        )
    )
    parser.add_argument("--data", help="Directory containing target graph .hdf5 files.")
    parser.add_argument("--targets", nargs="*", help="Optional target names to restrict the scan.")
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--cluster", action="store_true", help="Use cluster default paths when --data is omitted.")
    return parser.parse_args()


def iter_target_hdf5_paths(graph_data_dir: str | Path, targets: list[str] | None) -> list[Path]:
    graph_data_dir = Path(graph_data_dir)
    if targets:
        target_paths: list[Path] = []
        for target_name in targets:
            for suffix in (".hdf5", ".h5"):
                candidate = graph_data_dir / f"{target_name}{suffix}"
                if candidate.exists():
                    target_paths.append(candidate)
                    break
            else:
                raise FileNotFoundError(f"Could not find {target_name}.hdf5 or {target_name}.h5 in {graph_data_dir}")
        return target_paths

    target_paths = sorted(graph_data_dir.glob("*.hdf5"))
    if not target_paths:
        target_paths = sorted(graph_data_dir.glob("*.h5"))
    if not target_paths:
        raise FileNotFoundError(f"No graph HDF5 files found in {graph_data_dir}")
    return target_paths


def main() -> None:
    args = parse_args()
    if args.data is None:
        args.data = default_graph_data_path(cluster=args.cluster)

    target_paths = iter_target_hdf5_paths(args.data, args.targets)
    print(f"Writing model-reference lists for {len(target_paths)} target graph files...")

    for graph_hdf5_path in target_paths:
        target_name = graph_hdf5_path.stem
        references = build_target_model_references(graph_hdf5_path, target_name)
        csv_path, txt_path = write_target_model_references(references, args.reference_dir, target_name)
        print(f"  {target_name}: {len(references)} entries -> {csv_path} and {txt_path}")


if __name__ == "__main__":
    main()
