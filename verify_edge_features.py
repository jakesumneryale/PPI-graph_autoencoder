"""Check that every graph HDF5 carries the edge features a run will ask for.

``ProteinGraphHDF5Dataset`` only reads feature datasets when a graph is fetched,
so a target missing ``voronoi_contact_missing`` would not fail until training was
already underway on a GPU. Running this first turns that into a fast, cheap
failure with a list of exactly which targets need attention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory of graph HDF5 files, or one file.")
    parser.add_argument(
        "--edge-features",
        required=True,
        help="Comma-separated edge features that must be present in every graph group.",
    )
    parser.add_argument(
        "--targets-file",
        default=None,
        help="Optional file of target names (one per line) restricting which HDF5 files are checked.",
    )
    parser.add_argument(
        "--max-groups-per-file",
        type=int,
        default=0,
        help="Check only the first N graph groups per file (0 = every group).",
    )
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="Treat a target with no HDF5 file as a skip rather than a failure.",
    )
    return parser.parse_args()


def resolve_files(data: Path, targets_file: Path | None) -> tuple[list[Path], list[str]]:
    if data.is_file():
        return [data], []
    if targets_file is None:
        return sorted(data.glob("*.hdf5")) + sorted(data.glob("*.h5")), []
    names = [line.strip() for line in targets_file.read_text().splitlines() if line.strip()]
    files: list[Path] = []
    absent: list[str] = []
    for name in names:
        for suffix in (".hdf5", ".h5"):
            candidate = data / f"{name}{suffix}"
            if candidate.is_file():
                files.append(candidate)
                break
        else:
            absent.append(name)
    return files, absent


def check_file(path: Path, features: list[str], max_groups: int) -> tuple[int, dict[str, int]]:
    missing_counts: dict[str, int] = {}
    checked = 0
    with h5py.File(path, "r") as handle:
        group_names = sorted(handle.keys())
        if max_groups > 0:
            group_names = group_names[:max_groups]
        for group_name in group_names:
            edge_group = handle[group_name].get("edge_features")
            if edge_group is None:
                missing_counts["<no edge_features group>"] = missing_counts.get("<no edge_features group>", 0) + 1
                continue
            for feature in features:
                if feature not in edge_group:
                    missing_counts[feature] = missing_counts.get(feature, 0) + 1
            checked += 1
    return checked, missing_counts


def main() -> int:
    args = parse_args()
    data = Path(args.data)
    features = [item.strip() for item in args.edge_features.split(",") if item.strip()]
    targets_file = Path(args.targets_file) if args.targets_file else None

    files, absent = resolve_files(data, targets_file)
    if not files:
        print(f"No graph HDF5 files found under {data}", file=sys.stderr)
        return 2

    print(f"Verifying {features} across {len(files)} file(s) from {data}")
    failures: list[str] = []
    total_groups = 0
    for path in files:
        try:
            checked, missing = check_file(path, features, args.max_groups_per_file)
        except OSError as exc:
            failures.append(f"{path.name}: unreadable ({exc})")
            continue
        total_groups += checked
        if missing:
            detail = ", ".join(f"{name} missing in {count} group(s)" for name, count in sorted(missing.items()))
            failures.append(f"{path.name}: {detail}")

    if absent:
        message = f"{len(absent)} target(s) have no HDF5 file: {absent[:8]}"
        if args.allow_missing_files:
            print(f"SKIP: {message}")
        else:
            failures.append(message)

    print(f"Checked {total_groups} graph group(s).")
    if failures:
        print(f"\nFAILED: {len(failures)} file(s)/condition(s) incomplete:", file=sys.stderr)
        for line in failures[:25]:
            print(f"  {line}", file=sys.stderr)
        if len(failures) > 25:
            print(f"  ... and {len(failures) - 25} more", file=sys.stderr)
        return 1

    print("OK: every checked graph has all required edge features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
