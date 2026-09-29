"""Build a quality-stratified, reduced-size copy of the graph HDF5 dataset.

For every target, keeps a fixed fraction of that target's models, stratified
by DockQ-uniform-sampling bin (parsed from the model group name) so the
reduced set stays balanced across DockQ the same way the full set is. Source
files are never modified; each target gets one new, smaller HDF5 file with
every model group copied verbatim (all committed features included, so
whatever the full dataset has -- interface_node_degree, voronoi_contact_area,
voronoi_contact_missing, etc. -- carries over unchanged).

When --model-list-dir is given (the per-target .txt files written by
audit_voronoi_dataset.py's successful_models output), the candidate pool for
each target is restricted to that audit-passing set before subsampling, so
the resulting subset is self-consistent: every model it contains already has
a finite, correctly-shaped Voronoi contact-area feature, and a training run
against the subset does not need a separate --model-list-dir pass (which
would fail anyway, since a strict allowed-models check expects every listed
model to be present in the file, an invariant a subsample of the full
audit list cannot satisfy).

Selection is deterministic given --seed: each model gets a stable pseudo-random
rank from sha256(seed, target, model), so the same seed always picks the same
subset, and a different seed gives an independent subset for a robustness
check. Re-derived from the stratified-sampling logic already proven in
audit_voronoi_dataset.py, generalized to not require a Voronoi-specific
eligibility check -- this script assumes every model in the source file is
already eligible (the source directory is whatever full dataset the caller
points it at) and only subsamples.

Usage:
    python make_subset_dataset.py \
        --data-dir /path/to/full/graphs \
        --targets-file cluster/targets.txt \
        --output-dir /path/to/subset_10pct_hdf5 \
        --fraction 0.10 --seed 20250909
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py

UNIFORM_PATTERN = re.compile(
    r"^complex\.(?P<run>\d{1,2})_(?P<bin>\d{1,2})_(?P<model>\d{1,2})(?:_corrected)?$"
)
RANDOM_PATTERN = re.compile(r"^complex\.(?P<run>\d{1,5})_(?P<model>\d+)(?:_corrected)?$")


def classify_model(name: str) -> str:
    """DockQ-bin stratum key for one model group name, or 'other' if unparseable."""
    match = UNIFORM_PATTERN.fullmatch(name)
    if match:
        quality_bin = int(match.group("bin"))
        if 0 <= quality_bin <= 19:
            return f"uniform_bin_{quality_bin:02d}"
        return "uniform_invalid_bin"
    if RANDOM_PATTERN.fullmatch(name):
        return "random"
    return "other"


def stable_rank(seed: int, target: str, model: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{target}\0{model}".encode("utf-8")).digest()


def allocate_stratified_counts(strata: dict[str, list[str]], fraction: float) -> dict[str, int]:
    """Largest-remainder rounding so per-stratum counts sum to round(total * fraction)."""
    total = sum(len(names) for names in strata.values())
    desired_total = min(total, max(1, int(round(total * fraction)))) if total else 0
    raw = {key: len(names) * fraction for key, names in strata.items()}
    allocated = {key: min(len(strata[key]), int(value)) for key, value in raw.items()}
    remaining = desired_total - sum(allocated.values())
    order = sorted(strata, key=lambda key: (-(raw[key] - int(raw[key])), key))
    for key in order:
        if remaining <= 0:
            break
        if allocated[key] < len(strata[key]):
            allocated[key] += 1
            remaining -= 1
    return allocated


def select_subset(model_names: list[str], target: str, fraction: float, seed: int) -> list[str]:
    strata: dict[str, list[str]] = defaultdict(list)
    for name in model_names:
        strata[classify_model(name)].append(name)
    counts = allocate_stratified_counts(strata, fraction)
    selected: list[str] = []
    for stratum, names in sorted(strata.items()):
        ranked = sorted(names, key=lambda name: (stable_rank(seed, target, name), name))
        selected.extend(ranked[: counts[stratum]])
    return sorted(selected)


def write_subset(source_path: Path, destination_path: Path, selected: list[str], fraction: float, seed: int) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary_path, "w") as destination:
            for key, value in source.attrs.items():
                destination.attrs[key] = value
            destination.attrs["subset_source"] = str(source_path.resolve())
            destination.attrs["subset_fraction"] = fraction
            destination.attrs["subset_seed"] = seed
            destination.attrs["subset_model_count"] = len(selected)
            for model in selected:
                source.copy(model, destination, name=model)
            destination.flush()
        os.replace(temporary_path, destination_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def resolve_target_file(data_dir: Path, target: str) -> Path | None:
    for suffix in (".hdf5", ".h5"):
        candidate = data_dir / f"{target}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Full graph HDF5 directory (read-only).")
    parser.add_argument("--targets-file", type=Path, required=True, help="One target name per line.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where the reduced HDF5 files are written.")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20250909)
    parser.add_argument("--allow-missing-targets", action="store_true",
                        help="Skip a target with no source file instead of failing.")
    parser.add_argument("--model-list-dir", type=Path, default=None,
                        help="Directory of per-target <target>.txt audit-passing model lists "
                             "(e.g. voronoi_dataset_audit/successful_models). When given, only "
                             "models in this list are eligible for subsampling.")
    return parser.parse_args()


def load_allowed_models(model_list_dir: Path, target: str) -> set[str] | None:
    list_path = model_list_dir / f"{target}.txt"
    if not list_path.is_file():
        return None
    return {line.strip() for line in list_path.read_text().splitlines() if line.strip()}


def main() -> int:
    args = parse_args()
    targets = [line.strip() for line in args.targets_file.read_text().splitlines() if line.strip()]
    if not targets:
        sys.exit(f"No targets read from {args.targets_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    missing = []
    for target in targets:
        source_path = resolve_target_file(args.data_dir, target)
        if source_path is None:
            missing.append(target)
            if args.allow_missing_targets:
                continue
            sys.exit(f"No graph HDF5 for target {target!r} in {args.data_dir} (pass --allow-missing-targets to skip)")

        with h5py.File(source_path, "r") as handle:
            model_names = [name for name in handle if not name.startswith("__")]
        if not model_names:
            print(f"SKIP {target}: source file has no model groups", file=sys.stderr)
            continue

        if args.model_list_dir is not None:
            allowed = load_allowed_models(args.model_list_dir, target)
            if allowed is None:
                print(f"SKIP {target}: no audit list at {args.model_list_dir / f'{target}.txt'}",
                      file=sys.stderr)
                continue
            model_names = [name for name in model_names if name in allowed]
            if not model_names:
                print(f"SKIP {target}: no audit-passing models remain", file=sys.stderr)
                continue

        selected = select_subset(model_names, target, args.fraction, args.seed)
        destination_path = args.output_dir / source_path.name
        write_subset(source_path, destination_path, selected, args.fraction, args.seed)
        summary_rows.append({
            "target": target, "n_source": len(model_names), "n_subset": len(selected),
            "fraction_actual": round(len(selected) / len(model_names), 4),
        })
        print(f"{target}: {len(model_names)} -> {len(selected)} models "
              f"({len(selected) / len(model_names):.1%})")

    summary_path = args.output_dir / "subset_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "n_source", "n_subset", "fraction_actual"])
        writer.writeheader()
        writer.writerows(summary_rows)

    total_source = sum(r["n_source"] for r in summary_rows)
    total_subset = sum(r["n_subset"] for r in summary_rows)
    print(f"\n{len(summary_rows)} targets written to {args.output_dir}")
    print(f"Total models: {total_source} -> {total_subset} ({total_subset / max(total_source, 1):.1%})")
    print(f"Summary: {summary_path}")
    if missing:
        print(f"Skipped {len(missing)} target(s) with no source file: {missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
