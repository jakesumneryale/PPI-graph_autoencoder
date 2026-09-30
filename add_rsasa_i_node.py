"""Add per-residue rSASA_i to graph HDF5 files as a node feature.

For node ``i``, ``rsasa_i_node[i]`` is the ``rSASA_i_complex`` value for that
residue, read directly from the per-decoy CSV already produced by
``calc_rsasa.py``'s ``calc_rsasa_i()`` step (one CSV per decoy, named
``{decoy_name}.csv``, inside a ``{target}_rSASA/`` directory). Row order in
that CSV is ascending ``residue_ind`` with chains grouped contiguously, which
already matches graph node order exactly (verified against the 1acb example
data and its HDF5 graph) -- no re-indexing is needed, just a row-order copy.

This is a distinct feature from ``rsasa_i`` (the existing per-decoy graph-level
average, loaded from ``*_avg_rSASA_i.csv`` via ``OPTIONAL_NODE_FEATURE_SPECS``
in ``protein_hdf5_dataset.py``): that mechanism broadcasts one scalar to every
node, whereas this one is a genuine per-node value. Keeping the names distinct
avoids silently changing what ``rsasa_i`` has meant in prior runs/checkpoints.

Existing valid datasets are skipped, making the operation resumable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


FEATURE_NAME = "rsasa_i_node"
TEMPORARY_NAME = f"__in_progress__{FEATURE_NAME}"
VALUE_COLUMN = "rSASA_i_complex"
CHAIN_COLUMN = "chain_id"

CLUSTER_RSASA_DIR = Path("/nfs/roberts/pi/pi_co54/nb685/scratch_backup/rSASA")


def default_rsasa_dir(cluster: bool) -> Path | None:
    return CLUSTER_RSASA_DIR if cluster else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing target .hdf5/.h5 files.")
    parser.add_argument(
        "--rsasa-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing one '<target>_rSASA/' subdirectory per target, each holding "
            "one '<decoy_name>.csv' per decoy. Defaults to the read-only cluster location "
            "when --cluster is given."
        ),
    )
    parser.add_argument("--cluster", action="store_true", help="Use the cluster default rSASA source directory.")
    parser.add_argument("--target", help="Restrict to one target (its .hdf5/.h5 file stem).")
    parser.add_argument("--report", type=Path, help="Optional per-target summary CSV path.")
    parser.add_argument(
        "--model-list-dir",
        type=Path,
        help="Optional directory of <target>.txt model lists; unlisted groups are left unchanged.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute even valid existing datasets.")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def run_lengths(values) -> list[int]:
    """Lengths of consecutive equal-value runs, e.g. AAABB -> [3, 2]."""
    lengths: list[int] = []
    previous = object()
    for value in values:
        if value == previous:
            lengths[-1] += 1
        else:
            lengths.append(1)
            previous = value
    return lengths


def load_rsasa_i_node(csv_path: Path, num_nodes: int, chain_reference) -> np.ndarray:
    """Read ``rSASA_i_complex`` in row order, validated against the graph's chain layout."""
    values: list[float] = []
    chains: list[str] = []
    with csv_path.open(newline="", encoding="ascii") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or VALUE_COLUMN not in reader.fieldnames:
            raise ValueError(f"{csv_path} is missing column {VALUE_COLUMN!r}; found {reader.fieldnames}")
        for row in reader:
            values.append(float(row[VALUE_COLUMN]))
            chains.append(row.get(CHAIN_COLUMN, ""))

    if len(values) != num_nodes:
        raise ValueError(f"{csv_path} has {len(values)} residue rows, graph has {num_nodes} nodes")

    if run_lengths(chains) != run_lengths(chain_reference):
        raise ValueError(
            f"{csv_path} chain layout {run_lengths(chains)} does not match graph chain layout "
            f"{run_lengths(chain_reference)}"
        )

    array = np.asarray(values, dtype=np.float32)[:, None]
    if not np.isfinite(array).all():
        raise ValueError(f"{csv_path} contains NaN or infinite {VALUE_COLUMN} values")
    return array


def dataset_is_valid(dataset: h5py.Dataset, num_nodes: int) -> bool:
    if dataset.shape != (num_nodes, 1) or not np.issubdtype(dataset.dtype, np.floating):
        return False
    return bool(np.isfinite(dataset[()]).all())


def process_file(
    path: Path,
    rsasa_dir: Path,
    overwrite: bool,
    allowed_models: set[str] | None = None,
) -> dict[str, object]:
    target = path.stem
    target_rsasa_dir = rsasa_dir / f"{target}_rSASA"
    total = written = skipped = failed = 0
    errors: list[str] = []

    with h5py.File(path, "r+") as handle:
        if allowed_models is not None:
            missing = allowed_models - set(handle.keys())
            if missing:
                examples = ", ".join(sorted(missing)[:5])
                raise KeyError(f"{path} is missing {len(missing)} listed model(s); examples: {examples}")
        for model_name in sorted(handle.keys()):
            if allowed_models is not None and model_name not in allowed_models:
                continue
            total += 1
            try:
                model = handle[model_name]
                node_features = model["node_features"]
                num_nodes = node_features["chain"].shape[0]

                if TEMPORARY_NAME in node_features:
                    del node_features[TEMPORARY_NAME]
                if FEATURE_NAME in node_features and not overwrite:
                    if dataset_is_valid(node_features[FEATURE_NAME], num_nodes):
                        skipped += 1
                        continue
                    del node_features[FEATURE_NAME]

                csv_path = target_rsasa_dir / f"{model_name}.csv"
                if not csv_path.is_file():
                    raise FileNotFoundError(f"No rSASA CSV found at {csv_path}")

                chain_reference = np.asarray(node_features["chain"][()]).reshape(-1).tolist()
                values = load_rsasa_i_node(csv_path, num_nodes, chain_reference)

                temporary = node_features.create_dataset(TEMPORARY_NAME, data=values, dtype=np.float32)
                temporary.attrs["definition"] = f"per-residue {VALUE_COLUMN}, row-order-aligned to graph nodes"
                temporary.attrs["source"] = str(csv_path)
                handle.flush()
                if not dataset_is_valid(temporary, num_nodes):
                    raise RuntimeError("new dataset failed validation")
                if FEATURE_NAME in node_features:
                    del node_features[FEATURE_NAME]
                node_features.move(TEMPORARY_NAME, FEATURE_NAME)
                handle.flush()
                written += 1
            except Exception as exc:  # noqa: BLE001 - record each bad model and continue.
                failed += 1
                errors.append(f"{model_name}: {exc}")
                model = handle.get(model_name)
                if isinstance(model, h5py.Group) and "node_features" in model:
                    node_features = model["node_features"]
                    if TEMPORARY_NAME in node_features:
                        del node_features[TEMPORARY_NAME]
                        handle.flush()

    return {
        "target": target,
        "hdf5_path": str(path.resolve()),
        "total_models": total,
        "written_models": written,
        "skipped_valid_models": skipped,
        "failed_models": failed,
        "errors": " | ".join(errors),
    }


def resolve_target_file(data_dir: Path, target: str) -> Path:
    for suffix in (".hdf5", ".h5"):
        candidate = data_dir / f"{target}{suffix}"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"No {target}.hdf5 or {target}.h5 found in {data_dir}")


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {args.data_dir}")

    rsasa_dir = args.rsasa_dir or default_rsasa_dir(args.cluster)
    if rsasa_dir is None:
        raise SystemExit("--rsasa-dir is required unless --cluster is given.")
    if not rsasa_dir.is_dir():
        raise SystemExit(f"rSASA source directory not found: {rsasa_dir}")

    if args.target:
        paths = [resolve_target_file(args.data_dir, args.target)]
    else:
        paths = sorted({*args.data_dir.glob("*.hdf5"), *args.data_dir.glob("*.h5")})
    if not paths:
        raise SystemExit(f"No HDF5 files found in {args.data_dir}")

    rows = []
    for index, path in enumerate(paths, start=1):
        allowed_models = None
        if args.model_list_dir is not None:
            list_path = args.model_list_dir / f"{path.stem}.txt"
            if not list_path.is_file():
                raise SystemExit(f"No model list found for {path.stem}: {list_path}")
            allowed_models = {
                line.strip().split("\t", 1)[0]
                for line in list_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        row = process_file(path, rsasa_dir, overwrite=args.overwrite, allowed_models=allowed_models)
        rows.append(row)
        if args.log_every and (index % args.log_every == 0 or row["failed_models"]):
            print(
                f"[{index}/{len(paths)}] {row['target']}: {row['written_models']} written, "
                f"{row['skipped_valid_models']} already valid, {row['failed_models']} failed"
            )

    report_path = args.report or args.data_dir / "rsasa_i_node_summary.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(int(row["failed_models"]) for row in rows)
    total = sum(int(row["total_models"]) for row in rows)
    complete = sum(int(row["written_models"]) + int(row["skipped_valid_models"]) for row in rows)
    print(f"Validated {complete}/{total} models. Report: {report_path}")
    if failures:
        raise SystemExit(f"rsasa_i_node failed for {failures} models; training dependency will not run.")


if __name__ == "__main__":
    main()
