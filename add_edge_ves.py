"""Add per-edge electrostatic potential energy V_es to graph HDF5 files.

For edge ``i``, ``v_es[i]`` is the electrostatic interaction energy for that
contact, read directly from the per-target CSV already produced by
``nb_scripts/create_graph_edge_ves.py`` (one CSV per target, named
``{target}_edge_ves.csv``, with one row per decoy indexed by decoy name and an
``Edge_Ves`` column holding a Python-list-literal string). List order in that
column already matches ``edge_features/contacts`` order exactly (verified
against the 1acb example data and its HDF5 graph, across decoys with several
different edge counts) -- no re-indexing is needed, just a list-order copy.

``v_es`` is a screened, decoy-scale electrostatic term computed from PQR
partial charges. It is a different quantity from ``V_ae``, the APBS-derived
electrostatic feature named in the dissertation's planned (and still blocked)
fifth ladder rung -- the two should not be conflated.

A value of exactly 0 in the source data marks a non-interface edge (a real
computed value), not missing data, so no separate missing-value mask is
written, unlike ``voronoi_contact_missing``.

Existing valid datasets are skipped, making the operation resumable.
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import h5py
import numpy as np


FEATURE_NAME = "v_es"
TEMPORARY_NAME = f"__in_progress__{FEATURE_NAME}"
VALUE_COLUMN = "Edge_Ves"

CLUSTER_VES_DIR = Path("/home/jas485/project_pi_co54/jas485/ppi_decoy_interface_edge_charge")


def default_ves_dir(cluster: bool) -> Path | None:
    return CLUSTER_VES_DIR if cluster else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing target .hdf5/.h5 files.")
    parser.add_argument(
        "--ves-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing one '<target>_edge_ves.csv' per target. Defaults to the "
            "read/write cluster location when --cluster is given."
        ),
    )
    parser.add_argument("--cluster", action="store_true", help="Use the cluster default V_es source directory.")
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


def load_ves_by_decoy(csv_path: Path) -> dict[str, str]:
    """Map decoy name -> raw Edge_Ves list-literal string, parsed lazily per decoy."""
    with csv_path.open(newline="", encoding="ascii") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        decoy_column = "" if "" in fieldnames else fieldnames[0] if fieldnames else None
        if decoy_column is None or VALUE_COLUMN not in fieldnames:
            raise ValueError(f"{csv_path} must have a decoy-name column and {VALUE_COLUMN!r}; found {fieldnames}")
        return {row[decoy_column].strip(): row[VALUE_COLUMN] for row in reader if row[decoy_column].strip()}


def parse_v_es(raw_values: str, num_edges: int, source: str) -> np.ndarray:
    parsed = ast.literal_eval(raw_values)
    array = np.asarray(parsed, dtype=np.float32)
    if array.ndim != 1 or len(array) != num_edges:
        raise ValueError(f"{source}: {VALUE_COLUMN} has {len(array)} entries, graph has {num_edges} edges")
    if not np.isfinite(array).all():
        raise ValueError(f"{source}: {VALUE_COLUMN} contains NaN or infinite values")
    return array[:, None]


def dataset_is_valid(dataset: h5py.Dataset, num_edges: int) -> bool:
    if dataset.shape != (num_edges, 1) or not np.issubdtype(dataset.dtype, np.floating):
        return False
    return bool(np.isfinite(dataset[()]).all())


def process_file(
    path: Path,
    ves_dir: Path,
    overwrite: bool,
    allowed_models: set[str] | None = None,
) -> dict[str, object]:
    target = path.stem
    ves_csv_path = ves_dir / f"{target}_edge_ves.csv"
    total = written = skipped = failed = 0
    errors: list[str] = []

    if not ves_csv_path.is_file():
        raise FileNotFoundError(f"No V_es CSV found at {ves_csv_path}")
    ves_by_decoy = load_ves_by_decoy(ves_csv_path)

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
                edge_features = model["edge_features"]
                num_edges = edge_features["contacts"].shape[0]

                if TEMPORARY_NAME in edge_features:
                    del edge_features[TEMPORARY_NAME]
                if FEATURE_NAME in edge_features and not overwrite:
                    if dataset_is_valid(edge_features[FEATURE_NAME], num_edges):
                        skipped += 1
                        continue
                    del edge_features[FEATURE_NAME]

                if model_name not in ves_by_decoy:
                    raise KeyError(f"{model_name!r} not found in {ves_csv_path}")
                values = parse_v_es(ves_by_decoy[model_name], num_edges, f"{ves_csv_path}:{model_name}")

                temporary = edge_features.create_dataset(TEMPORARY_NAME, data=values, dtype=np.float32)
                temporary.attrs["definition"] = "per-edge screened electrostatic potential energy, V_es"
                temporary.attrs["source"] = str(ves_csv_path)
                handle.flush()
                if not dataset_is_valid(temporary, num_edges):
                    raise RuntimeError("new dataset failed validation")
                if FEATURE_NAME in edge_features:
                    del edge_features[FEATURE_NAME]
                edge_features.move(TEMPORARY_NAME, FEATURE_NAME)
                handle.flush()
                written += 1
            except Exception as exc:  # noqa: BLE001 - record each bad model and continue.
                failed += 1
                errors.append(f"{model_name}: {exc}")
                model = handle.get(model_name)
                if isinstance(model, h5py.Group) and "edge_features" in model:
                    edge_features = model["edge_features"]
                    if TEMPORARY_NAME in edge_features:
                        del edge_features[TEMPORARY_NAME]
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

    ves_dir = args.ves_dir or default_ves_dir(args.cluster)
    if ves_dir is None:
        raise SystemExit("--ves-dir is required unless --cluster is given.")
    if not ves_dir.is_dir():
        raise SystemExit(f"V_es source directory not found: {ves_dir}")

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
        row = process_file(path, ves_dir, overwrite=args.overwrite, allowed_models=allowed_models)
        rows.append(row)
        if args.log_every and (index % args.log_every == 0 or row["failed_models"]):
            print(
                f"[{index}/{len(paths)}] {row['target']}: {row['written_models']} written, "
                f"{row['skipped_valid_models']} already valid, {row['failed_models']} failed"
            )

    report_path = args.report or args.data_dir / "v_es_summary.csv"
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
        raise SystemExit(f"v_es failed for {failures} models; training dependency will not run.")


if __name__ == "__main__":
    main()
