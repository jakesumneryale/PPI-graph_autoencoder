"""Add the number of unique interface-node neighbors to graph HDF5 files.

For node ``i``, ``interface_node_degree[i]`` is the number of distinct nodes
``j`` for which an edge ``(i, j)`` exists and ``interface_nodes[j]`` is true.
Contacts are treated as undirected and duplicate/reversed edges are counted
only once. Existing valid datasets are skipped, making the operation resumable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


FEATURE_NAME = "interface_node_degree"
TEMPORARY_NAME = f"__in_progress__{FEATURE_NAME}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing target .hdf5/.h5 files.")
    parser.add_argument("--report", type=Path, help="Optional per-target summary CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even valid existing datasets.")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def normalize_contacts(contacts: np.ndarray, num_nodes: int) -> np.ndarray:
    contacts = np.asarray(contacts)
    if contacts.ndim != 2 or contacts.shape[1] != 2:
        raise ValueError(f"contacts must have shape [num_edges, 2], got {contacts.shape}")
    if not np.issubdtype(contacts.dtype, np.integer):
        if not np.isfinite(contacts).all() or not np.equal(contacts, np.floor(contacts)).all():
            raise ValueError("contacts contains non-integer node indices")
    contacts = contacts.astype(np.int64, copy=False)
    if not len(contacts):
        return contacts
    if contacts.min() >= 0 and contacts.max() < num_nodes:
        return contacts
    if contacts.min() >= 1 and contacts.max() == num_nodes:
        return contacts - 1
    raise ValueError(
        f"contact indices [{contacts.min()}, {contacts.max()}] are incompatible with {num_nodes} nodes"
    )


def calculate_interface_node_degree(interface_nodes: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    interface_nodes = np.asarray(interface_nodes)
    if interface_nodes.ndim == 2 and interface_nodes.shape[1] == 1:
        interface_nodes = interface_nodes[:, 0]
    if interface_nodes.ndim != 1:
        raise ValueError(f"interface_nodes must have shape [num_nodes] or [num_nodes, 1], got {interface_nodes.shape}")
    if not np.issubdtype(interface_nodes.dtype, np.number):
        raise ValueError(f"interface_nodes must be numeric, got {interface_nodes.dtype}")
    if not np.isfinite(interface_nodes).all():
        raise ValueError("interface_nodes contains NaN or infinity")

    num_nodes = len(interface_nodes)
    contacts = normalize_contacts(contacts, num_nodes)
    is_interface = interface_nodes != 0
    interface_neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, destination in contacts:
        source = int(source)
        destination = int(destination)
        if source == destination:
            continue
        if is_interface[destination]:
            interface_neighbors[source].add(destination)
        if is_interface[source]:
            interface_neighbors[destination].add(source)

    return np.asarray([len(neighbors) for neighbors in interface_neighbors], dtype=np.int32)[:, None]


def dataset_is_valid(dataset: h5py.Dataset, num_nodes: int) -> bool:
    if dataset.shape != (num_nodes, 1) or not np.issubdtype(dataset.dtype, np.integer):
        return False
    values = dataset[()]
    return bool((values >= 0).all() and (values <= num_nodes).all())


def process_file(path: Path, overwrite: bool) -> dict[str, object]:
    total = written = skipped = failed = 0
    errors: list[str] = []
    with h5py.File(path, "r+") as handle:
        for model_name in sorted(handle.keys()):
            total += 1
            try:
                model = handle[model_name]
                node_features = model["node_features"]
                edge_features = model["edge_features"]
                interface_nodes = node_features["interface_nodes"][()]
                num_nodes = len(interface_nodes)

                if TEMPORARY_NAME in node_features:
                    del node_features[TEMPORARY_NAME]
                if FEATURE_NAME in node_features and not overwrite:
                    if dataset_is_valid(node_features[FEATURE_NAME], num_nodes):
                        skipped += 1
                        continue
                    del node_features[FEATURE_NAME]

                values = calculate_interface_node_degree(interface_nodes, edge_features["contacts"][()])
                temporary = node_features.create_dataset(TEMPORARY_NAME, data=values, dtype=np.int32)
                temporary.attrs["definition"] = "number of unique adjacent nodes whose interface_nodes value is nonzero"
                temporary.attrs["contacts_treated_as_undirected"] = True
                handle.flush()
                if not dataset_is_valid(temporary, num_nodes):
                    raise RuntimeError("new dataset failed validation")
                if FEATURE_NAME in node_features:
                    del node_features[FEATURE_NAME]
                node_features.move(TEMPORARY_NAME, FEATURE_NAME)
                handle.flush()
                written += 1
            except Exception as exc:  # noqa: BLE001 - record each malformed graph and continue the audit.
                failed += 1
                errors.append(f"{model_name}: {exc}")
                model = handle.get(model_name)
                if isinstance(model, h5py.Group) and "node_features" in model:
                    node_features = model["node_features"]
                    if TEMPORARY_NAME in node_features:
                        del node_features[TEMPORARY_NAME]
                        handle.flush()

    return {
        "target": path.stem,
        "hdf5_path": str(path.resolve()),
        "total_models": total,
        "written_models": written,
        "skipped_valid_models": skipped,
        "failed_models": failed,
        "errors": " | ".join(errors),
    }


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {args.data_dir}")
    paths = sorted({*args.data_dir.glob("*.hdf5"), *args.data_dir.glob("*.h5")})
    if not paths:
        raise SystemExit(f"No HDF5 files found in {args.data_dir}")

    rows = []
    for index, path in enumerate(paths, start=1):
        row = process_file(path, overwrite=args.overwrite)
        rows.append(row)
        if args.log_every and (index % args.log_every == 0 or row["failed_models"]):
            print(
                f"[{index}/{len(paths)}] {row['target']}: {row['written_models']} written, "
                f"{row['skipped_valid_models']} already valid, {row['failed_models']} failed"
            )

    report_path = args.report or args.data_dir / "interface_node_degree_summary.csv"
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
        raise SystemExit(f"Interface-node degree failed for {failures} models; training dependency will not run.")


if __name__ == "__main__":
    main()
