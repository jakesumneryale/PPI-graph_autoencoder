"""Audit committed Voronoi features and build a balanced, reduced HDF5 dataset.

The source graph files are never modified.  A model is eligible for training only
when its committed Voronoi feature is finite and has shape (number of contacts, 1).
Checkpoint files are cross-checked when available, but are not required: the
committed source graph is the training-time source of truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np

from voronoi_edge_features.common import checkpoint_model_is_complete


DEFAULT_DATA_DIR = Path("/home/jas485/project_pi_co54/jas485/ppi_processed_graphs")
DEFAULT_CHECKPOINT_DIR = Path("voronoi_edge_features_data/contact_area_hdf5")
DEFAULT_OUTPUT_DIR = Path("voronoi_dataset_audit")
UNIFORM_PATTERN = re.compile(
    r"^complex\.(?P<run>\d{1,2})_(?P<bin>\d{1,2})_(?P<model>\d{1,2})(?:_corrected)?$"
)
RANDOM_PATTERN = re.compile(r"^complex\.(?P<run>\d{1,5})_(?P<model>\d+)(?:_corrected)?$")


@dataclass
class AuditRow:
    target: str
    model: str
    usable: bool
    model_type: str
    quality_bin: str
    reason: str
    contact_edges: int
    checkpoint_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-dir", type=Path, help="Default: OUTPUT_DIR/subset_hdf5")
    parser.add_argument("--feature-name", default="voronoi_contact_area")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20250909)
    parser.add_argument("--audit-only", action="store_true", help="Do not create reduced HDF5 files.")
    return parser.parse_args()


def classify_model(name: str) -> tuple[str, str, str]:
    match = UNIFORM_PATTERN.fullmatch(name)
    if match:
        quality_bin = int(match.group("bin"))
        if not 0 <= quality_bin <= 19:
            return "uniform_invalid_bin", str(quality_bin), "uniform_invalid_bin"
        return "uniform", str(quality_bin), f"uniform_bin_{quality_bin:02d}"
    if RANDOM_PATTERN.fullmatch(name):
        return "random", "", "random"
    return "unrecognized", "", "unrecognized"


def checkpoint_path(checkpoint_dir: Path, target: str) -> Path:
    return checkpoint_dir / f"{target}_voronoi_contact_areas.hdf5"


def audit_target(
    graph_path: Path,
    checkpoint_dir: Path,
    feature_name: str,
) -> list[AuditRow]:
    """Audit one target while keeping all HDF5 handles local to the scan."""
    target = graph_path.stem
    rows: list[AuditRow] = []
    cp_path = checkpoint_path(checkpoint_dir, target)
    checkpoint = None
    checkpoint_open_error = ""
    if cp_path.exists():
        try:
            checkpoint = h5py.File(cp_path, "r")
        except OSError as exc:
            checkpoint_open_error = f"unreadable: {exc}"

    try:
        with h5py.File(graph_path, "r") as graph:
            for model in sorted(graph.keys()):
                model_type, quality_bin, _ = classify_model(model)
                reason = "ok"
                edge_count = 0
                feature_values = None
                group = graph[model]
                if not isinstance(group, h5py.Group):
                    reason = "root entry is not a group"
                elif "edge_features" not in group:
                    reason = "missing edge_features group"
                else:
                    edges = group["edge_features"]
                    if "contacts" not in edges:
                        reason = "missing contacts dataset"
                    else:
                        contacts = edges["contacts"]
                        edge_count = len(contacts) if contacts.ndim else 0
                        if contacts.ndim != 2 or contacts.shape[1] != 2:
                            reason = f"invalid contacts shape {contacts.shape}"
                        elif feature_name not in edges:
                            reason = f"missing {feature_name}"
                        else:
                            feature_values = edges[feature_name][()]
                            if feature_values.shape != (edge_count, 1):
                                reason = f"invalid feature shape {feature_values.shape}; expected ({edge_count}, 1)"
                            elif not np.issubdtype(feature_values.dtype, np.number):
                                reason = f"non-numeric feature dtype {feature_values.dtype}"
                            elif not np.isfinite(feature_values).all():
                                reason = "feature contains NaN or infinity"

                if checkpoint_open_error:
                    cp_status = checkpoint_open_error
                elif checkpoint is None:
                    cp_status = "missing_file"
                elif model not in checkpoint:
                    cp_status = "missing_group"
                else:
                    try:
                        cp_complete = checkpoint_model_is_complete(
                            checkpoint[model], expected_graph_edges=edge_count
                        )
                        if not cp_complete:
                            cp_status = "incomplete_or_shape_mismatch"
                        elif feature_values is not None:
                            cp_values = checkpoint[model]["graph_contact_area"][()]
                            cp_status = (
                                "match" if np.array_equal(feature_values, cp_values) else "value_mismatch"
                            )
                        else:
                            cp_status = "complete"
                    except (KeyError, TypeError, ValueError, OSError) as exc:
                        cp_status = f"invalid: {exc}"

                rows.append(
                    AuditRow(
                        target=target,
                        model=model,
                        usable=reason == "ok",
                        model_type=model_type,
                        quality_bin=quality_bin,
                        reason=reason,
                        contact_edges=edge_count,
                        checkpoint_status=cp_status,
                    )
                )
    finally:
        if checkpoint is not None:
            checkpoint.close()
    return rows


def stable_rank(seed: int, target: str, model: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{target}\0{model}".encode("utf-8")).digest()


def allocate_stratified_counts(strata: dict[str, list[str]], fraction: float) -> dict[str, int]:
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


def select_subset(rows: list[AuditRow], fraction: float, seed: int) -> list[str]:
    strata: dict[str, list[str]] = defaultdict(list)
    target = rows[0].target if rows else ""
    for row in rows:
        if not row.usable:
            continue
        _, _, stratum = classify_model(row.model)
        strata[stratum].append(row.model)
    counts = allocate_stratified_counts(strata, fraction)
    selected = []
    for stratum, names in sorted(strata.items()):
        ranked = sorted(names, key=lambda name: (stable_rank(seed, target, name), name))
        selected.extend(ranked[: counts[stratum]])
    return sorted(selected)


def write_subset(source_path: Path, destination_path: Path, selected: list[str]) -> None:
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
            destination.attrs["voronoi_subset_source"] = str(source_path.resolve())
            destination.attrs["voronoi_subset_model_count"] = len(selected)
            for model in selected:
                source.copy(model, destination, name=model)
            destination.flush()
        os.replace(temporary_path, destination_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise SystemExit("--fraction must be greater than 0 and at most 1")
    if not args.data_dir.is_dir():
        raise SystemExit(f"Graph data directory does not exist: {args.data_dir}")

    graph_paths = sorted({*args.data_dir.glob("*.hdf5"), *args.data_dir.glob("*.h5")})
    if not graph_paths:
        raise SystemExit(f"No .hdf5 or .h5 files found in {args.data_dir}")
    path_by_target = defaultdict(list)
    for path in graph_paths:
        path_by_target[path.stem].append(path)
    duplicates = {target: paths for target, paths in path_by_target.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(f"{target}: {', '.join(map(str, paths))}" for target, paths in duplicates.items())
        raise SystemExit(f"Multiple HDF5 files resolve to the same target; remove the ambiguity: {details}")

    output_dir = args.output_dir.resolve()
    success_dir = output_dir / "successful_models"
    subset_dir = (args.subset_dir or output_dir / "subset_hdf5").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    success_dir.mkdir(parents=True, exist_ok=True)
    if not args.audit_only:
        subset_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[AuditRow] = []
    target_summaries = []
    subset_manifest = []
    for index, graph_path in enumerate(graph_paths, start=1):
        target_rows = audit_target(graph_path, args.checkpoint_dir, args.feature_name)
        all_rows.extend(target_rows)
        usable_names = sorted(row.model for row in target_rows if row.usable)
        (success_dir / f"{graph_path.stem}.txt").write_text(
            "".join(f"{name}\n" for name in usable_names), encoding="utf-8"
        )
        selected = select_subset(target_rows, args.fraction, args.seed)
        if not args.audit_only:
            write_subset(graph_path, subset_dir / graph_path.name, selected)
        for model in selected:
            model_type, quality_bin, stratum = classify_model(model)
            subset_manifest.append(
                {"target": graph_path.stem, "model": model, "model_type": model_type,
                 "quality_bin": quality_bin, "stratum": stratum}
            )
        reasons = Counter(row.reason for row in target_rows if not row.usable)
        target_summaries.append(
            {
                "target": graph_path.stem,
                "total_models": len(target_rows),
                "usable_models": len(usable_names),
                "attrition_models": len(target_rows) - len(usable_names),
                "retention_percent": round(100 * len(usable_names) / len(target_rows), 3) if target_rows else 0,
                "subset_models": len(selected),
                "attrition_reasons": json.dumps(dict(sorted(reasons.items())), sort_keys=True),
            }
        )
        print(
            f"[{index}/{len(graph_paths)}] {graph_path.stem}: "
            f"{len(usable_names)}/{len(target_rows)} usable; {len(selected)} selected"
        )

    audit_fields = list(AuditRow.__dataclass_fields__)
    write_csv(output_dir / "model_audit.csv", [asdict(row) for row in all_rows], audit_fields)
    write_csv(output_dir / "target_attrition.csv", target_summaries, list(target_summaries[0]))
    manifest_fields = ["target", "model", "model_type", "quality_bin", "stratum"]
    write_csv(output_dir / "subset_manifest.csv", subset_manifest, manifest_fields)

    total = len(all_rows)
    usable = sum(row.usable for row in all_rows)
    summary = {
        "data_dir": str(args.data_dir.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "feature_name": args.feature_name,
        "fraction": args.fraction,
        "seed": args.seed,
        "target_count": len(graph_paths),
        "total_models": total,
        "usable_models": usable,
        "attrition_models": total - usable,
        "subset_models": len(subset_manifest),
        "subset_dir": None if args.audit_only else str(subset_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Finished: {usable}/{total} models usable; reports written to {output_dir}")


if __name__ == "__main__":
    main()
