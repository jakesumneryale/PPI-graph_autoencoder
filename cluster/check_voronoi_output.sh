#!/usr/bin/env bash
# Read-only progress/integrity check for one target's Voronoi feature data.
# Usage: bash cluster/check_voronoi_output.sh TARGET [MODEL_GROUP]
# Example: bash cluster/check_voronoi_output.sh 1acb complex.0_0_11
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
GRAPH_DATA_DIR="${GRAPH_DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
FEATURE_NAME="${FEATURE_NAME:-voronoi_contact_area}"
TARGET_NAME="${1:-}"
MODEL_NAME="${2:-}"

if [[ -z "$TARGET_NAME" ]]; then
  echo "Usage: bash cluster/check_voronoi_output.sh TARGET [MODEL_GROUP]" >&2
  exit 2
fi

PYTHON="${VORONOI_PYTHON:-$PROJECT_DIR/venv/bin/python}"
CHECKPOINT="$PROJECT_DIR/voronoi_edge_features_data/contact_area_hdf5/${TARGET_NAME}_voronoi_contact_areas.hdf5"
REFERENCE_CSV="$PROJECT_DIR/voronoi_edge_features_data/model_references/${TARGET_NAME}.csv"
MARKER="$PROJECT_DIR/voronoi_run/committed/${TARGET_NAME}.done"

if [[ -f "$GRAPH_DATA_DIR/${TARGET_NAME}.hdf5" ]]; then
  GRAPH_HDF5="$GRAPH_DATA_DIR/${TARGET_NAME}.hdf5"
elif [[ -f "$GRAPH_DATA_DIR/${TARGET_NAME}.h5" ]]; then
  GRAPH_HDF5="$GRAPH_DATA_DIR/${TARGET_NAME}.h5"
else
  echo "ERROR: no graph HDF5 for $TARGET_NAME in $GRAPH_DATA_DIR" >&2
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: project Python not found at $PYTHON" >&2
  exit 2
fi

echo "Target:       $TARGET_NAME"
echo "Graph HDF5:  $GRAPH_HDF5"
echo "Checkpoint:  $CHECKPOINT"
echo "References:  $REFERENCE_CSV"
echo "Done marker: $MARKER"
echo

"$PYTHON" - "$GRAPH_HDF5" "$CHECKPOINT" "$REFERENCE_CSV" "$MARKER" "$FEATURE_NAME" "$MODEL_NAME" <<'PY'
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sys

import h5py
import numpy as np


graph_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
reference_path = Path(sys.argv[3])
marker_path = Path(sys.argv[4])
feature_name = sys.argv[5]
requested_model = sys.argv[6]


def file_info(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    changed = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
    return f"{stat.st_size / 1024**2:.1f} MiB; modified {changed}"


reference_names: list[str] = []
if reference_path.exists():
    with reference_path.open(newline="") as handle:
        reference_names = [row["graph_group_name"] for row in csv.DictReader(handle)]

print("FILES")
print(f"  source graph: {file_info(graph_path)}")
print(f"  checkpoint:   {file_info(checkpoint_path)}")
print(f"  reference CSV:{' present' if reference_path.exists() else ' missing'}")
print(f"  completion:   {'DONE' if marker_path.is_file() else 'not complete'}")
print()

try:
    with h5py.File(graph_path, "r") as graph:
        graph_names = list(graph.keys())
        committed = []
        invalid = []
        for name in graph_names:
            edge_group = graph[name].get("edge_features")
            if edge_group is None or feature_name not in edge_group:
                continue
            values = edge_group[feature_name]
            contacts = edge_group.get("contacts")
            if contacts is None or values.shape[0] != contacts.shape[0] or values.ndim != 2 or values.shape[1] != 1:
                invalid.append(name)
            else:
                committed.append(name)

        expected = reference_names or graph_names
        expected_set = set(expected)
        committed_set = set(committed) & expected_set
        uncommitted = sorted(expected_set - committed_set)
        print("SOURCE GRAPH COMMIT PROGRESS")
        print(f"  valid feature datasets: {len(committed_set)}/{len(expected)}")
        print(f"  malformed datasets:     {len(invalid)}")
        if committed:
            print(f"  examples committed:      {', '.join(committed[:5])}")
        if uncommitted:
            print(f"  examples not committed:  {', '.join(uncommitted[:5])}")
        if invalid:
            print(f"  malformed examples:      {', '.join(invalid[:5])}")

        model_name = requested_model or (committed[0] if committed else "")
        graph_snapshot = None
        if model_name:
            if model_name not in graph:
                print(f"\nMODEL SPOT CHECK\n  {model_name}: not found in source graph")
            else:
                edges = graph[model_name].get("edge_features")
                if edges is None or feature_name not in edges:
                    print(f"\nMODEL SPOT CHECK\n  {model_name}: feature not committed yet")
                else:
                    values = edges[feature_name][()]
                    graph_snapshot = values
                    finite = np.isfinite(values)
                    print("\nMODEL SPOT CHECK")
                    print(f"  model:        {model_name}")
                    print(f"  path:         /{model_name}/edge_features/{feature_name}")
                    print(f"  shape/dtype:  {values.shape} / {values.dtype}")
                    print(f"  finite:       {int(finite.sum())}/{values.size}")
                    print(f"  nonzero:      {int(np.count_nonzero(values))}/{values.size}")
                    if values.size and finite.any():
                        print(f"  min/mean/max: {values[finite].min():.4f} / {values[finite].mean():.4f} / {values[finite].max():.4f} Å²")
except OSError as exc:
    print(f"ERROR: could not read source graph HDF5: {exc}")
    raise SystemExit(3)

print()
print("CHECKPOINT COMPUTE PROGRESS")
if not checkpoint_path.exists():
    print("  no checkpoint exists yet")
    raise SystemExit(0)

try:
    with h5py.File(checkpoint_path, "r") as checkpoint:
        computed = list(checkpoint.keys())
        expected_count = len(reference_names) if reference_names else len(graph_names)
        print(f"  computed model groups: {len(computed)}/{expected_count}")
        if computed:
            print(f"  examples computed:     {', '.join(computed[:5])}")

        model_name = requested_model or (committed[0] if committed else (computed[0] if computed else ""))
        if model_name and model_name in checkpoint:
            group = checkpoint[model_name]
            required = ("contact_pairs", "contact_area", "graph_contacts", "graph_contact_area", "graph_contact_missing_mask")
            missing = [name for name in required if name not in group]
            print(f"  {model_name} checkpoint datasets: {'OK' if not missing else 'MISSING ' + ', '.join(missing)}")
            if "contact_area" in group:
                areas = group["contact_area"][()]
                print(f"  residue contacts:      {len(areas)}; summed area {float(areas.sum()):.2f} Å²")
            if graph_snapshot is not None and "graph_contact_area" in group:
                checkpoint_values = group["graph_contact_area"][()]
                same = graph_snapshot.shape == checkpoint_values.shape and np.array_equal(graph_snapshot, checkpoint_values)
                print(f"  checkpoint == source: {'YES (exact match)' if same else 'NO'}")
except OSError as exc:
    print(f"  checkpoint could not be opened: {exc}")
    print("  This commonly means its job is actively writing it; retry after that task stops.")
PY
