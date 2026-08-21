#!/usr/bin/env bash
# Per-array-task worker: computes (and resumes) one target's Voronoi contact
# areas into the checkpoint HDF5, then atomically commits whatever is
# computed into the source graph HDF5. Safe to rerun/requeue at any point --
# the compute step skips models already in the checkpoint file, and the
# commit step only ever swaps in a fully-written temp copy of the graph file.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
GRAPH_DATA_DIR="${GRAPH_DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
PDB_BASE_DIR="${PDB_BASE_DIR:-/nfs/roberts/project/pi_co54/jas485/uniformly_sampled_target_data}"
RUN_DIR="$PROJECT_DIR/voronoi_run"
TARGETS_FILE="$PROJECT_DIR/cluster/targets.txt"
COMMIT_MARKER_DIR="$RUN_DIR/committed"
FEATURE_NAME="voronoi_contact_area"

mkdir -p "$RUN_DIR/logs" "$COMMIT_MARKER_DIR"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script is meant to be run as a SLURM array task (SLURM_ARRAY_TASK_ID unset)." >&2
  exit 1
fi

TARGET_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TARGETS_FILE")
if [[ -z "$TARGET_NAME" ]]; then
  echo "No target found at line ${SLURM_ARRAY_TASK_ID} of $TARGETS_FILE" >&2
  exit 1
fi

# A few graph targets may not yet have validated structure inputs. Since each
# array task is independent, skip an unavailable target cleanly. Deliberately
# do not create its completion marker, so adding the input later makes it
# eligible for a future submission.
TARGET_DIR="$PDB_BASE_DIR/$TARGET_NAME"
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "SKIP: $TARGET_NAME has no PDB target directory at $TARGET_DIR" >&2
  exit 0
fi
if [[ ! -f "$GRAPH_DATA_DIR/${TARGET_NAME}.hdf5" && ! -f "$GRAPH_DATA_DIR/${TARGET_NAME}.h5" ]]; then
  echo "SKIP: $TARGET_NAME has no graph HDF5 in $GRAPH_DATA_DIR" >&2
  exit 0
fi

MARKER="$COMMIT_MARKER_DIR/${TARGET_NAME}.done"
if [[ -f "$MARKER" ]]; then
  echo "$TARGET_NAME already fully committed; skipping."
  exit 0
fi

source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

echo "=== [$(date)] Phase A: compute (target=$TARGET_NAME, array_task=$SLURM_ARRAY_TASK_ID, node=${SLURM_JOB_NODELIST:-unknown}) ==="
python generate_voronoi_contact_area_data.py \
  "$TARGET_DIR" \
  --pdb-root "$PDB_BASE_DIR" \
  --data "$GRAPH_DATA_DIR" \
  --feature-name "$FEATURE_NAME" \
  --log-every 50

echo "=== [$(date)] Phase B: commit into graph HDF5 (target=$TARGET_NAME) ==="
python -m voronoi_edge_features.commit_graph_features \
  "$TARGET_NAME" \
  --data "$GRAPH_DATA_DIR" \
  --feature-name "$FEATURE_NAME" \
  --marker-path "$MARKER"

echo "=== [$(date)] $TARGET_NAME complete. ==="
