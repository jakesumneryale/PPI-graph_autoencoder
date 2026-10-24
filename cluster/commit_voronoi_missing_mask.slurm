#!/usr/bin/env bash
#SBATCH --job-name=gate_commit_mask
#SBATCH --partition=pi_co54
#SBATCH --array=1-146%24
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# Commit voronoi_contact_missing into the graph HDF5 files, one target per task.
#
# No recomputation is involved: graph_contact_missing_mask is part of
# CHECKPOINT_MODEL_DATASETS, so every model that already passed the completeness
# check has it sitting in the per-target checkpoint HDF5 -- it was simply never
# copied across into the training graphs. This job does that copy.
#
# Safe to rerun. commit_graph_features treats a model as already committed only
# when BOTH the area and the mask are present, so finished targets are skipped
# and interrupted ones resume.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
GRAPH_DATA_DIR="${DATA_DIR:-${GRAPH_DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}}"
RUN_DIR="$PROJECT_DIR/voronoi_run"
TARGETS_FILE="$PROJECT_DIR/cluster/targets.txt"
MASK_MARKER_DIR="$RUN_DIR/committed_mask"
FEATURE_NAME="voronoi_contact_area"
MISSING_MASK_NAME="voronoi_contact_missing"

mkdir -p "$RUN_DIR/logs" "$MASK_MARKER_DIR"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi

TARGET_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TARGETS_FILE")
if [[ -z "$TARGET_NAME" ]]; then
  echo "No target at line ${SLURM_ARRAY_TASK_ID} of $TARGETS_FILE" >&2
  exit 1
fi

MARKER="$MASK_MARKER_DIR/${TARGET_NAME}.done"
if [[ -f "$MARKER" ]]; then
  echo "$TARGET_NAME: mask already committed; skipping."
  exit 0
fi

# 21 of the 146 targets have no models at all. Skipping them cleanly matters:
# the downstream jobs use afterok, so a spurious failure here would block the
# entire GPU array.
if [[ ! -f "$GRAPH_DATA_DIR/${TARGET_NAME}.hdf5" && ! -f "$GRAPH_DATA_DIR/${TARGET_NAME}.h5" ]]; then
  echo "SKIP: $TARGET_NAME has no graph HDF5 in $GRAPH_DATA_DIR"
  exit 0
fi

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

echo "=== [$(date)] Committing $MISSING_MASK_NAME for $TARGET_NAME ==="

set +e
python -m voronoi_edge_features.commit_graph_features \
  "$TARGET_NAME" \
  --data "$GRAPH_DATA_DIR" \
  --feature-name "$FEATURE_NAME" \
  --missing-mask-name "$MISSING_MASK_NAME"
COMMIT_STATUS=$?
set -e

case "$COMMIT_STATUS" in
  0)
    touch "$MARKER"
    echo "=== [$(date)] $TARGET_NAME committed. ==="
    ;;
  1)
    # No checkpoint HDF5 or no reference CSV: this target was never computed.
    # Not an error for this job; the training model lists exclude it anyway.
    echo "SKIP: $TARGET_NAME has nothing to commit (exit 1 from commit step)."
    ;;
  3)
    # Some models are computed and now committed, others were never computed.
    # The marker is deliberately withheld so a later pass can finish the target.
    echo "PARTIAL: $TARGET_NAME committed what exists; some models still uncomputed."
    ;;
  *)
    echo "ERROR: $TARGET_NAME commit failed with status $COMMIT_STATUS" >&2
    exit "$COMMIT_STATUS"
    ;;
esac
