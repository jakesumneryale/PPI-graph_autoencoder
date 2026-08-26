#!/usr/bin/env bash
# Per-array-task worker: computes APBS surface electrostatics for every model
# PDB of one target into that target's HDF5 in the shared data store.
#
# Deliberately a single phase, unlike the Voronoi worker. There is no commit
# step because nothing is written back into the training graph HDF5 files --
# the electrostatics live entirely in their own store, and each model group is
# renamed into place only once fully written. Safe to rerun or requeue at any
# point: finished models are skipped.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
PDB_BASE_DIR="${PDB_BASE_DIR:-/nfs/roberts/pi/pi_co54/jas485/uniformly_sampled_target_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/roberts/pi/pi_co54/jas485/ppi_gnn_data_store/apbs_model_data}"
ENV_PREFIX="${APBS_ENV_PREFIX:-$PROJECT_DIR/apbs_env}"
RUN_DIR="$PROJECT_DIR/apbs_run"
TARGETS_FILE="$PROJECT_DIR/apbs_analysis/cluster/targets.txt"
DONE_MARKER_DIR="$RUN_DIR/completed"

mkdir -p "$RUN_DIR/logs" "$DONE_MARKER_DIR" "$OUTPUT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_DIR/matplotlib_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_DIR/cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# APBS's multigrid solver is single-threaded per elec block here; keeping the
# BLAS/OpenMP thread pools at one avoids oversubscribing the allocated cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script is meant to be run as a SLURM array task (SLURM_ARRAY_TASK_ID unset)." >&2
  exit 1
fi

TARGET_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TARGETS_FILE")
if [[ -z "$TARGET_NAME" ]]; then
  echo "No target found at line ${SLURM_ARRAY_TASK_ID} of $TARGETS_FILE" >&2
  exit 1
fi

# Targets whose structures have not landed yet are skipped cleanly without a
# marker, so adding the input later makes them eligible for a resubmission.
SAMPLED_DIR="$PDB_BASE_DIR/sampled_$TARGET_NAME"
if [[ ! -d "$SAMPLED_DIR" ]]; then
  echo "SKIP: $TARGET_NAME has no sampled PDB directory at $SAMPLED_DIR" >&2
  exit 0
fi

MARKER="$DONE_MARKER_DIR/${TARGET_NAME}.done"
if [[ -f "$MARKER" ]]; then
  echo "$TARGET_NAME already complete; skipping."
  exit 0
fi

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$PROJECT_DIR"

# Intermediates (PQR, APBS input, DX) are node-local so thousands of small
# temporary files never touch NFS. They are deleted after each model.
SCRATCH_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/apbs_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIR"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

echo "=== [$(date)] Environment preflight (target=$TARGET_NAME) ==="
python apbs_analysis/cluster/check_apbs_environment.py

echo "=== [$(date)] APBS surface electrostatics (target=$TARGET_NAME, array_task=$SLURM_ARRAY_TASK_ID, node=${SLURM_JOB_NODELIST:-unknown}) ==="
# Only atom- and residue-level results are stored: a per-point surface cloud is
# ~7 MB and a raw potential volume ~70 MB per model, which does not scale to
# thousands of models per target. Add --store-surface-points / --store-grid for
# a small, deliberately selected subset instead.
# --workers matches --cpus-per-task: APBS's solver is single-threaded, so
# concurrency has to come from running models side by side, not from threads.
#
# Run Python in the background and forward TERM to it by hand. SLURM's
# --signal=B:TERM sends only to this batch shell, and bash defers a trap until
# the foreground command finishes -- so with a plain foreground call the
# Python process would never see the signal and would simply be killed at the
# wall-clock limit, losing every model in flight. Backgrounding plus an
# explicit forward gives the runner its 120 s to finish and flush.
forward_term() {
  echo "=== [$(date)] TERM received; asking the runner to stop after the current model ==="
  kill -TERM "$RUNNER_PID" 2>/dev/null || true
}
trap forward_term TERM

python -m apbs_analysis.generate_apbs_surface_data \
  "$TARGET_NAME" \
  --pdb-root "$PDB_BASE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --scratch-dir "$SCRATCH_DIR" \
  --workers "${APBS_WORKERS:-${SLURM_CPUS_PER_TASK:-2}}" \
  --log-every 50 &
RUNNER_PID=$!

# `wait` returns immediately (>128) when the trap fires, so loop until the
# child has genuinely exited and its real status is in hand.
STATUS=0
wait "$RUNNER_PID" || STATUS=$?
while kill -0 "$RUNNER_PID" 2>/dev/null; do
  STATUS=0
  wait "$RUNNER_PID" || STATUS=$?
done
trap - TERM

# The marker is withheld on any non-zero exit -- a failed model, or a stop
# partway through -- so resubmit_failed.sh picks this target up again. Nothing
# already computed is redone.
if [[ "$STATUS" -ne 0 ]]; then
  echo "=== [$(date)] $TARGET_NAME incomplete (exit $STATUS); marker withheld. ==="
  exit "$STATUS"
fi

echo "=== [$(date)] $TARGET_NAME complete. ==="
touch "$MARKER"
