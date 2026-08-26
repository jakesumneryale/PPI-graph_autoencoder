#!/usr/bin/env bash
# Read-only progress/integrity check for one target's APBS output.
# Usage: bash apbs_analysis/cluster/check_apbs_output.sh TARGET [MODEL_GROUP]
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
PDB_BASE_DIR="${PDB_BASE_DIR:-/nfs/roberts/pi/pi_co54/jas485/uniformly_sampled_target_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/roberts/pi/pi_co54/jas485/ppi_gnn_data_store/apbs_model_data}"
ENV_PREFIX="${APBS_ENV_PREFIX:-$PROJECT_DIR/apbs_env}"
TARGET_NAME="${1:-}"
MODEL_NAME="${2:-}"

if [[ -z "$TARGET_NAME" ]]; then
  echo "Usage: bash apbs_analysis/cluster/check_apbs_output.sh TARGET [MODEL_GROUP]" >&2
  exit 2
fi

PYTHON="${APBS_PYTHON:-$ENV_PREFIX/bin/python}"
HDF5="$OUTPUT_DIR/${TARGET_NAME}_apbs_surface.hdf5"
SUMMARY="$OUTPUT_DIR/${TARGET_NAME}_apbs_summary.csv"
MARKER="$PROJECT_DIR/apbs_run/completed/${TARGET_NAME}.done"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: environment Python not found at $PYTHON (run cluster/setup_env.sh)" >&2
  exit 2
fi
if [[ ! -f "$HDF5" ]]; then
  echo "No output yet for $TARGET_NAME at $HDF5" >&2
  exit 1
fi

echo "Target:      $TARGET_NAME"
echo "Completion:  $([[ -f "$MARKER" ]] && echo DONE || echo 'not complete')"
if [[ -f "$SUMMARY" ]]; then
  echo "Summary:     $SUMMARY"
  awk -F, 'NR>1 {count[$5]++} END {for (status in count) printf "  %-16s %d\n", status, count[status]}' "$SUMMARY"
  echo "  slowest models:"
  awk -F, 'NR>1 {print $7, $2}' "$SUMMARY" | sort -rn | head -3 | awk '{printf "    %-24s %ss\n", $2, $1}'
fi
echo

cd "$PROJECT_DIR"
"$PYTHON" -m apbs_analysis.inspect_apbs_output "$HDF5" ${MODEL_NAME:+"$MODEL_NAME"} \
  --pdb-dir "$PDB_BASE_DIR/sampled_$TARGET_NAME"
