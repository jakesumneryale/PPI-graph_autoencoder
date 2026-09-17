#!/usr/bin/env bash
# Submit preprocessing -> frozen matrix -> unconstrained GPU array -> comparison.
# Set paths as environment variables. Preparation uses the existing 10% subset.
set -euo pipefail
export RUN_MODE="${1:-full}"
case "$RUN_MODE" in
  quick|full) ;;
  *) echo "Usage: bash cluster/submit_model_extensions.sh [quick|full]" >&2; exit 2 ;;
esac
export EPOCHS="${EPOCHS:-50}"
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export SUBSET_DIR="${SUBSET_DIR:-$PROJECT_DIR/voronoi_dataset_audit/subset_hdf5}"
export EXTENSION_DATA="${EXTENSION_DATA:-$PROJECT_DIR/extension_data_10pct_$RUN_MODE}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/gate_run/model_extensions_10pct_$RUN_MODE}"
export APBS_DIR="${APBS_DIR:-/nfs/roberts/pi/pi_co54/jas485/ppi_gnn_data_store/apbs_model_data}"
export PDB_ROOT="${PDB_ROOT:-/nfs/roberts/pi/pi_co54/jas485/uniformly_sampled_target_data}"
export ESM_ROOT="${ESM_ROOT:-/nfs/roberts/pi/pi_co54/nb685/scratch_backup/SS_embeds}"
export OPTIONAL_NODE_FEATURES_DIR="${OPTIONAL_NODE_FEATURES_DIR:-/home/jas485/project_pi_co54/jas485/rsasa_i_graph_data}"
export FEATURE_DATA="${FEATURE_DATA:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
cd "$PROJECT_DIR"
mkdir -p "$EXPERIMENT_DIR/logs"
if [[ -e "$EXPERIMENT_DIR/matrix.json" || -e "$EXPERIMENT_DIR/targets.txt" ]]; then
  echo "Experiment already exists; use a new EXPERIMENT_DIR or resume its individual stages." >&2
  exit 2
fi
# Inspect the HDF5 inputs in the cluster environment before submitting arrays.
PREFLIGHT=$(sbatch --parsable --export=ALL,STAGE=preflight \
  --output="$EXPERIMENT_DIR/logs/preflight_%j.out" cluster/prepare_model_extensions.slurm)
echo "Mode: $RUN_MODE; preflight and downstream submission: $PREFLIGHT"
