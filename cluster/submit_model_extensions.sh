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
export ESM_ROOT="${ESM_ROOT:-/nfs/roberts/pi/pi_co54/nb586/scratch_backup/SS_embeds}"
export OPTIONAL_NODE_FEATURES_DIR="${OPTIONAL_NODE_FEATURES_DIR:-/home/jas485/project_pi_co54/jas485/rsasa_i_graph_data}"
export FEATURE_DATA="${FEATURE_DATA:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
cd "$PROJECT_DIR"
mkdir -p "$EXPERIMENT_DIR/logs"
if [[ -e "$EXPERIMENT_DIR/matrix.json" || -e "$EXPERIMENT_DIR/targets.txt" ]]; then
  echo "Experiment already exists; use a new EXPERIMENT_DIR or resume its individual stages." >&2
  exit 2
fi
# Shell glob preserves the exact target cohort without loading a Python environment.
for path in "$SUBSET_DIR"/*.h5 "$SUBSET_DIR"/*.hdf5; do
  [[ -f "$path" ]] || continue
  name="${path##*/}"
  echo "${name%.*}"
done | sort -u > "$EXPERIMENT_DIR/targets.txt"
COUNT=$(wc -l < "$EXPERIMENT_DIR/targets.txt")
[[ "$COUNT" -gt 0 ]]
DEPENDENCY=()
if [[ "${SKIP_PREP:-0}" == 1 ]]; then
  [[ -d "$EXTENSION_DATA" ]] || { echo "Prepared data missing: $EXTENSION_DATA" >&2; exit 2; }
else
  PREP=$(sbatch --parsable --array="1-$COUNT" --export=ALL,STAGE=data \
    --output="$EXPERIMENT_DIR/logs/data_%A_%a.out" cluster/prepare_model_extensions.slurm)
  DEPENDENCY+=(--dependency="afterok:${PREP%%;*}")
fi
MATRIX=$(sbatch --parsable "${DEPENDENCY[@]}" --export=ALL,STAGE=matrix \
  --output="$EXPERIMENT_DIR/logs/matrix_%j.out" cluster/prepare_model_extensions.slurm)
echo "Mode: $RUN_MODE; preparation: ${PREP:-reused}; matrix and GPU submission: $MATRIX"
