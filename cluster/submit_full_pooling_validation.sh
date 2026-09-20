#!/usr/bin/env bash
# Full graphs, not the old "full" seed mode of the 10% experiment launcher.
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
export RSASA_DIR="${RSASA_DIR:-/nfs/roberts/project/pi_co54/jas485/rsasa_i_graph_data}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/gate_run/full_pooling_validation}"
export AUDIT_DIR="${AUDIT_DIR:-$PROJECT_DIR/full_pooling_audit}"
export PRIOR_SPLIT="${PRIOR_SPLIT:-$PROJECT_DIR/gate_run/model_extensions_10pct_quick_no_apbs/target_splits.json}"
export SOURCE_MATRIX="${SOURCE_MATRIX:-$PROJECT_DIR/gate_run/dockq_followup_sweep/matrix.json}"
export MIN_ELIGIBLE_MODELS="${MIN_ELIGIBLE_MODELS:-150000}"
cd "$PROJECT_DIR"
if [[ "${RESUME_BUILD:-0}" == 1 ]]; then
  [[ -f "$EXPERIMENT_DIR/paths.json" && ! -e "$EXPERIMENT_DIR/matrix.json" ]] || {
    echo "Build-only retry requires an existing inventory and no completed matrix." >&2; exit 2;
  }
elif [[ -e "$EXPERIMENT_DIR" ]]; then
  echo "Experiment directory already exists; choose a fresh EXPERIMENT_DIR. Cached audits can be reused." >&2; exit 2
fi
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "$EXPERIMENT_DIR/logs" "$AUDIT_DIR"
python train_gate.py --help > /dev/null
BUILD_DEPENDENCY=()
AUDIT='reused completed audits'
if [[ "${RESUME_BUILD:-0}" != 1 ]]; then
 COUNT=$(python full_pooling_validation.py inventory --data "$DATA_DIR" --output "$EXPERIMENT_DIR/paths.json")
 AUDIT=$(sbatch --parsable --array="1-$COUNT" --export=ALL,STAGE=audit \
  --output="$EXPERIMENT_DIR/logs/audit_%A_%a.out" cluster/full_pooling_preflight.slurm)
 AUDIT_ID=${AUDIT%%;*}
 BUILD_DEPENDENCY=("--dependency=afterok:$AUDIT_ID")
fi
BUILD=$(sbatch --parsable "${BUILD_DEPENDENCY[@]}" --export=ALL,STAGE=build \
 --output="$EXPERIMENT_DIR/logs/build_%j.out" cluster/full_pooling_preflight.slurm)
BUILD_ID=${BUILD%%;*}
TRAIN=$(sbatch --parsable --array=1-12 --dependency="afterok:$BUILD_ID" --export=ALL \
 --output="$EXPERIMENT_DIR/logs/train_%A_%a.out" cluster/train_model_extensions.slurm)
printf 'Read-only audit: %s\nCoverage/matrix: %s\n12 GPU runs (depend on audit and coverage): %s\nResults: %s\n' "$AUDIT" "$BUILD" "$TRAIN" "$EXPERIMENT_DIR"
