#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/gate_run/dockq_followup_sweep}"
SOURCE_MATRIX="${SOURCE_MATRIX:-$PROJECT_DIR/gate_run/dockq_priority_sweep/matrix.json}"
PREFLIGHT_MATRIX="${PREFLIGHT_MATRIX:-$PROJECT_DIR/gate_run/model_extensions_10pct_quick_no_apbs/matrix.json}"
cd "$PROJECT_DIR"
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python train_gate.py --help > /dev/null
python preflight_dockq_priority_sweep.py --source "$PREFLIGHT_MATRIX" --require-interface
python build_dockq_followup_sweep.py --source "$SOURCE_MATRIX" --output "$EXPERIMENT_DIR"
DIAGNOSTICS=$(sbatch --parsable --array=1-3 --export=ALL \
  --output="$EXPERIMENT_DIR/logs/train_%A_%a.out" cluster/train_model_extensions.slurm)
DIAGNOSTICS_ID=${DIAGNOSTICS%%;*}
printf 'Diagnostic array: %s\n' "$DIAGNOSTICS"
TRAINING=$(sbatch --parsable --array=4-36 --dependency="afterok:$DIAGNOSTICS_ID" --export=ALL \
  --output="$EXPERIMENT_DIR/logs/train_%A_%a.out" cluster/train_model_extensions.slurm)
printf 'Follow-up training array: %s\nResults and logs: %s\n' "$TRAINING" "$EXPERIMENT_DIR"
