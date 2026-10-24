#!/usr/bin/env bash
# Reuse the completed APBS-free data and split. Submit only four new hypotheses.
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/gate_run/generalization_diagnostics_seed7}"
SOURCE_MATRIX="${SOURCE_MATRIX:-$PROJECT_DIR/gate_run/model_extensions_10pct_quick_no_apbs/matrix.json}"
cd "$PROJECT_DIR"
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python build_generalization_diagnostics.py --source "$SOURCE_MATRIX" --output "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR/logs"
JOB=$(sbatch --parsable --array=1-4 --export=ALL \
  --output="$EXPERIMENT_DIR/logs/train_%A_%a.out" cluster/train_model_extensions.slurm)
echo "Four validation-only diagnostic jobs: $JOB"
echo "Reusing existing data, frozen split, and baseline control. No test predictions will be generated."
echo "Results and logs: $EXPERIMENT_DIR"
