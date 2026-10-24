#!/usr/bin/env bash
# Run this from the cluster login node; preparation is serial and reuses existing data.
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/gate_run/dockq_priority_sweep}"
SOURCE_MATRIX="${SOURCE_MATRIX:-$PROJECT_DIR/gate_run/model_extensions_10pct_quick_no_apbs/matrix.json}"
cd "$PROJECT_DIR"
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Fail before submission if the environment cannot import the updated trainer.
python train_gate.py --help > /dev/null
python preflight_dockq_priority_sweep.py --source "$SOURCE_MATRIX"
python build_dockq_priority_sweep.py --source "$SOURCE_MATRIX" --output "$EXPERIMENT_DIR"
JOB=$(sbatch --parsable --array=1-36 --export=ALL \
  --output="$EXPERIMENT_DIR/logs/train_%A_%a.out" cluster/train_model_extensions.slurm)
printf 'Submitted 36 validation-only jobs: %s\nResults/logs: %s\n' "$JOB" "$EXPERIMENT_DIR"
