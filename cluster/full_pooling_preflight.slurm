#!/usr/bin/env bash
#SBATCH --job-name=full_pooling_audit
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
set -euo pipefail
cd "$PROJECT_DIR"
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [[ "${SLURM_CPUS_PER_TASK:-1}" != 1 ]]; then
 echo 'This stage is serial and requires exactly one CPU.' >&2; exit 2
fi
echo "Full dataset $STAGE: 1 process, 1 numerical-library thread; Slurm CPUs=${SLURM_CPUS_PER_TASK:-1}"
if [[ "$STAGE" == audit ]]; then
 python full_pooling_validation.py audit --paths "$EXPERIMENT_DIR/paths.json" \
  --task "$SLURM_ARRAY_TASK_ID" --rsasa "$RSASA_DIR" --output "$AUDIT_DIR"
elif [[ "$STAGE" == build ]]; then
 python full_pooling_validation.py build --paths "$EXPERIMENT_DIR/paths.json" \
  --audit-dir "$AUDIT_DIR" --prior-split "$PRIOR_SPLIT" --source-matrix "$SOURCE_MATRIX" \
  --output "$EXPERIMENT_DIR" --rsasa "$RSASA_DIR" --minimum "$MIN_ELIGIBLE_MODELS"
else
 echo "Unknown STAGE: $STAGE" >&2; exit 2
fi
