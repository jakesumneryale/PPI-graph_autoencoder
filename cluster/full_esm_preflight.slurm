#!/usr/bin/env bash
#SBATCH --job-name=full_esm_prepare
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[[ "${SLURM_CPUS_PER_TASK:-1}" == 1 ]] || { echo 'Serial stage requires one CPU'; exit 2; }
echo "ESM $STAGE: 1 process, 1 numerical-library thread"
case "$STAGE" in
 prepare) python esm_sidecars.py --parent "$FULL_PARENT_DIR" --task "$SLURM_ARRAY_TASK_ID" --esm-root "$ESM_ROOT" --output "$ESM_SIDECARS" ;;
 build) python build_full_esm_validation.py --parent "$FULL_PARENT_DIR" --sidecars "$ESM_SIDECARS" --output "$EXPERIMENT_DIR" ;;
 *) exit 2 ;;
esac
