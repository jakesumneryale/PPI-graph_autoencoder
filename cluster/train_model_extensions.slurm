#!/usr/bin/env bash
#SBATCH --job-name=model_extensions
#SBATCH --partition=pi_co54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6-23:00:00
set -euo pipefail
cd "$PROJECT_DIR"
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1
python model_extension_experiments.py run --matrix "$EXPERIMENT_DIR/matrix.json" --task "$SLURM_ARRAY_TASK_ID"
