#!/usr/bin/env bash
#SBATCH --job-name=esm_embeddings
#SBATCH --partition=pi_co54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00

# ESM2 embeddings for the targets in <targets file> (compute_esm_embeddings.py): one process, one forward pass per chain
# on the GPU, so one CPU. Resumable. Submitted by cluster/submit_homolog_free_fold4.sh esm.
set -euo pipefail
TARGETS=${1:?usage: sbatch compute_esm_embeddings.slurm <targets file> <graph dir> <model list dir> <output dir> <stored embedding dir>}
GRAPHS=${2:?}; LISTS=${3:?}; OUT=${4:?}; CHECK=${5:?}
PROJECT_DIR="${PROJECT_DIR:?}"
export TORCH_HOME="${TORCH_HOME:-$(dirname "$PROJECT_DIR")/esm_models}"
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$PROJECT_DIR"
nvidia-smi || true
python compute_esm_embeddings.py --targets "$TARGETS" --graph-dir "$GRAPHS" --model-list-dir "$LISTS" --output "$OUT" --check-dir "$CHECK"
