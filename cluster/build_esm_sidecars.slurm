#!/usr/bin/env bash
#SBATCH --job-name=esm_sidecars
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# One target's ESM sidecar per array task (esm_sidecars.py --graph): serial h5py reads and writes, one CPU. Submitted by
# cluster/submit_homolog_free_fold4.sh esm. Task file line i = "<graph path> <embedding directory>".
set -euo pipefail
TASKS=${1:?usage: sbatch build_esm_sidecars.slurm <task file> <model list dir> <output dir>}
LISTS=${2:?}; OUT=${3:?}
PROJECT_DIR="${PROJECT_DIR:?}"
read -r GRAPH EMBEDS <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")"
[[ -n $GRAPH ]] || { echo "no task on line $SLURM_ARRAY_TASK_ID of $TASKS" >&2; exit 2; }
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$PROJECT_DIR"
python esm_sidecars.py --graph "$GRAPH" --model-list-dir "$LISTS" --esm-root "$EMBEDS" --output "$OUT"
