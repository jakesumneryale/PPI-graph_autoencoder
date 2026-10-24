#!/usr/bin/env bash
#SBATCH --job-name=audit_expansion_graphs
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00

# Model list for one target-expansion graph file per array task (audit_expansion_graphs.py): serial h5py reads,
# one CPU. Submitted by cluster/submit_dataset_size_comparison.sh audit, which sets the array range and the
# task file (line i = graph path).
set -euo pipefail
TASKS=${1:?usage: sbatch audit_expansion_graphs.slurm <task file> <output dir>}
OUT=${2:?usage: sbatch audit_expansion_graphs.slurm <task file> <output dir>}
PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
GRAPH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
[[ -n $GRAPH ]] || { echo "no task on line $SLURM_ARRAY_TASK_ID of $TASKS" >&2; exit 2; }

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$PROJECT_DIR"
python audit_expansion_graphs.py --graph "$GRAPH" --output-dir "$OUT"
