#!/usr/bin/env bash
#SBATCH --job-name=bundle_ppi_local
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
MODE="${MODE:-full}"
BUNDLE_DIR="${BUNDLE_DIR:-$PROJECT_DIR/local_transfer_$MODE}"
cd "$PROJECT_DIR"
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Serial file/group copying, one CPU. No GPU or worker pool.
echo "Bundle execution: 1 process, 1 numerical-library thread; Slurm CPUs=${SLURM_CPUS_PER_TASK:-1}"
python bundle_local_extension_data.py --mode "$MODE" --output "$BUNDLE_DIR" "$@"
