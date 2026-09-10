#!/usr/bin/env bash
#SBATCH --job-name=interface_degree
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/voronoi_dataset_audit/subset_hdf5}"
REPORT="${REPORT:-$PROJECT_DIR/gate_run/node_feature_ablation_lr5e-6/interface_node_degree_summary.csv}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

python add_interface_node_degree.py "$DATA_DIR" --report "$REPORT"
