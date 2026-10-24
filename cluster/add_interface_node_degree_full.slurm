#!/usr/bin/env bash
#SBATCH --job-name=interface_degree_full
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/home/jas485/project_pi_co54/jas485/ppi_processed_graphs}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/voronoi_dataset_audit/successful_models}"
REPORT="${REPORT:-$PROJECT_DIR/gate_run/full_feature_ablation_lr5e-6/interface_node_degree_summary.csv}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

python add_interface_node_degree.py \
  "$DATA_DIR" \
  --model-list-dir "$MODEL_LIST_DIR" \
  --report "$REPORT"
