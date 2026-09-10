#!/usr/bin/env bash
#SBATCH --job-name=audit_voronoi
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
GRAPH_DATA_DIR="${GRAPH_DATA_DIR:-/home/jas485/project_pi_co54/jas485/ppi_processed_graphs}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/voronoi_dataset_audit}"

mkdir -p "$PROJECT_DIR/voronoi_run/logs"
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

python audit_voronoi_dataset.py \
  --data-dir "$GRAPH_DATA_DIR" \
  --checkpoint-dir "$PROJECT_DIR/voronoi_edge_features_data/contact_area_hdf5" \
  --output-dir "$OUTPUT_DIR" \
  --fraction 0.1 \
  --seed 20250909
