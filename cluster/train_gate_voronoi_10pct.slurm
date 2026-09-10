#!/usr/bin/env bash
#SBATCH --job-name=gate_voronoi_10pct
#SBATCH --partition=pi_co54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/voronoi_dataset_audit/subset_hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/gate_run/voronoi_10pct_50epochs}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: ${SLURM_JOB_NODELIST:-not-set}"
echo "Dataset: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
nvidia-smi

python train_gate.py \
  --cluster \
  --data "$DATA_DIR" \
  --device cuda \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-5 \
  --num-workers 8 \
  --worker-start-method spawn \
  --persistent-workers \
  --node-feature-set all \
  --use-rsasa-i \
  --use-drsasa \
  --edge-features interface_edges,ca_dist,voronoi_contact_area \
  --output-dir "$OUTPUT_DIR"
