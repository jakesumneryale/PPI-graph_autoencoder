#!/usr/bin/env bash
#SBATCH --job-name=gate_node_ablation
#SBATCH --partition=pi_co54
#SBATCH --array=1-12%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# Twelve controlled runs:
#   tasks 1-6:  original two-layer GAT encoder
#   tasks 7-12: four-layer GAT encoder with residual connections
# Within each architecture, node features are added cumulatively in this order:
# aa_type, chain, interface_nodes, rsasa_i, interface_node_degree. The sixth
# stage keeps all five node features and adds the Voronoi contact-area edge
# feature, giving a direct paired test of its contribution.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/voronoi_dataset_audit/subset_hdf5}"
ABLATION_DIR="${ABLATION_DIR:-$PROJECT_DIR/gate_run/node_feature_ablation_lr5e-6}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PROJECT_DIR/gate_run/voronoi_10pct_50epochs/target_splits.json}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Dataset directory not found: $DATA_DIR" >&2
  exit 2
fi
if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  echo "Shared target split manifest not found: $SPLIT_MANIFEST" >&2
  exit 2
fi

FEATURE_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) % 6 ))
ARCHITECTURE_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 6 ))

FEATURE_LABELS=(
  "01_aa_type"
  "02_plus_chain"
  "03_plus_interface_nodes"
  "04_plus_rsasa_i"
  "05_plus_interface_node_degree"
  "06_plus_voronoi_contact_area"
)
NODE_FEATURES=(
  "aa_type"
  "aa_type,chain"
  "aa_type,chain,interface_nodes"
  "aa_type,chain,interface_nodes,rsasa_i"
  "aa_type,chain,interface_nodes,rsasa_i,interface_node_degree"
  "aa_type,chain,interface_nodes,rsasa_i,interface_node_degree"
)
EDGE_FEATURES=(
  "interface_edges,ca_dist"
  "interface_edges,ca_dist"
  "interface_edges,ca_dist"
  "interface_edges,ca_dist"
  "interface_edges,ca_dist"
  "interface_edges,ca_dist,voronoi_contact_area"
)

if (( ARCHITECTURE_INDEX == 0 )); then
  ARCHITECTURE_LABEL="gat2"
  ARCHITECTURE_ARGS=()
else
  ARCHITECTURE_LABEL="gat4_residual"
  ARCHITECTURE_ARGS=(--residual_connections)
fi

RUN_LABEL="${ARCHITECTURE_LABEL}_${FEATURE_LABELS[$FEATURE_INDEX]}"
OUTPUT_DIR="$ABLATION_DIR/$RUN_LABEL"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Architecture: $ARCHITECTURE_LABEL"
echo "Node features: ${NODE_FEATURES[$FEATURE_INDEX]}"
echo "Edge features: ${EDGE_FEATURES[$FEATURE_INDEX]}"
echo "Dataset: $DATA_DIR"
echo "Shared split: $SPLIT_MANIFEST"
echo "Output: $OUTPUT_DIR"
nvidia-smi

python train_gate.py \
  --cluster \
  --data "$DATA_DIR" \
  --device cuda \
  --epochs 50 \
  --batch-size 16 \
  --lr 5e-6 \
  --num-workers 8 \
  --worker-start-method spawn \
  --persistent-workers \
  --node-features "${NODE_FEATURES[$FEATURE_INDEX]}" \
  --edge-features "${EDGE_FEATURES[$FEATURE_INDEX]}" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  "${ARCHITECTURE_ARGS[@]}"
