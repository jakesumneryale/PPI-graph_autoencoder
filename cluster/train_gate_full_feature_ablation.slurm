#!/usr/bin/env bash
#SBATCH --job-name=gate_full_ablation
#SBATCH --partition=pi_co54
#SBATCH --array=1-6%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=6-23:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# Six full-data runs. Tasks 1-3 use the original two-layer GAT; tasks 4-6
# use the four-layer residual GAT. All runs use only models whose committed
# Voronoi calculation passed the earlier audit, keeping the sample population
# fixed even before Voronoi area is enabled as an input feature.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/home/jas485/project_pi_co54/jas485/ppi_processed_graphs}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/voronoi_dataset_audit/successful_models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/full_feature_ablation_lr5e-6}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Full graph directory not found: $DATA_DIR" >&2
  exit 2
fi
if [[ ! -d "$MODEL_LIST_DIR" ]]; then
  echo "Successful-model list directory not found: $MODEL_LIST_DIR" >&2
  exit 2
fi

STAGE_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))
ARCHITECTURE_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))

STAGE_LABELS=(
  "01_rsasa_i"
  "02_plus_interface_node_degree"
  "03_plus_voronoi_contact_area"
)
NODE_FEATURES=(
  "aa_type,chain,interface_nodes,rsasa_i"
  "aa_type,chain,interface_nodes,rsasa_i,interface_node_degree"
  "aa_type,chain,interface_nodes,rsasa_i,interface_node_degree"
)
EDGE_FEATURES=(
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

RUN_LABEL="${ARCHITECTURE_LABEL}_${STAGE_LABELS[$STAGE_INDEX]}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_LABEL"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Architecture: $ARCHITECTURE_LABEL"
echo "Node features: ${NODE_FEATURES[$STAGE_INDEX]}"
echo "Edge features: ${EDGE_FEATURES[$STAGE_INDEX]}"
echo "Full dataset: $DATA_DIR"
echo "Successful-model lists: $MODEL_LIST_DIR"
echo "Output: $OUTPUT_DIR"
nvidia-smi

python train_gate.py \
  --cluster \
  --data "$DATA_DIR" \
  --model-list-dir "$MODEL_LIST_DIR" \
  --device cuda \
  --epochs 50 \
  --batch-size 16 \
  --lr 5e-6 \
  --num-workers 8 \
  --worker-start-method spawn \
  --persistent-workers \
  --node-features "${NODE_FEATURES[$STAGE_INDEX]}" \
  --edge-features "${EDGE_FEATURES[$STAGE_INDEX]}" \
  --output-dir "$OUTPUT_DIR" \
  "${ARCHITECTURE_ARGS[@]}"
