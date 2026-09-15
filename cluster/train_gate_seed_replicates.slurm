#!/usr/bin/env bash
#SBATCH --job-name=gate_seed_replicates
#SBATCH --partition=pi_co54
#SBATCH --array=1-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=6-23:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# Six runs: two configurations x three seeds.
#
#   tasks 1-3  CONFIG=base  base features, no Voronoi, no interface-node degree
#   tasks 4-6  CONFIG=full  all features, Voronoi area as an input-only feature
#
# base is the control. It was the best cell of the previous ablation, so if
# full does not beat it by more than the seed spread, the extra features are
# not earning their place.
#
# Held identical across the three:
#   * the train/val/test target split, from the shared manifest written by
#     cluster/prepare_seed_replicate_splits.slurm
#   * the edge-feature normalisation statistics, pinned with --edge-stats-seed
#   * every hyperparameter
#
# What --seed varies: model initialisation, training batch order, dropout masks
# and negative-edge sampling. The spread across the three runs is therefore a
# fair estimate of run-to-run noise -- which is what you need before claiming a
# feature or a loss change helped, given the ablation's per-target effects were
# all inside p = 0.07-0.77.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/home/jas485/project_pi_co54/jas485/ppi_processed_graphs}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/voronoi_dataset_audit/successful_models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/seed_replicates}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$OUTPUT_ROOT/shared_target_splits.json}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Graph directory not found: $DATA_DIR" >&2
  exit 2
fi
if [[ ! -d "$MODEL_LIST_DIR" ]]; then
  echo "Successful-model list directory not found: $MODEL_LIST_DIR" >&2
  exit 2
fi
if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  echo "Shared split manifest not found: $SPLIT_MANIFEST" >&2
  echo "Run cluster/prepare_seed_replicate_splits.slurm first." >&2
  exit 2
fi

SEEDS=(7 17 27)
CONFIGS=(base full)
SEED_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))
CONFIG_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))
SEED="${SEEDS[$SEED_INDEX]}"
CONFIG="${CONFIGS[$CONFIG_INDEX]}"

case "$CONFIG" in
  full)
    NODE_FEATURES="aa_type,chain,interface_nodes,rsasa_i,interface_node_degree"
    EDGE_FEATURES="interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing"
    EDGE_RECON_FEATURES="interface_edges,ca_dist"
    EDGE_TRANSFORM_ARGS=(--edge-feature-transforms "voronoi_contact_area=log1p")
    EDGE_STANDARDIZE_ARGS=(--standardize-edge-features "voronoi_contact_area")
    ;;
  base)
    NODE_FEATURES="aa_type,chain,interface_nodes,rsasa_i"
    EDGE_FEATURES="interface_edges,ca_dist"
    EDGE_RECON_FEATURES="interface_edges,ca_dist"
    EDGE_TRANSFORM_ARGS=()
    EDGE_STANDARDIZE_ARGS=()
    ;;
  *)
    echo "Unknown CONFIG: $CONFIG (expected 'full' or 'base')" >&2
    exit 2
    ;;
esac

RUN_LABEL="gat4_residual_${CONFIG}_seed${SEED}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_LABEL"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "Shared split manifest: $SPLIT_MANIFEST"
echo "Node features: $NODE_FEATURES"
echo "Edge features: $EDGE_FEATURES"
echo "Edge features reconstructed: $EDGE_RECON_FEATURES"
echo "Loss shares: node=0.15 edge_attr=0.15 edge_presence=0.10 target=0.60 (normalised)"
echo "Checkpoint metric: val_target_mse"
echo "Learning rate: 3e-4, 1000-step warmup, cosine decay"
echo "Output: $OUTPUT_DIR"
nvidia-smi

python train_gate.py \
  --cluster \
  --data "$DATA_DIR" \
  --model-list-dir "$MODEL_LIST_DIR" \
  --device cuda \
  --epochs 50 \
  --batch-size 16 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --warmup-steps 1000 \
  --num-workers 8 \
  --worker-start-method spawn \
  --persistent-workers \
  --node-features "$NODE_FEATURES" \
  --edge-features "$EDGE_FEATURES" \
  --edge-recon-features "$EDGE_RECON_FEATURES" \
  ${EDGE_TRANSFORM_ARGS[@]+"${EDGE_TRANSFORM_ARGS[@]}"} \
  ${EDGE_STANDARDIZE_ARGS[@]+"${EDGE_STANDARDIZE_ARGS[@]}"} \
  --edge-stats-seed 7 \
  --loss-weight-mode shares \
  --node-share 0.15 \
  --edge-attr-share 0.15 \
  --edge-presence-share 0.10 \
  --target-share 0.60 \
  --checkpoint-metric target_mse \
  --seed "$SEED" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --residual_connections
