#!/usr/bin/env bash
#SBATCH --job-name=gate_loss_reweight
#SBATCH --partition=pi_co54
#SBATCH --array=1-6%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=6-23:00:00
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# Loss-reweighting sweep: the same six feature/architecture cells as
# train_gate_full_feature_ablation.slurm, but with the DockQ term promoted to
# 60% of the loss and the checkpoint selected on validation DockQ MSE.
#
# In the unweighted ablation the raw edge-attribute MSE carried 84-98% of the
# total loss (it is computed on Angstrom-scale features), the DockQ term carried
# 0.9-13%, and the checkpoint was chosen on that total -- so every run saved its
# final epoch even though validation DockQ MSE had bottomed out by epoch 2-5 and
# was 16-18% worse by then.
#
# Stage 3 additionally requires the Voronoi missing mask to have been committed:
#   python -m voronoi_edge_features.commit_graph_features --target-name <TARGET> ...
# Without it, voronoi_contact_missing is absent from the graph HDF5 and the run
# fails fast at dataset construction.
#
# --loss-weight-mode shares divides each term by a running estimate of its own
# magnitude before applying the share, so the shares below are the fractions of
# the loss each term actually contributes. Setting the raw --*-weight flags to
# these numbers would NOT do that: 0.60 x 0.085 still loses to 0.15 x 9.8.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/home/jas485/project_pi_co54/jas485/ppi_processed_graphs}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/voronoi_dataset_audit/successful_models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/loss_reweight_lr5e-6}"

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
  "interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing"
)
# Voronoi area is an encoder input only: it informs attention but is not a
# regression target, so its raw Angstrom^2 scale can no longer take over the
# loss. The companion missing mask separates "tiny contact" from "no data".
EDGE_RECON_FEATURES=(
  "interface_edges,ca_dist"
  "interface_edges,ca_dist"
  "interface_edges,ca_dist"
)
# log1p then z-score, applied on load. ca_dist is deliberately left raw so
# ca_dist_mae_angstrom stays comparable with every earlier run.
EDGE_TRANSFORMS=(
  ""
  ""
  "voronoi_contact_area=log1p"
)
EDGE_STANDARDIZE=(
  ""
  ""
  "voronoi_contact_area"
)

if (( ARCHITECTURE_INDEX == 0 )); then
  ARCHITECTURE_LABEL="gat2"
  ARCHITECTURE_ARGS=()
else
  ARCHITECTURE_LABEL="gat4_residual"
  ARCHITECTURE_ARGS=(--residual_connections)
fi

EDGE_TRANSFORM_ARGS=()
if [[ -n "${EDGE_TRANSFORMS[$STAGE_INDEX]}" ]]; then
  EDGE_TRANSFORM_ARGS=(--edge-feature-transforms "${EDGE_TRANSFORMS[$STAGE_INDEX]}")
fi
EDGE_STANDARDIZE_ARGS=()
if [[ -n "${EDGE_STANDARDIZE[$STAGE_INDEX]}" ]]; then
  EDGE_STANDARDIZE_ARGS=(--standardize-edge-features "${EDGE_STANDARDIZE[$STAGE_INDEX]}")
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
echo "Learning rate: 3e-4, 1000-step warmup, cosine decay"
echo "Edge features reconstructed: ${EDGE_RECON_FEATURES[$STAGE_INDEX]}"
echo "Edge transforms: ${EDGE_TRANSFORMS[$STAGE_INDEX]:-none}"
echo "Loss shares: node=0.15 edge_attr=0.15 edge_presence=0.10 target=0.60 (normalised)"
echo "Checkpoint metric: val_target_mse"
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
  --node-features "${NODE_FEATURES[$STAGE_INDEX]}" \
  --edge-features "${EDGE_FEATURES[$STAGE_INDEX]}" \
  --edge-recon-features "${EDGE_RECON_FEATURES[$STAGE_INDEX]}" \
  ${EDGE_TRANSFORM_ARGS[@]+"${EDGE_TRANSFORM_ARGS[@]}"} \
  ${EDGE_STANDARDIZE_ARGS[@]+"${EDGE_STANDARDIZE_ARGS[@]}"} \
  --loss-weight-mode shares \
  --node-share 0.15 \
  --edge-attr-share 0.15 \
  --edge-presence-share 0.10 \
  --target-share 0.60 \
  --checkpoint-metric target_mse \
  --output-dir "$OUTPUT_DIR" \
  "${ARCHITECTURE_ARGS[@]}"
