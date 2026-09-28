#!/usr/bin/env bash
#SBATCH --job-name=gate_recon_weight_zero_10pct
#SBATCH --partition=pi_co54
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --nice=300
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# PRIORITY 4a. One piece of the "architecture ablations" TODO: reconstruction
# weight -> 0, the purely supervised control that isolates whatever joint
# training with the reconstruction objective is contributing. 3 seeds, full
# available feature set, same 10% subset and split as the other jobs here.
#
# This is the only architecture-ablation sub-item runnable without code
# changes (--node-share/--edge-attr-share/--edge-presence-share all support
# 0 directly). The other two named in the TODO -- attention versus plain
# graph convolution, and with/without edge conditioning -- are NOT included:
# GATE_model.py hard-codes GATConv with edge_dim always set whenever
# in_edge_feats > 0 (see __init__), so testing either requires adding a
# second encoder path first. Flagging this rather than skipping it silently;
# see the dissertation TODO this job partially addresses.
#
# --nice=300: lowest scheduling priority of the four jobs in this sweep.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/gate_run/subset_10pct_hdf5}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/voronoi_dataset_audit/successful_models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/recon_weight_zero_10pct}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PROJECT_DIR/gate_run/feature_ladder_10pct/shared_target_splits.json}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "10% subset not found at $DATA_DIR -- run build_10pct_subset.slurm first." >&2
  exit 2
fi
if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  echo "Shared split manifest not found at $SPLIT_MANIFEST." >&2
  echo "Run train_gate_feature_ladder_10pct.slurm (task 1) first, or set SPLIT_MANIFEST." >&2
  exit 2
fi

SEEDS=(7 17 27)
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID - 1))]}"
RUN_LABEL="gat4_residual_recon0_seed${SEED}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_LABEL"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "Array task: $SLURM_ARRAY_TASK_ID  (seed=$SEED)"
echo "Reconstruction weights: node=0 edge_attr=0 edge_presence=0 (fixed mode, not shares)"
echo "Output: $OUTPUT_DIR"
nvidia-smi

# Fixed-weight mode here, not --loss-weight-mode shares: the point of this
# control is a literal zero reconstruction weight, and the share-normalized
# balancer has no representation for "zero share" that also zeroes the
# gradient contribution as directly as a fixed multiplicative 0 does.
python train_gate.py \
  --data "$DATA_DIR" \
  --model-list-dir "$MODEL_LIST_DIR" \
  --device cuda \
  --epochs 50 \
  --batch-size 16 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --warmup-steps 200 \
  --num-workers 8 \
  --worker-start-method spawn \
  --persistent-workers \
  --node-features "aa_type,chain,interface_nodes,interface_node_degree,rsasa_i" \
  --edge-features "interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing" \
  --edge-recon-features "interface_edges,ca_dist" \
  --edge-feature-transforms "voronoi_contact_area=log1p" \
  --standardize-edge-features "voronoi_contact_area" \
  --edge-stats-seed 5 \
  --loss-weight-mode fixed \
  --node-weight 0 \
  --edge-attr-weight 0 \
  --edge-presence-weight 0 \
  --target-weight 1 \
  --checkpoint-metric target_mse \
  --seed "$SEED" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --residual_connections
