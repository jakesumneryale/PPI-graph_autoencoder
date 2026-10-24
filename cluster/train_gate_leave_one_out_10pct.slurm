#!/usr/bin/env bash
#SBATCH --job-name=gate_leave_one_out_10pct
#SBATCH --partition=pi_co54
#SBATCH --array=1-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --nice=200
#SBATCH --output=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder/voronoi_run/logs/%x_%A_%a.err

# PRIORITY 3. Leave-one-out ablation at the full available feature set
# (core + interface degree + rSASA_i + Voronoi contact area -- electrostatics
# is excluded, same reason as in the feature-ladder job), 3 seeds per
# configuration, on the same 10% subset and split as the ladder job.
#
# --nice=200 deliberately queues this behind the un-niced ladder job
# (Priority 1) when both are runnable and GPUs are contended; it does not
# block on the ladder finishing (no --dependency), since the two ask
# different questions and can run concurrently once a GPU is free.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/gate_run/subset_10pct_hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/leave_one_out_10pct}"
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

# Full available feature set: degree, rsasa_i, voronoi. One run per array
# task drops exactly one; "full" (nothing dropped) is already the "voronoi"
# rung of the feature-ladder job, so it is not repeated here.
DROPPED=(degree rsasa voronoi)
SEEDS=(7 17 27)
CONFIG_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))
SEED_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))
DROP="${DROPPED[$CONFIG_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

FULL_NODE="aa_type,chain,interface_nodes,interface_node_degree,rsasa_i"
FULL_EDGE="interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing"
EDGE_RECON_FEATURES="interface_edges,ca_dist"
EXTRA_ARGS=(--edge-feature-transforms "voronoi_contact_area=log1p" --standardize-edge-features "voronoi_contact_area")

case "$DROP" in
  degree)
    NODE_FEATURES="aa_type,chain,interface_nodes,rsasa_i"
    EDGE_FEATURES="$FULL_EDGE"
    ;;
  rsasa)
    NODE_FEATURES="aa_type,chain,interface_nodes,interface_node_degree"
    EDGE_FEATURES="$FULL_EDGE"
    ;;
  voronoi)
    NODE_FEATURES="$FULL_NODE"
    EDGE_FEATURES="interface_edges,ca_dist"
    EDGE_RECON_FEATURES="interface_edges,ca_dist"
    EXTRA_ARGS=()
    ;;
  *)
    echo "Unknown feature to drop: $DROP" >&2; exit 2 ;;
esac

RUN_LABEL="gat4_residual_minus_${DROP}_seed${SEED}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_LABEL"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "Array task: $SLURM_ARRAY_TASK_ID  (dropping=$DROP seed=$SEED)"
echo "Node features: $NODE_FEATURES"
echo "Edge features: $EDGE_FEATURES"
echo "Output: $OUTPUT_DIR"
nvidia-smi

python train_gate.py \
  --cluster \
  --data "$DATA_DIR" \
  --device cuda \
  --epochs 50 \
  --batch-size 16 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --warmup-steps 200 \
  --num-workers 8 \
  --worker-start-method spawn \
  --persistent-workers \
  --node-features "$NODE_FEATURES" \
  --edge-features "$EDGE_FEATURES" \
  --edge-recon-features "$EDGE_RECON_FEATURES" \
  "${EXTRA_ARGS[@]}" \
  --edge-stats-seed 5 \
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
