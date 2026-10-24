#!/usr/bin/env bash
#SBATCH --job-name=gate_anti_memorization_full
#SBATCH --partition=pi_co54
#SBATCH --array=1-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=4-12:00:00

# Six anti-memorization arms x three seeds on the full repaired cohort, all with
# the full electrostatics-rung feature set from cluster/train_gate_feature_ladder_full.slurm
# and the same data, model lists and frozen target split. Submit through
# cluster/submit_anti_memorization_full.sh, which sets the log paths.
#
# Every feature-ladder run selected epoch 1-2 while validation DockQ MSE then
# rose for 48 epochs. The arms test ways of stopping the network memorizing
# each training target's native contacts:
#   A control        fixed-weight objective only (plus step validation, below)
#   B detached       DockQ head on detached node embeddings: encoder sees reconstruction only
#   C adversary      training-target classifier behind gradient reversal on the graph embedding
#   D augment        denoising input corruption: edge dropout, node masking, C-alpha jitter
#   E balanced_ema   256 decoys/target/epoch + EMA weights for validation/checkpoints
#   F combined       C + D + E
#
# Shared objective (replaces the ladder's adaptive loss shares, whose DockQ
# weight climbed from ~14 to ~260): DockQ MSE + 1 x (node MSE + edge MSE +
# 0.1 x edge BCE), capped inverse-sqrt DockQ range weighting (alpha 0.5),
# combined whole-graph + interface pooling, dropout 0.3, LR 3e-4 cosine with
# 1,000 warmup steps, batch 16, 20 epochs. Every arm validates every 2,000
# optimizer steps and at each epoch end and keeps the best checkpoint by exact
# graph-average validation DockQ MSE. Test evaluation is disabled throughout.
#
# CPU allocation: one main process (one PyTorch/BLAS thread) plus 7 DataLoader
# workers = 8 CPUs. Workers are not persistent. During a mid-epoch validation
# the paused training pool is idle (blocked on its index queue) while the
# 7-worker validation pool runs, so at most 8 processes are busy at once.
#
# Walltime: the 50-epoch ladder on this cohort fit in 6-23:00:00; 20 epochs plus
# ~15% step-validation overhead is well under half of that.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/gate_run/full_pooling_repaired/model_lists}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PROJECT_DIR/gate_run/full_pooling_repaired/target_splits.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/anti_memorization_full}"
EPOCHS="${EPOCHS:-20}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-2000}"
LOADER_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
if (( LOADER_WORKERS < 1 )); then
  echo "Need at least 2 CPUs per task (main process + loader workers)." >&2
  exit 2
fi
for required in "$DATA_DIR" "$MODEL_LIST_DIR"; do
  [[ -d "$required" ]] || { echo "Required directory not found: $required" >&2; exit 2; }
done
[[ -f "$SPLIT_MANIFEST" ]] || { echo "Split manifest not found: $SPLIT_MANIFEST" >&2; exit 2; }

ARMS=(A_control B_detached C_adversary D_augment E_balanced_ema F_combined)
SEEDS=(7 17 27)
ARM="${ARMS[$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))]}"
SEED="${SEEDS[$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))]}"

ADVERSARY=(--target-adversary-weight 0.1)
AUGMENT=(--edge-dropout 0.1 --node-feature-mask 0.1 --ca-dist-noise 0.25)
BALANCED_EMA=(--decoys-per-target 256 --ema-decay 0.999)
case "$ARM" in
  A_control)      ARM_ARGS=() ;;
  B_detached)     ARM_ARGS=(--dockq-head-mode detached) ;;
  C_adversary)    ARM_ARGS=("${ADVERSARY[@]}") ;;
  D_augment)      ARM_ARGS=("${AUGMENT[@]}") ;;
  E_balanced_ema) ARM_ARGS=("${BALANCED_EMA[@]}") ;;
  F_combined)     ARM_ARGS=("${ADVERSARY[@]}" "${AUGMENT[@]}" "${BALANCED_EMA[@]}") ;;
  *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/${ARM}_seed${SEED}"
if [[ -e "$OUTPUT_DIR/gate_model.pt" ]]; then
  echo "Refusing to overwrite completed run: $OUTPUT_DIR" >&2
  exit 2
fi

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Read-only HDF5 access from many concurrent workers; see train_gate_feature_ladder_full.slurm.
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "$OUTPUT_DIR"
echo "Array task $SLURM_ARRAY_TASK_ID: arm=$ARM seed=$SEED"
echo "Arm options: ${ARM_ARGS[*]:-(none)}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-unset}; main process + $LOADER_WORKERS loader workers; 1 thread each"
echo "Output: $OUTPUT_DIR"
nvidia-smi

python train_gate.py \
  --cluster \
  --data "$DATA_DIR" \
  --model-list-dir "$MODEL_LIST_DIR" \
  --split-manifest "$SPLIT_MANIFEST" \
  --device cuda \
  --epochs "$EPOCHS" \
  --batch-size 16 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --warmup-steps 1000 \
  --hidden-dim 64 \
  --latent-dim 32 \
  --gat-heads 4 \
  --dropout 0.3 \
  --residual_connections \
  --pooling combined \
  --node-features "aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node" \
  --edge-features "interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing,v_es" \
  --edge-recon-features "interface_edges,ca_dist" \
  --edge-feature-transforms "voronoi_contact_area=log1p" \
  --standardize-edge-features "voronoi_contact_area,v_es" \
  --edge-stats-seed 7 \
  --loss-weight-mode fixed \
  --reconstruction-lambda 1 \
  --dockq-range-weighting inverse-sqrt \
  --dockq-weight-exponent 0.5 \
  --dockq-range-diagnostics \
  --checkpoint-metric target_mse \
  --no-test-evaluation \
  --val-every-steps "$VAL_EVERY_STEPS" \
  --num-workers "$LOADER_WORKERS" \
  --worker-start-method spawn \
  --cpu-threads 1 \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  "${ARM_ARGS[@]}"
