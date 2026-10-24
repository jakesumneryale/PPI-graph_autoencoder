#!/usr/bin/env bash
#SBATCH --job-name=gate_long_screen_full
#SBATCH --partition=pi_co54
#SBATCH --array=1-21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=3-12:00:00

# 40-epoch screen (7 configurations x 3 seeds, full data). Every configuration is
# the anti-memorization control recipe (A: fixed weights, reconstruction lambda 1,
# dropout 0.3, capped inverse-sqrt range weighting, combined pooling) trained for
# 40 epochs with the cosine schedule spread over all 40, changing ONE thing:
#
#   control          A at 40 epochs
#   lambda3          reconstruction weight 3 (DockQ's relative pull on the encoder down)
#   wd3e-2           AdamW weight decay 3e-2 (was 1e-5, effectively off)
#   dropout05        dropout 0.5
#   hinge_max_b5     DockQ weight 1 below 0.23 rising to beta=5 at DockQ 1, evaluated at
#                    max(true, predicted), mean-normalized; replaces range weighting
#   capri_head       + CAPRI-class head (4-way softmax cross-entropy, weight 0.2)
#   rank             + within-target pairwise ranking loss (weight 0.1, pairs with a
#                    decoy >= 0.23); batches of 4 targets x 4 decoys so pairs exist
#
# The first three target long-run stability (does validation keep falling over 40
# epochs?); the last three target high-DockQ accuracy. Measurement is unchanged:
# validation every 2,000 steps and at epoch ends, exact graph-average validation
# DockQ MSE for checkpointing, no test evaluation.
#
# CPU: main process (one PyTorch/BLAS thread) + 7 non-persistent spawn loader
# workers = 8 CPUs. Time: 20-epoch runs of this recipe logged ~74 min/epoch
# including step validations, so 40 epochs is ~50 h.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/gate_run/full_pooling_repaired/model_lists}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PROJECT_DIR/gate_run/full_pooling_repaired/target_splits.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/long_screen_full}"
EPOCHS=40
VAL_EVERY_STEPS=2000
LOADER_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a SLURM array job." >&2
  exit 2
fi
(( LOADER_WORKERS >= 1 )) || { echo "Need at least 2 CPUs per task." >&2; exit 2; }
for required in "$DATA_DIR" "$MODEL_LIST_DIR"; do
  [[ -d "$required" ]] || { echo "Required directory not found: $required" >&2; exit 2; }
done
[[ -f "$SPLIT_MANIFEST" ]] || { echo "Split manifest not found: $SPLIT_MANIFEST" >&2; exit 2; }

CONFIGS=(control lambda3 wd3e-2 dropout05 hinge_max_b5 capri_head rank)
SEEDS=(7 17 27)
NAME="${CONFIGS[$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))]}"
SEED="${SEEDS[$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))]}"

LAMBDA=1; WEIGHT_DECAY=1e-5; DROPOUT=0.3
RANGE_ARGS=(--dockq-range-weighting inverse-sqrt --dockq-weight-exponent 0.5)
EXTRA_ARGS=()
case "$NAME" in
  control)      ;;
  lambda3)      LAMBDA=3 ;;
  wd3e-2)       WEIGHT_DECAY=3e-2 ;;
  dropout05)    DROPOUT=0.5 ;;
  hinge_max_b5) RANGE_ARGS=(--dockq-range-weighting hinge-max --dockq-hinge-beta 5 --dockq-hinge-start 0.23) ;;
  capri_head)   EXTRA_ARGS=(--capri-head-weight 0.2) ;;
  rank)         EXTRA_ARGS=(--rank-loss-weight 0.1 --rank-temperature 0.1 --targets-per-batch 4) ;;
  *) echo "Unknown config: $NAME" >&2; exit 2 ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/${NAME}_seed${SEED}"
if [[ -e "$OUTPUT_DIR/gate_model.pt" ]]; then
  echo "Refusing to overwrite existing run: $OUTPUT_DIR" >&2
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
echo "Array task $SLURM_ARRAY_TASK_ID: config=$NAME seed=$SEED"
echo "lambda=$LAMBDA weight_decay=$WEIGHT_DECAY dropout=$DROPOUT range=${RANGE_ARGS[*]} extra=${EXTRA_ARGS[*]:-(none)}"
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
  --weight-decay "$WEIGHT_DECAY" \
  --lr-schedule cosine \
  --warmup-steps 1000 \
  --hidden-dim 64 \
  --latent-dim 32 \
  --gat-heads 4 \
  --dropout "$DROPOUT" \
  --residual_connections \
  --pooling combined \
  --node-features "aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node" \
  --edge-features "interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing,v_es" \
  --edge-recon-features "interface_edges,ca_dist" \
  --edge-feature-transforms "voronoi_contact_area=log1p" \
  --standardize-edge-features "voronoi_contact_area,v_es" \
  --edge-stats-seed 7 \
  --loss-weight-mode fixed \
  --reconstruction-lambda "$LAMBDA" \
  "${RANGE_ARGS[@]}" \
  --dockq-range-diagnostics \
  --checkpoint-metric target_mse \
  --no-test-evaluation \
  --val-every-steps "$VAL_EVERY_STEPS" \
  --num-workers "$LOADER_WORKERS" \
  --worker-start-method spawn \
  --cpu-threads 1 \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
