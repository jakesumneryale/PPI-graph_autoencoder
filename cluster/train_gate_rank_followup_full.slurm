#!/usr/bin/env bash
#SBATCH --job-name=gate_rank_followup_full
#SBATCH --partition=pi_co54
#SBATCH --array=1-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=2-00:00:00

# Ranking-loss follow-up (4 configurations x 3 seeds, full data, 20 epochs) with a
# single test-set evaluation per run.
#
# The within-target ranking loss is now standard: every configuration is the
# control recipe (fixed weights, dropout 0.3, capped inverse-sqrt range weighting,
# combined pooling, cosine over 20 epochs) PLUS the ranking loss (weight 0.1,
# temperature 0.1, pairs with a decoy >= 0.23) and grouped batches (4 targets x 4
# decoys). One further change per configuration:
#
#   rank           none (the new reference at 20 epochs)
#   rank_capri     + CAPRI-class head, cross-entropy weight 0.2 (as in the 40-epoch screen)
#   rank_lambda3   reconstruction weight 3
#   rank_lr1e-4    peak learning rate 1e-4 instead of 3e-4
#
# Test split: sealed during training (--validation-only-during-training). After
# training, the checkpoint with the lowest validation DockQ MSE (validated every
# 2,000 steps and at epoch ends) is reloaded and evaluated once on the 29 test
# targets: test_predictions.csv, a "test" block in prediction_metrics.json, and the
# reconstruction summary on test (reconstruction_summary.csv / _by_target.csv).
# Validation predictions from the same checkpoint are exported as before.
#
# CPU: main process (one PyTorch/BLAS thread) + 7 non-persistent spawn loader
# workers = 8 CPUs. Time: ~75 min/epoch measured on this recipe, so ~25 h of
# training plus test prediction and reconstruction evaluation.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/gate_run/full_pooling_repaired/model_lists}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PROJECT_DIR/gate_run/full_pooling_repaired/target_splits.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/rank_followup_full}"
EPOCHS=20
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

CONFIGS=(rank rank_capri rank_lambda3 rank_lr1e-4)
SEEDS=(7 17 27)
NAME="${CONFIGS[$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))]}"
SEED="${SEEDS[$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))]}"

LAMBDA=1; LR=3e-4; EXTRA_ARGS=()
case "$NAME" in
  rank)         ;;
  rank_capri)   EXTRA_ARGS=(--capri-head-weight 0.2) ;;
  rank_lambda3) LAMBDA=3 ;;
  rank_lr1e-4)  LR=1e-4 ;;
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
echo "lambda=$LAMBDA lr=$LR extra=${EXTRA_ARGS[*]:-(none)}; ranking loss + grouped batches in every run"
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
  --lr "$LR" \
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
  --reconstruction-lambda "$LAMBDA" \
  --dockq-range-weighting inverse-sqrt \
  --dockq-weight-exponent 0.5 \
  --dockq-range-diagnostics \
  --rank-loss-weight 0.1 \
  --rank-temperature 0.1 \
  --targets-per-batch 4 \
  --checkpoint-metric target_mse \
  --validation-only-during-training \
  --val-every-steps "$VAL_EVERY_STEPS" \
  --num-workers "$LOADER_WORKERS" \
  --worker-start-method spawn \
  --cpu-threads 1 \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
