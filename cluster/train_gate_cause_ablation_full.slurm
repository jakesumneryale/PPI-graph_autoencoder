#!/usr/bin/env bash
#SBATCH --job-name=gate_cause_ablation_full
#SBATCH --partition=pi_co54
#SBATCH --array=1-27
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=3-00:00:00

# Which setting removed the rising validation DockQ MSE? The feature ladder
# (L) rose from epoch 1-2; the anti-memorization control (A) did not. They
# differ in five training settings. This array switches them one at a time in
# both directions: each L_plus_* adds one A setting to the ladder recipe; each
# A_minus_* removes one from the control. The control itself is reused from
# gate_run/anti_memorization_full/A_control_seed*, whose recipe is unchanged.
#
#   setting           ladder (L)                      control (A)
#   loss weighting    adaptive shares .15/.15/.10/.60 fixed, reconstruction lambda 1
#   range weighting   none                            capped inverse-sqrt, alpha 0.5
#   pooling           whole graph                     whole graph + interface
#   dropout           0.1                             0.3
#   cosine schedule   50 epochs                       20 epochs
#
# Range weighting needs fixed weights (train_gate.py rejects it with shares), so
# A_minus_fixed also drops range weighting; A_minus_range separates the two.
# Every run trains 20 epochs. A "50-epoch schedule" run trains the first 20
# epochs of a 50-epoch cosine (--lr-schedule-epochs 50), i.e. the learning rate
# the ladder actually had over those epochs; the ladder's rise was clear by
# epoch 4-5. Measurement is identical everywhere: validation every 2,000 steps
# and at epoch ends, exact graph-average validation MSE, best-step checkpoint,
# no test evaluation. Data, features, model lists and split match the feature
# ladder's electrostatics rung.
#
# CPU allocation: main process (one PyTorch/BLAS thread) + 7 non-persistent
# DataLoader workers = 8 CPUs. 20-epoch runs of the same recipe finished within
# a day in gate_run/anti_memorization_full; the walltime leaves ample margin.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_co54/jas485/ppi_processed_graphs}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-$PROJECT_DIR/gate_run/full_pooling_repaired/model_lists}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PROJECT_DIR/gate_run/full_pooling_repaired/target_splits.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gate_run/cause_ablation_full}"
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

#        name               weighting range pooling  dropout schedule_epochs
CONFIGS=("L_ladder          shares    none  all      0.1     50"
         "L_plus_fixed      fixed     none  all      0.1     50"
         "L_plus_dropout    shares    none  all      0.3     50"
         "L_plus_sched20    shares    none  all      0.1     20"
         "A_minus_fixed     shares    none  combined 0.3     20"
         "A_minus_dropout   fixed     inv   combined 0.1     20"
         "A_minus_sched     fixed     inv   combined 0.3     50"
         "A_minus_range     fixed     none  combined 0.3     20"
         "A_minus_combined  fixed     inv   all      0.3     20")
SEEDS=(7 17 27)
read -r NAME WEIGHTING RANGE POOLING DROPOUT SCHEDULE_EPOCHS \
  <<< "${CONFIGS[$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))]}"
SEED="${SEEDS[$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))]}"

case "$WEIGHTING" in
  shares) WEIGHT_ARGS=(--loss-weight-mode shares --node-share 0.15 --edge-attr-share 0.15
                       --edge-presence-share 0.10 --target-share 0.60) ;;
  fixed)  WEIGHT_ARGS=(--loss-weight-mode fixed --reconstruction-lambda 1) ;;
  *) echo "Unknown weighting: $WEIGHTING" >&2; exit 2 ;;
esac
case "$RANGE" in
  none) RANGE_ARGS=(--dockq-range-weighting none) ;;
  inv)  RANGE_ARGS=(--dockq-range-weighting inverse-sqrt --dockq-weight-exponent 0.5) ;;
  *) echo "Unknown range weighting: $RANGE" >&2; exit 2 ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/${NAME}_seed${SEED}"
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
echo "Array task $SLURM_ARRAY_TASK_ID: config=$NAME seed=$SEED"
echo "weighting=$WEIGHTING range=$RANGE pooling=$POOLING dropout=$DROPOUT schedule_epochs=$SCHEDULE_EPOCHS (train $EPOCHS)"
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
  --lr-schedule-epochs "$SCHEDULE_EPOCHS" \
  --warmup-steps 1000 \
  --hidden-dim 64 \
  --latent-dim 32 \
  --gat-heads 4 \
  --dropout "$DROPOUT" \
  --residual_connections \
  --pooling "$POOLING" \
  --node-features "aa_type,chain,interface_nodes,interface_node_degree,rsasa_i_node" \
  --edge-features "interface_edges,ca_dist,voronoi_contact_area,voronoi_contact_missing,v_es" \
  --edge-recon-features "interface_edges,ca_dist" \
  --edge-feature-transforms "voronoi_contact_area=log1p" \
  --standardize-edge-features "voronoi_contact_area,v_es" \
  --edge-stats-seed 7 \
  "${WEIGHT_ARGS[@]}" \
  "${RANGE_ARGS[@]}" \
  --dockq-range-diagnostics \
  --checkpoint-metric target_mse \
  --no-test-evaluation \
  --val-every-steps "$VAL_EVERY_STEPS" \
  --num-workers "$LOADER_WORKERS" \
  --worker-start-method spawn \
  --cpu-threads 1 \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"
