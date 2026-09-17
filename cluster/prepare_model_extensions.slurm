#!/usr/bin/env bash
#SBATCH --job-name=prepare_extensions
#SBATCH --partition=pi_co54
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail
cd "$PROJECT_DIR"
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env
source "$PROJECT_DIR/venv/bin/activate"
# Each process uses one numerical-library thread. Training parallelism comes
# from DataLoader worker processes; preparation is parallel across array tasks.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
if [[ "$STAGE" == preflight ]]; then
  python preflight_model_extensions.py --data "$SUBSET_DIR" --output "$EXPERIMENT_DIR"
  COUNT=$(wc -l < "$EXPERIMENT_DIR/targets.txt")
  DEPENDENCY=()
  if [[ "${SKIP_PREP:-0}" == 1 ]]; then
    [[ -d "$EXTENSION_DATA" ]] || { echo "Prepared data missing: $EXTENSION_DATA" >&2; exit 2; }
  else
    # Prove one nonempty target works before releasing the rest of the array.
    FIRST=$(sbatch --parsable --array=1 --export=ALL,STAGE=data \
      --output="$EXPERIMENT_DIR/logs/data_%A_%a.out" cluster/prepare_model_extensions.slurm)
    REST=$(sbatch --parsable --array="2-$COUNT" --dependency="afterok:${FIRST%%;*}" --export=ALL,STAGE=data \
      --output="$EXPERIMENT_DIR/logs/data_%A_%a.out" cluster/prepare_model_extensions.slurm)
    DEPENDENCY+=(--dependency="afterok:${FIRST%%;*}:${REST%%;*}")
    echo "Single-target preparation: $FIRST; remaining targets: $REST"
  fi
  MATRIX=$(sbatch --parsable "${DEPENDENCY[@]}" --export=ALL,STAGE=matrix \
    --output="$EXPERIMENT_DIR/logs/matrix_%j.out" cluster/prepare_model_extensions.slurm)
  echo "Matrix validation and GPU submission: $MATRIX"
elif [[ "$STAGE" == data ]]; then
  TARGET=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT_DIR/targets.txt")
  EXTRA=()
  [[ "${ASSUME_AREA_WEIGHTED:-0}" == 1 ]] && EXTRA+=(--assume-area-weighted)
  [[ -n "${FEATURE_DATA:-}" ]] && EXTRA+=(--feature-data "$FEATURE_DATA")
  [[ -n "${EMBEDDING_TEMPLATE:-}" ]] && EXTRA+=(--embedding-template "$EMBEDDING_TEMPLATE")
  [[ -n "${FASTA_TEMPLATE:-}" ]] && EXTRA+=(--fasta-template "$FASTA_TEMPLATE")
  python prepare_model_extensions.py --target "$TARGET" --data "$SUBSET_DIR" \
    --output "$EXTENSION_DATA" --apbs-dir "$APBS_DIR" --pdb-root "$PDB_ROOT" \
    --esm-root "$ESM_ROOT" "${EXTRA[@]}"
elif [[ "$STAGE" == matrix ]]; then
  case "${RUN_MODE:-full}" in
    quick) SEEDS=(7) ;;
    full) SEEDS=(7 17 27) ;;
    *) echo "Unknown RUN_MODE" >&2; exit 2 ;;
  esac
  EXTRA=(--seeds "${SEEDS[@]}" --epochs "${EPOCHS:-50}")
  [[ -n "${PRIOR_SPLIT:-}" ]] && EXTRA+=(--prior-split "$PRIOR_SPLIT")
  python model_extension_experiments.py prepare --data "$EXTENSION_DATA" \
    --output "$EXPERIMENT_DIR" --targets-file "$EXPERIMENT_DIR/targets.txt" --optional-node-features-dir "$OPTIONAL_NODE_FEATURES_DIR" "${EXTRA[@]}"
  COUNT=$(python -c 'import json,os; print(len(json.load(open(os.path.join(os.environ["EXPERIMENT_DIR"], "matrix.json")))["runs"]))')
  GPU=$(sbatch --parsable --array="1-$COUNT" --output="$EXPERIMENT_DIR/logs/train_%A_%a.out" cluster/train_model_extensions.slurm)
  sbatch --dependency="afterok:${GPU%%;*}" --export=ALL,STAGE=compare \
    --output="$EXPERIMENT_DIR/logs/compare_%j.out" cluster/prepare_model_extensions.slurm
  echo "Submitted $COUNT GPU tasks: $GPU"
elif [[ "$STAGE" == compare ]]; then
  python compare_model_extensions.py --matrix "$EXPERIMENT_DIR/matrix.json" --output "$EXPERIMENT_DIR/comparison"
  python compare_model_extensions.py --matrix "$EXPERIMENT_DIR/matrix.json" --reference seed_replicate_base --output "$EXPERIMENT_DIR/comparison_vs_seed_base"
else
  echo "Unknown STAGE: $STAGE" >&2
  exit 2
fi
